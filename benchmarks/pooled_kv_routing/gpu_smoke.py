# SPDX-License-Identifier: Apache-2.0
"""Single-GPU functional serving check, not LMCache/P2P or performance qualification.

Run this standalone script on the CUDA host. Both endpoints are HTTP origins
(no /v1 suffix); only the direct endpoint needs /v1/models and
/server_info?config_format=json. Corpus JSON is a list of nonempty token-ID
lists. The output directory must not exist. Each case sends the same deterministic
vLLM /v1/completions generation payload directly and via the proxy. Only the proxy
wire request adds request_id: gpu-smoke-<24 hex output-directory hash>-<case index>.
Raw bounded SSE bytes,
requests, client-side timing, filtered endpoint receipts and failure diagnostics
are retained. No proxy parser or sibling benchmark module is imported.
"""

# Standard
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit
import argparse
import asyncio
import hashlib
import json
import math
import time

# Third Party
import httpx
import torch

MAX_BYTES = 4 * 1024 * 1024
MAX_PROMPTS = 64
MAX_TOKENS = 32768


def _object(value: object) -> dict[str, object]:
    """Validate an untrusted JSON object; raise ValueError on other types."""
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return cast(dict[str, object], value)


def _reject_constant(value: str) -> None:
    """Reject non-finite JSON extensions rather than accepting NaN/Infinity."""
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(raw: str | bytes) -> object:
    """Decode strict finite JSON, raising ValueError on malformed input."""

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    return json.loads(raw, parse_constant=_reject_constant, parse_float=finite_float)


def _ids(value: object, limit: int) -> list[int]:
    """Validate bounded token IDs; booleans and out-of-range IDs are rejected."""
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError("token_ids must be a bounded list")
    if any(type(v) is not int or not 0 <= v < 2**31 for v in value):
        raise ValueError("token_ids must contain nonnegative int32 IDs")
    return cast(list[int], value)


def _origin(value: str) -> str:
    """Require HTTP origins without credentials, paths, queries or fragments."""
    url = urlsplit(value)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("endpoint must be an HTTP origin without credentials")
    if url.port == 0:
        raise ValueError("endpoint port must be positive")
    return value.rstrip("/")


async def _receipt(client: httpx.AsyncClient, origin: str, path: str) -> object:
    """Read one bounded HTTP JSON receipt; caller enforces the total deadline."""
    async with client.stream("GET", origin + path) as response:
        response.raise_for_status()
        raw = bytearray()
        async for chunk in response.aiter_bytes():
            if len(raw) + len(chunk) > MAX_BYTES:
                raise ValueError("endpoint receipt exceeds size limit")
            raw.extend(chunk)
        return _json(bytes(raw))


async def _stream(
    client: httpx.AsyncClient,
    origin: str,
    request: dict[str, object],
    raw_path: Path,
    observation: dict[str, object],
) -> dict[str, object]:
    """Observe bounded SSE independently of the proxy; reject incomplete streams."""
    started = cast(float, observation["started_monotonic_seconds"])
    tokens: list[int] = []
    observation["token_ids"] = tokens
    prompt_seen = False
    text = ""
    finish: str | None = None
    usage: dict[str, object] | None = None
    first_token: float | None = None
    done = False
    request_id: str | None = None
    pending = b""
    event_lines: list[str] = []
    size = 0
    with raw_path.open("xb") as raw:
        async with client.stream(
            "POST", origin + "/v1/completions", json=request
        ) as response:
            observation["http_status"] = response.status_code
            # Retain error bodies as well as partial successful streams.
            async for chunk in response.aiter_bytes():
                raw.write(chunk[: MAX_BYTES - size])
                raw.flush()
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("stream exceeds size limit")
                if response.status_code != 200:
                    continue
                if (
                    response.headers.get("content-type", "").split(";")[0]
                    != "text/event-stream"
                ):
                    raise ValueError("expected text/event-stream")
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    decoded = line.rstrip(b"\r").decode("utf-8", errors="strict")
                    if decoded.startswith(":"):
                        continue
                    if decoded:
                        if not decoded.startswith("data:"):
                            raise ValueError("unsupported SSE field")
                        event_lines.append(decoded[5:].lstrip(" "))
                        continue
                    if not event_lines:
                        continue
                    data = "\n".join(event_lines)
                    event_lines.clear()
                    if done:
                        raise ValueError("data after DONE")
                    if data == "[DONE]":
                        done = True
                        observation["done"] = True
                        continue
                    event = _object(_json(data))
                    if "error" in event:
                        raise ValueError("error-bearing completion envelope")
                    if (
                        event.get("object") != "text_completion"
                        or type(event.get("created")) is not int
                        or cast(int, event["created"]) < 0
                    ):
                        raise ValueError("invalid completion envelope metadata")
                    if event.get("model") != request["model"]:
                        raise ValueError("response model mismatch")
                    identity = event.get("id")
                    if not isinstance(identity, str) or not identity:
                        raise ValueError("missing response id")
                    if request_id is not None and identity != request_id:
                        raise ValueError("response id drift")
                    request_id = identity
                    choices = event.get("choices")
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise ValueError("expected at most one choice")
                    if choices:
                        if finish is not None or usage is not None:
                            raise ValueError("choice after finish/usage")
                        choice = _object(choices[0])
                        if "error" in choice:
                            raise ValueError("error-bearing completion choice")
                        if type(choice.get("index")) is not int or choice["index"] != 0:
                            raise ValueError("invalid choice index")
                        delta = _ids(
                            choice.get("token_ids"), cast(int, request["max_tokens"])
                        )
                        if delta and first_token is None:
                            first_token = time.perf_counter() - started
                            observation["ttft_seconds"] = first_token
                        tokens.extend(delta)
                        if len(tokens) > cast(int, request["max_tokens"]):
                            raise ValueError("too many completion tokens")
                        piece = choice.get("text")
                        if not isinstance(piece, str):
                            raise ValueError("missing text")
                        text += piece
                        reason = choice.get("finish_reason")
                        if reason is not None:
                            if reason not in {"stop", "length"}:
                                raise ValueError("invalid finish_reason")
                            finish = cast(str, reason)
                            observation["finish_reason"] = finish
                        prompt_ids = choice.get("prompt_token_ids")
                        if not prompt_seen and prompt_ids is None:
                            raise ValueError(
                                "first choice must include prompt_token_ids"
                            )
                        if (
                            prompt_ids is not None
                            and _ids(prompt_ids, MAX_TOKENS) != request["prompt"]
                        ):
                            raise ValueError("prompt token IDs mismatch")
                        if prompt_ids is not None:
                            prompt_seen = True
                            observation["prompt_token_ids"] = prompt_ids
                    if event.get("usage") is not None:
                        if usage is not None or finish is None:
                            raise ValueError("duplicate or premature usage")
                        usage = _object(event["usage"])
                        if "error" in usage:
                            raise ValueError("error-bearing completion usage")
                        observation["usage"] = usage
            response.raise_for_status()
    if (
        pending
        or event_lines
        or not done
        or not tokens
        or not prompt_seen
        or finish is None
        or usage is None
    ):
        raise ValueError(
            "incomplete stream: require token_ids, finish_reason, usage and DONE"
        )
    expected = {
        "prompt_tokens": len(cast(list[int], request["prompt"])),
        "completion_tokens": len(tokens),
        "total_tokens": len(cast(list[int], request["prompt"])) + len(tokens),
    }
    if any(type(usage.get(k)) is not int or usage[k] != v for k, v in expected.items()):
        raise ValueError("usage/count mismatch")
    if finish == "length" and len(tokens) != request["max_tokens"]:
        raise ValueError("length finish before requested max_tokens")
    return {
        "token_ids": tokens,
        "text": text,
        "finish_reason": finish,
        "usage": expected,
        "done": done,
        "response_id": request_id,
        "ttft_seconds": first_token,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_file": raw_path.name,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }


async def _serve(
    args: argparse.Namespace, corpus: list[list[int]], report: dict[str, object]
) -> None:
    """Collect endpoint receipts and all direct/proxy cases under one deadline."""
    async with httpx.AsyncClient(
        timeout=args.deadline_seconds, trust_env=False, follow_redirects=False
    ) as client:
        models = _object(await _receipt(client, args.direct_endpoint, "/v1/models"))
        entries = models.get("data")
        if not isinstance(entries, list):
            raise ValueError("missing model receipt")
        selected = [
            e for e in entries if isinstance(e, dict) and e.get("id") == args.model
        ]
        if len(selected) != 1:
            raise ValueError("model absent or duplicated in endpoint receipt")
        info = _object(
            await _receipt(
                client, args.direct_endpoint, "/server_info?config_format=json"
            )
        )
        config = _object(info.get("vllm_config"))
        device = _object(config.get("device_config")).get("device")
        if device not in {"cuda", "cuda:0"}:
            raise ValueError("endpoint CUDA device receipt missing")
        model_config = _object(config.get("model_config"))
        configured_model = model_config.get("model")
        if (
            not isinstance(configured_model, str)
            or not configured_model
            or configured_model not in {args.model, selected[0].get("root")}
        ):
            raise ValueError("endpoint model config mismatch")
        parallel = _object(config.get("parallel_config"))
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        ):
            if type(parallel.get(name)) is not int or parallel[name] != 1:
                raise ValueError("single-GPU parallelism receipt missing")
        # Do not persist server_info environment variables or arbitrary config secrets.
        report["endpoint_receipt"] = {
            "model": args.model,
            "root": selected[0].get("root"),
            "configured_model": model_config["model"],
            "device": device,
            "parallelism": {
                k: parallel[k]
                for k in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                )
            },
        }
        cases: list[dict[str, object]] = []
        report["cases"] = cases
        for index, prompt in enumerate(corpus):
            request: dict[str, object] = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "seed": 0,
                "n": 1,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
                "add_special_tokens": False,
            }
            case: dict[str, object] = {"request": request, "status": "FAIL"}
            wire_requests: dict[str, object] = {}
            case["wire_requests"] = wire_requests
            cases.append(case)
            for name, origin in (
                ("direct", args.direct_endpoint),
                ("proxy", args.proxy_endpoint),
            ):
                outgoing = dict(request)
                if name == "proxy":
                    outgoing["request_id"] = f"gpu-smoke-{report['run_id']}-{index}"
                wire_requests[name] = outgoing
                observation: dict[str, object] = {
                    "status": "FAIL",
                    "started_unix_ns": time.time_ns(),
                    "started_monotonic_seconds": time.perf_counter(),
                    "ttft_seconds": None,
                    "http_status": None,
                    "done": False,
                    "raw_file": f"{index}-{name}.sse",
                }
                case[name] = observation
                try:
                    result = await _stream(
                        client,
                        origin,
                        outgoing,
                        args.output_dir / f"{index}-{name}.sse",
                        observation,
                    )
                    observation.update(result)
                    observation["status"] = "PASS"
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    OSError,
                    RuntimeError,
                    httpx.HTTPError,
                    asyncio.CancelledError,
                ) as exc:
                    observation["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    observation["elapsed_seconds"] = time.perf_counter() - cast(
                        float, observation["started_monotonic_seconds"]
                    )
            direct, proxy = _object(case["direct"]), _object(case["proxy"])
            for key in ("token_ids", "text", "finish_reason", "usage"):
                if direct[key] != proxy[key]:
                    raise ValueError(f"direct/proxy {key} mismatch in case {index}")
            case["status"] = "PASS"


def _cuda_receipt() -> dict[str, object]:
    """Query actual local CUDA without allocating tensors or claiming route mapping."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ValueError("actual local torch CUDA device unavailable")
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and persist evidence; return 0 only for complete functional success.

    Args:
        argv: CLI arguments, or None for sys.argv. See --help for required inputs.
    Returns:
        Zero on success; one on validation/network/CUDA failure.
    Raises:
        SystemExit: argparse errors or an existing/unwritable output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-endpoint", required=True)
    parser.add_argument("--proxy-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus-token-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--deadline-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"output directory must be new and writable: {exc}")
    report: dict[str, object] = {
        "schema_version": 1,
        "run_id": hashlib.sha256(str(args.output_dir.resolve()).encode()).hexdigest()[
            :24
        ],
        "status": "FAIL",
        "scope": "single_gpu_serving_only",
        "native_lmcache_p2p": "NOT RUN",
        "performance_gt_5_percent": "NOT RUN",
        "cuda": "NOT RUN",
        "cases": [],
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        args.direct_endpoint = _origin(args.direct_endpoint)
        args.proxy_endpoint = _origin(args.proxy_endpoint)
        if args.direct_endpoint == args.proxy_endpoint:
            raise ValueError("direct and proxy endpoints must differ")
        if not args.model.strip() or len(args.model) > 4096:
            raise ValueError("model ID must be nonempty and bounded")
        if not 1 <= args.max_tokens <= 4096:
            raise ValueError("max_tokens must be in [1, 4096]")
        if (
            not math.isfinite(args.deadline_seconds)
            or not 0 < args.deadline_seconds <= 3600
        ):
            raise ValueError("deadline_seconds must be finite in (0, 3600]")
        with args.corpus_token_ids.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("corpus exceeds size limit")
        value = _json(raw)
        if not isinstance(value, list) or not 1 <= len(value) <= MAX_PROMPTS:
            raise ValueError("corpus must contain 1..64 prompts")
        corpus = [_ids(v, MAX_TOKENS) for v in value]
        if any(not prompt for prompt in corpus):
            raise ValueError("empty prompt")
        report["inputs"] = {
            "model": args.model,
            "corpus_token_ids": corpus,
            "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "direct_endpoint": args.direct_endpoint,
            "proxy_endpoint": args.proxy_endpoint,
            "deadline_seconds": args.deadline_seconds,
            "max_tokens": args.max_tokens,
        }
        report["cuda"] = _cuda_receipt()

        async def run() -> None:
            await asyncio.wait_for(_serve(args, corpus, report), args.deadline_seconds)

        asyncio.run(run())
        report["status"] = "PASS"
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        httpx.HTTPError,
        TimeoutError,
        UnicodeError,
    ) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    except KeyboardInterrupt:
        report["error"] = "KeyboardInterrupt"
    finally:
        report["raw_streams"] = [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(args.output_dir.glob("*.sse"))
        ]
        (args.output_dir / "evidence.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
