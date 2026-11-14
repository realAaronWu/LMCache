# SPDX-License-Identifier: Apache-2.0
"""CPU HTTP contract tests; mocked CUDA receipts are not GPU evidence."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import ModuleType
import hashlib
import importlib.util
import json
import time

# Third Party
import pytest


@pytest.fixture
def smoke() -> ModuleType:
    """Load the standalone benchmark without importing LMCache GPU fixtures."""
    # test-import: allow -- standalone benchmark is not an installed package
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/pooled_kv_routing/gpu_smoke.py"
    )
    assert path.is_file(), "standalone GPU smoke CLI is missing"
    spec = importlib.util.spec_from_file_location("gpu_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stream(mode: str) -> bytes:
    """Build protocol examples, including distinct invalid acceptance paths."""
    choice = {
        "index": 0,
        "text": "ok",
        "token_ids": [7, 8],
        "prompt_token_ids": [1, 2, 3],
        "finish_reason": "length",
    }
    if mode in {"omitted_prompt", "top_level_prompt", "late_prompt"}:
        del choice["prompt_token_ids"]
    if mode == "null_prompt":
        choice["prompt_token_ids"] = None
    if mode == "wrong_prompt":
        choice["prompt_token_ids"] = [1, 2, 4]
    if mode == "bool_prompt":
        choice["prompt_token_ids"] = [True, 2, 3]
    if mode == "mismatch":
        choice["token_ids"] = [7, 9]
    if mode == "missing_tokens":
        del choice["token_ids"]
    if mode == "bad_type":
        choice["token_ids"] = [True, 8]
    if mode == "bad_finish":
        choice["finish_reason"] = "error"
    event = {
        "id": "req",
        "model": "tiny",
        "object": "text_completion",
        "created": 123,
        "choices": [choice],
    }
    usage: dict[str, object] = {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    if mode == "bad_usage":
        usage["completion_tokens"] = 3
    if mode == "bool_usage":
        usage["completion_tokens"] = True
    tail = {
        "id": "req",
        "model": "tiny",
        "object": "text_completion",
        "created": 123,
        "choices": [],
        "usage": usage,
    }
    events = [event, tail]
    if mode in {"split", "later_wrong_prompt"}:
        choice["token_ids"] = [7]
        choice["text"] = "o"
        choice["finish_reason"] = None
        events.insert(
            1,
            {
                **event,
                "choices": [
                    {
                        **choice,
                        "token_ids": [8],
                        "text": "k",
                        "finish_reason": "length",
                        "prompt_token_ids": [1, 2, 4]
                        if mode == "later_wrong_prompt"
                        else None,
                    }
                ],
            },
        )
    if mode == "top_level_prompt":
        event["prompt_token_ids"] = [1, 2, 3]
    if mode == "late_prompt":
        choice["finish_reason"] = None
        events.insert(
            1,
            {
                **event,
                "choices": [
                    {
                        **choice,
                        "prompt_token_ids": [1, 2, 3],
                        "token_ids": [],
                        "finish_reason": "length",
                    }
                ],
            },
        )
    if mode == "attached_error":
        event["error"] = {"message": "failure"}
    if mode == "null_error":
        event["error"] = None
    if mode == "choice_error":
        choice["error"] = "failure"
    if mode == "usage_error":
        usage["error"] = "failure"
    if mode == "late_error":
        events.append({**tail, "usage": None, "error": {"message": "late failure"}})
    if mode == "bad_object":
        event["object"] = []
    if mode == "bad_created":
        event["created"] = "yesterday"
    if mode == "bool_created":
        event["created"] = True
    if mode == "missing_object":
        del event["object"]
    if mode == "missing_created":
        del event["created"]
    if mode == "no_usage":
        events.pop()
    if mode == "wrong_model":
        event["model"] = "other"
    if mode == "duplicate_finish":
        events.insert(1, event)
    data = b"".join(b"data: " + json.dumps(e).encode() + b"\r\n\r\n" for e in events)
    if mode != "no_done":
        data += b"data: [DONE]\r\n\r\n"
    if mode == "after_done":
        data += b"data: {}\n\n"
    if mode == "malformed":
        data = b"data: {oops}\n\n"
    if mode == "nonfinite":
        data = data.replace(b'"index": 0', b'"index": 0, "extra": 1e999')
    if mode == "oversize":
        data = b"x" * (4 * 1024 * 1024 + 1)
    return data


@contextmanager
def endpoint(
    mode: str = "ok", *, role: str = "direct"
) -> Iterator[tuple[str, list[dict[str, object]]]]:
    """Serve HTTP; proxy requires an ID and direct rejects proxy-only metadata."""
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            body: dict[str, object]
            if self.path == "/v1/models":
                body = {
                    "data": [
                        {"id": "tiny", "root": None if mode == "no_model" else "tiny"}
                    ]
                }
            else:
                body = {
                    "vllm_config": {
                        "model_config": {
                            "model": None if mode == "no_model" else "tiny"
                        },
                        "device_config": {
                            "device": "cpu" if mode == "no_device" else "cuda"
                        },
                        "parallel_config": {
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "data_parallel_size": 1,
                        },
                    }
                }
            self.send_response(404 if mode == "missing_endpoint" else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_POST(self) -> None:
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            rid = requests[-1].get("request_id")
            valid_id = (
                isinstance(rid, str)
                and 1 <= len(rid) <= 128
                and rid.isascii()
                and all(33 <= ord(c) <= 126 for c in rid)
            )
            if (role == "proxy" and not valid_id) or (
                role == "direct" and "request_id" in requests[-1]
            ):
                self.send_response(422)
                self.end_headers()
                self.wfile.write(b'{"error":"request_id contract violation"}')
                return
            self.send_response(500 if mode == "http_error" else 200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                if mode == "timeout":
                    self.wfile.write(b": partial\n\n")
                    self.wfile.flush()
                    time.sleep(0.3)
                self.wfile.write(stream(mode))
                if mode == "hang_after_done":
                    self.wfile.flush()
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def invoke(
    smoke: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    direct: str,
    proxy: str,
    *extra: str,
) -> tuple[int, dict[str, object]]:
    """Exercise CLI orchestration with an explicitly test-only CUDA stub."""
    monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(smoke.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(smoke.torch.cuda, "get_device_name", lambda _: "CPU test stub")
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[[1, 2, 3]]")
    out = tmp_path / "evidence"
    code = smoke.main(
        [
            "--direct-endpoint",
            direct,
            "--proxy-endpoint",
            proxy,
            "--model",
            "tiny",
            "--corpus-token-ids",
            str(corpus),
            "--output-dir",
            str(out),
            "--max-tokens",
            "2",
            *extra,
        ]
    )
    return code, json.loads((out / "evidence.json").read_text())


@pytest.mark.parametrize("mode", ["ok", "split"])
def test_complete_direct_proxy(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    corpus = tmp_path / "repeated-prompts.json"
    corpus.write_text("[[1, 2, 3], [1, 2, 3]]")
    with (
        endpoint(mode) as (direct, requests),
        endpoint(mode, role="proxy") as (proxy, proxy_requests),
    ):
        code, report = invoke(
            smoke,
            monkeypatch,
            tmp_path,
            direct,
            proxy,
            "--corpus-token-ids",
            str(corpus),
        )
    assert code == 0
    assert report["status"] == "PASS"
    assert report["scope"] == "single_gpu_serving_only"
    assert report["native_lmcache_p2p"] == "NOT RUN"
    assert report["performance_gt_5_percent"] == "NOT RUN"
    expected_run_id = hashlib.sha256(
        str((tmp_path / "evidence").resolve()).encode()
    ).hexdigest()[:24]
    assert report["run_id"] == expected_run_id
    assert "request_id" not in requests[0]
    assert proxy_requests[0] == {
        **requests[0],
        "request_id": f"gpu-smoke-{expected_run_id}-0",
    }
    assert len(requests) == len(proxy_requests) == 2
    assert "request_id" not in requests[1]
    assert proxy_requests[1] == {
        **requests[1],
        "request_id": f"gpu-smoke-{expected_run_id}-1",
    }
    assert requests[0]["prompt"] == [1, 2, 3]
    assert requests[0]["temperature"] == 0
    assert requests[0]["return_token_ids"] is True
    evidence = json.loads((tmp_path / "evidence" / "evidence.json").read_text())
    case = evidence["cases"][0]
    result = case["direct"]
    assert case["request"] == requests[0]
    assert case["wire_requests"] == {"direct": requests[0], "proxy": proxy_requests[0]}
    second_case = evidence["cases"][1]
    assert second_case["wire_requests"] == {
        "direct": requests[1],
        "proxy": proxy_requests[1],
    }
    assert result["token_ids"] == [7, 8]
    assert 0 <= result["ttft_seconds"] <= result["elapsed_seconds"]
    assert (tmp_path / "evidence" / "0-proxy.sse").read_bytes().endswith(b"\r\n\r\n")


@pytest.mark.parametrize(
    "mode",
    [
        "no_done",
        "no_usage",
        "bad_usage",
        "bool_usage",
        "missing_tokens",
        "bad_type",
        "bad_finish",
        "wrong_model",
        "duplicate_finish",
        "after_done",
        "malformed",
        "oversize",
        "http_error",
        "mismatch",
        "timeout",
        "nonfinite",
        "omitted_prompt",
        "null_prompt",
        "wrong_prompt",
        "bool_prompt",
        "top_level_prompt",
        "late_prompt",
        "attached_error",
        "null_error",
        "choice_error",
        "usage_error",
        "late_error",
        "bad_object",
        "bad_created",
        "bool_created",
        "hang_after_done",
        "missing_object",
        "missing_created",
        "later_wrong_prompt",
    ],
)
def test_invalid_stream_retains_evidence(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    with endpoint() as (direct, _), endpoint(mode, role="proxy") as (proxy, _):
        extra = (
            ("--deadline-seconds", "0.1")
            if mode in {"timeout", "hang_after_done"}
            else ()
        )
        code, report = invoke(smoke, monkeypatch, tmp_path, direct, proxy, *extra)
    assert code != 0
    assert report["status"] == "FAIL"
    assert report["error"]
    case = json.loads((tmp_path / "evidence" / "evidence.json").read_text())["cases"][0]
    arm = case["proxy"]
    assert arm["status"] == ("PASS" if mode == "mismatch" else "FAIL")
    assert arm["http_status"] == (500 if mode == "http_error" else 200)
    assert arm["started_unix_ns"] > 0
    assert arm["started_monotonic_seconds"] > 0
    assert arm["elapsed_seconds"] >= 0
    assert (
        arm["ttft_seconds"] is None
        or 0 <= arm["ttft_seconds"] <= arm["elapsed_seconds"]
    )
    if mode in {"bad_usage", "hang_after_done"}:
        assert arm["ttft_seconds"] is not None
        assert arm["token_ids"] == [7, 8]
    if mode != "mismatch":
        assert arm["error"]
    assert (tmp_path / "evidence" / "0-proxy.sse").stat().st_size > 0
    assert (tmp_path / "evidence" / "0-proxy.sse").stat().st_size <= 4 * 1024 * 1024


@pytest.mark.parametrize("mode", ["no_device", "missing_endpoint", "no_model"])
def test_missing_receipt_refused(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    with endpoint(mode) as (direct, requests), endpoint(role="proxy") as (proxy, _):
        code, report = invoke(smoke, monkeypatch, tmp_path, direct, proxy)
    assert code == 1
    assert report["status"] == "FAIL"
    assert not requests


def test_cli_queries_local_cuda(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: False)
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[[1]]")
    out = tmp_path / "out"
    code = smoke.main(
        [
            "--direct-endpoint",
            "http://127.0.0.1:1",
            "--proxy-endpoint",
            "http://127.0.0.1:2",
            "--model",
            "tiny",
            "--corpus-token-ids",
            str(corpus),
            "--output-dir",
            str(out),
        ]
    )
    assert code == 1
    assert "CUDA" in json.loads((out / "evidence.json").read_text())["error"]


@pytest.mark.parametrize(
    "extra",
    [
        ("--deadline-seconds", "nan"),
        ("--deadline-seconds", "inf"),
        ("--max-tokens", "0"),
        ("--direct-endpoint", "http://user:secret@localhost"),
        ("--proxy-endpoint", ""),
        ("--proxy-endpoint", "http://localhost/v1"),
    ],
)
def test_invalid_cli_inputs(
    smoke: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: tuple[str, str],
) -> None:
    code, report = invoke(
        smoke, monkeypatch, tmp_path, "http://127.0.0.1:1", "http://127.0.0.1:2", *extra
    )
    assert code == 1
    assert report["status"] == "FAIL"


def test_existing_output_not_overwritten(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invoke(smoke, monkeypatch, tmp_path, "bad", "bad")
    evidence = tmp_path / "evidence" / "evidence.json"
    original = evidence.read_bytes()
    with pytest.raises(SystemExit):
        invoke(smoke, monkeypatch, tmp_path, "bad", "bad")
    assert evidence.read_bytes() == original
