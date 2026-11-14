# SPDX-License-Identifier: Apache-2.0
"""Standalone token-addressed streaming proxy with fail-closed CPU evidence.

HTTP close is not worker cancellation acknowledgement. Any dispatched request
without complete terminal evidence permanently taints its worker in this app;
restart only after the operator has independently confirmed worker termination.
Journals are exclusive, fsynced JSONL files, not a recovery/resume database.
"""

# Standard
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
import argparse
import asyncio
import base64
import copy
import json
import logging
import math
import os
import time

# Third Party
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send
import anyio
import httpx
import uvicorn

logger = logging.getLogger(__name__)
LoadMode = Literal["native_load", "skip_external"]
FinishStatus = Literal[
    "success", "error", "timeout", "cancelled", "worker_tainted", "overload"
]


def _reject_constant(value: str) -> object:
    """Reject non-JSON numeric constants while decoding wire evidence."""
    raise ValueError(f"non-finite number: {value}")


def _finite(value: object) -> None:
    """Reject overflowed floats recursively, including otherwise unused fields."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def _object(raw: bytes) -> dict[str, object]:
    """Decode a finite JSON object; raise ValueError on invalid evidence."""
    value = json.loads(raw, parse_constant=_reject_constant)
    _finite(value)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _integer(value: object, minimum: int = 0) -> int:
    """Validate an integer without accepting Python's bool-as-int coercion."""
    if type(value) is not int or value < minimum:
        raise ValueError("expected bounded integer")
    return value


def _validate(body: dict[str, object], limits: "ProxyLimits") -> str:
    """Validate the single-output token request before calling the router."""
    rid = body.get("request_id")
    if (
        not isinstance(rid, str)
        or not rid
        or len(rid) > 128
        or not rid.isascii()
        or any(c.isspace() or ord(c) < 33 for c in rid)
    ):
        raise ValueError("request_id must be 1..128 printable ASCII characters")
    prompt = body.get("prompt")
    if not isinstance(prompt, list) or not 1 <= len(prompt) <= limits.max_input_tokens:
        raise ValueError("prompt must be a bounded nonempty token-ID list")
    for token in prompt:
        if _integer(token) > 2**31 - 1:
            raise ValueError("token ID exceeds int32")
    if _integer(body.get("max_tokens"), 1) > limits.max_output_tokens:
        raise ValueError("max_tokens exceeds limit")
    if body.get("stream") is not True or _integer(body.get("n"), 1) != 1:
        raise ValueError("stream=true and n=1 required")
    if "arrival_ns" in body:
        _integer(body["arrival_ns"])
    if body.get("echo", False) is not False or body.get("best_of", 1) != 1:
        raise ValueError("echo and best_of are unsupported")
    if "extra_args" in body:
        raise ValueError("use top-level kv_transfer_params, not extra_args")
    if not isinstance(body.get("kv_transfer_params", {}), dict):
        raise ValueError("kv_transfer_params must be an object")
    return rid


@asynccontextmanager
async def _worker_stream(
    url: str,
    outgoing: dict[str, object],
    rid: str,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None,
) -> AsyncIterator[httpx.Response]:
    """Open one worker stream; shield both close operations against disconnect."""
    client = httpx.AsyncClient(
        transport=transport, timeout=timeout, follow_redirects=False, trust_env=False
    )
    response: httpx.Response | None = None
    try:
        response = await client.send(
            client.build_request(
                "POST", url, json=outgoing, headers={"X-Request-Id": rid}
            ),
            stream=True,
        )
        response.stream = _ShieldedWorkerStream(
            cast(httpx.AsyncByteStream, response.stream), timeout
        )
        yield response
    finally:
        with anyio.CancelScope(shield=True):
            try:
                if response is not None:
                    with anyio.fail_after(timeout):
                        await response.aclose()
            finally:
                with anyio.fail_after(timeout):
                    await client.aclose()


@dataclass(frozen=True)
class ProxyRoute:
    """Router result: worker identity, full HTTP endpoint, mode, journal metadata."""

    worker_id: str
    url: str
    load_mode: LoadMode
    metadata: dict[str, object]


@dataclass(frozen=True)
class ProxyFinish:
    """Completion notification; false confirmation must retain worker accounting."""

    request_id: str
    worker_id: str
    status: FinishStatus
    worker_terminal_confirmed: bool


@dataclass(frozen=True)
class ProxyLimits:
    """Hard bounds per app/request; timeout covers routing and full streaming.

    max_requests bounds remembered IDs and journal growth; start a fresh app and
    journal for each arm. No IDs are evicted or retried. All bounds are positive.
    """

    max_body_bytes: int = 1_048_576
    max_input_tokens: int = 65_536
    max_output_tokens: int = 4096
    max_stream_bytes: int = 16_777_216
    max_event_bytes: int = 1_048_576
    max_requests: int = 100_000
    max_inflight: int = 1024
    request_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        """Reject unusable bounds at app construction."""
        for value in (
            self.max_body_bytes,
            self.max_input_tokens,
            self.max_output_tokens,
            self.max_stream_bytes,
            self.max_event_bytes,
            self.max_requests,
            self.max_inflight,
        ):
            _integer(value, 1)
        if (
            isinstance(self.request_timeout_s, bool)
            or not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0
        ):
            raise ValueError("timeout must be finite and positive")


_DEFAULT_LIMITS = ProxyLimits()


class _ShieldedWorkerStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, timeout: float) -> None:
        self.stream = stream
        self.timeout = timeout

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Preserve the worker's observed delivery batch boundaries."""
        async for chunk in self.stream:
            yield chunk

    async def aclose(self) -> None:
        """Shield httpx's automatic EOF close as well as explicit cleanup."""
        with anyio.CancelScope(shield=True):
            with anyio.fail_after(self.timeout):
                await self.stream.aclose()


class _OwnedStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncGenerator[bytes, None],
        cleanup: Callable[[], Awaitable[None]],
        rid: str,
        deadline: float,
    ) -> None:
        super().__init__(
            content, media_type="text/event-stream", headers={"X-Request-Id": rid}
        )
        self.content = content
        self.cleanup = cleanup
        self.deadline = deadline

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Close the producer even when downstream fails before its first read."""

        async def bounded_send(message: dict[str, object]) -> None:
            with anyio.fail_after(max(0, self.deadline - time.monotonic())):
                await send(message)

        try:
            await super().__call__(scope, receive, bounded_send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.content.aclose()
                finally:
                    await self.cleanup()


class _Journal:
    def __init__(self, path: Path) -> None:
        # Exclusive creation prevents accidentally mixing evidence across runs.
        with path.open("x", encoding="utf-8") as file:
            file.flush()
            os.fsync(file.fileno())
        self.path = path

    def record(self, event: str, request_id: str, **fields: object) -> None:
        line = json.dumps(
            {
                "event": event,
                "request_id": request_id,
                "monotonic_ns": time.monotonic_ns(),
                **fields,
            },
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())


class _Observer:
    def __init__(self, prompt_count: int, max_tokens: int, event_limit: int) -> None:
        self.prompt_count = prompt_count
        self.max_tokens = max_tokens
        self.event_limit = event_limit
        self.buffer = b""
        self.data: list[bytes] = []
        self.event_size = 0
        self.tokens: list[int] = []
        self.finish_reason: str | None = None
        self.usage: dict[str, object] | None = None
        self.done = False

    def feed(self, raw: bytes) -> Iterator[bytes]:
        """Observe events, returning wire frames except the withheld DONE marker."""
        self.buffer += raw
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.removesuffix(b"\r")
            self.event_size += len(line) + 1
            if self.event_size > self.event_limit:
                raise ValueError("SSE event exceeds bound")
            if not line:
                if self.data:
                    data = b"\n".join(self.data)
                    self._event(data)
                    if data != b"[DONE]":
                        yield b"data: " + data.replace(b"\n", b"\ndata: ") + b"\n\n"
                self.data = []
                self.event_size = 0
            elif line.startswith(b"data:"):
                self.data.append(line[5:].removeprefix(b" "))
            elif not line.startswith(b":"):
                raise ValueError("unsupported SSE field")
        if len(self.buffer) + self.event_size > self.event_limit:
            raise ValueError("SSE event exceeds bound")

    def complete(self) -> None:
        """Require a clean EOF and consistent terminal worker evidence."""
        if self.buffer or self.data or not self.done:
            raise ValueError("truncated stream or missing DONE")
        if self.finish_reason is None or self.usage is None or not self.tokens:
            raise ValueError("missing finish reason, usage, or generated token IDs")
        prompt = _integer(self.usage.get("prompt_tokens"))
        generated = _integer(self.usage.get("completion_tokens"))
        total = _integer(self.usage.get("total_tokens"))
        if (
            prompt != self.prompt_count
            or generated != len(self.tokens)
            or total != prompt + generated
        ):
            raise ValueError("usage/token count mismatch")
        if self.finish_reason == "length" and generated != self.max_tokens:
            raise ValueError("length termination before output budget")

    def _event(self, data: bytes) -> None:
        if self.done:
            raise ValueError("event after DONE")
        if data == b"[DONE]":
            if self.finish_reason is None or self.usage is None:
                raise ValueError("DONE before terminal evidence")
            self.done = True
            return
        obj = _object(data)
        if "error" in obj:
            raise ValueError("worker error event")
        choices = obj.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("expected zero or one choices")
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or _integer(choice.get("index")) != 0:
                raise ValueError("invalid choice index")
            tokens = choice.get("token_ids", [])
            if not isinstance(tokens, list):
                raise ValueError("token_ids must be a list")
            if choice.get("text") and not tokens:
                raise ValueError("generated text without worker token IDs")
            if self.finish_reason is not None:
                raise ValueError("choice after finish reason")
            for token in tokens:
                if _integer(token) > 2**31 - 1:
                    raise ValueError("invalid generated token ID")
                self.tokens.append(token)
            if len(self.tokens) > self.max_tokens:
                raise ValueError("output budget exceeded")
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("stop", "length"):
                    raise ValueError("invalid finish reason")
                self.finish_reason = reason
        usage = obj.get("usage")
        if usage is not None:
            if not isinstance(usage, dict) or self.usage is not None:
                raise ValueError("invalid or duplicate usage")
            self.usage = usage


def create_app(
    route: Callable[[dict[str, object]], Awaitable[ProxyRoute]],
    journal_path: Path,
    *,
    on_finish: Callable[[ProxyFinish], Awaitable[None]] | None = None,
    limits: ProxyLimits = _DEFAULT_LIMITS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the standalone POST /v1/completions ASGI proxy.

    Args:
        route: Async callback accepting validated completion JSON (including
            request_id and optional arrival_ns/affinity), returning ProxyRoute.
        journal_path: New, exclusive JSONL path with an existing parent directory.
        on_finish: Async callback for each admitted ID, including predispatch
            failures. Must finish within request_timeout_s. Empty worker_id means
            routing never selected a worker. Only complete protocol evidence sets
            worker_terminal_confirmed; HTTP close alone never does.
        limits: Positive body/token/stream/admission/deadline bounds.
        transport: Optional httpx transport for CPU fixtures, closed per request.

    Returns:
        FastAPI app. Duplicate IDs return 409; invalid JSON/input 422; body limit
        413. Stream failures emit an error event without DONE and retain evidence.

    Raises:
        OSError: Journal creation/write failure (never downgraded to success).

    Notes:
        Single event-loop process only. No retries, cancel RPC, state imports or
        GPU/P2P qualification. All events use this process's monotonic clock.
        Deadline begins at ingress; arrival_ns is preserved, not trusted as this
        host's clock. Client must enforce its scheduled-arrival deadline too.
    """
    app = FastAPI()
    journal = _Journal(journal_path)
    seen: set[str] = set()
    tainted: set[str] = set()
    inflight = 0

    @app.post("/v1/completions", response_model=None)
    async def completions(request: Request) -> JSONResponse | StreamingResponse:
        nonlocal inflight
        ingress_ns = time.monotonic_ns()
        try:
            raw = bytearray()
            with anyio.fail_after(limits.request_timeout_s):
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw) > limits.max_body_bytes:
                        return JSONResponse({"error": "body exceeds bound"}, 413)
            body = _object(bytes(raw))
            rid = _validate(body, limits)
        except (ValueError, UnicodeError):
            logger.exception("Invalid completion input")
            return JSONResponse({"error": "invalid completion input"}, 422)
        except TimeoutError:
            logger.exception("Request body timeout")
            return JSONResponse({"error": "body timeout"}, 408)
        if rid in seen:
            return JSONResponse({"error": "duplicate request_id"}, 409)
        if len(seen) >= limits.max_requests:
            return JSONResponse({"error": "request inventory full"}, 503)
        seen.add(rid)
        journal.record("ingress", rid, ingress_ns=ingress_ns, request=body)
        overloaded = inflight >= limits.max_inflight
        inflight += 1

        worker_id = ""
        status: FinishStatus = "error"
        dispatched = False
        confirmed = False
        diagnostic = ""
        finalized = False
        observer = _Observer(
            len(cast(list[int], body["prompt"])),
            cast(int, body["max_tokens"]),
            limits.max_event_bytes,
        )

        async def finalize() -> None:
            nonlocal inflight
            nonlocal finalized
            if finalized:
                return
            finalized = True
            if dispatched and not confirmed:
                tainted.add(worker_id)
            inflight -= 1
            with anyio.CancelScope(shield=True):
                try:
                    journal.record(
                        "end",
                        rid,
                        worker_id=worker_id,
                        status=status,
                        worker_terminal_confirmed=confirmed,
                        diagnostic=diagnostic,
                        token_ids=observer.tokens,
                        usage=observer.usage,
                        finish_reason=observer.finish_reason,
                    )
                finally:
                    if on_finish is not None:
                        with anyio.fail_after(limits.request_timeout_s):
                            await on_finish(
                                ProxyFinish(rid, worker_id, status, confirmed)
                            )

        async def produce(
            sender: anyio.abc.ObjectSendStream[bytes | Exception],
        ) -> None:
            nonlocal worker_id, status, dispatched, confirmed, diagnostic
            try:
                remaining = (
                    limits.request_timeout_s - (time.monotonic_ns() - ingress_ns) / 1e9
                )
                with anyio.fail_after(max(0, remaining)):
                    if overloaded:
                        status = "overload"
                        raise ValueError("inflight bound exceeded")
                    decision = await route(copy.deepcopy(body))
                    worker_id = decision.worker_id
                    if (
                        not worker_id
                        or decision.load_mode not in ("native_load", "skip_external")
                        or httpx.URL(decision.url).scheme not in ("http", "https")
                    ):
                        raise ValueError("invalid proxy route")
                    if worker_id in tainted:
                        status = "worker_tainted"
                        raise ValueError("worker lacks confirmed termination")
                    outgoing = {
                        key: value
                        for key, value in body.items()
                        if key not in ("request_id", "arrival_ns", "affinity")
                    }
                    params = dict(
                        cast(dict[str, object], body.get("kv_transfer_params", {}))
                    )
                    params["lmcache.skip_load"] = decision.load_mode == "skip_external"
                    outgoing.update(
                        kv_transfer_params=params,
                        return_token_ids=True,
                        stream_options={"include_usage": True},
                    )
                    journal.record(
                        "dispatch",
                        rid,
                        worker_id=worker_id,
                        url=decision.url,
                        load_mode=decision.load_mode,
                        metadata=decision.metadata,
                        outgoing=outgoing,
                    )
                    dispatched = True
                    async with _worker_stream(
                        decision.url, outgoing, rid, limits.request_timeout_s, transport
                    ) as response:
                        journal.record(
                            "worker_status", rid, http_status=response.status_code
                        )
                        if (
                            response.is_success
                            and response.headers.get("content-type", "")
                            .split(";")[0]
                            .strip()
                            != "text/event-stream"
                        ):
                            raise ValueError("worker response is not SSE")
                        size = 0
                        first = False
                        chunk_index = 0
                        async for chunk in response.aiter_bytes():
                            received_ns = time.monotonic_ns()
                            chunk_index += 1
                            size += len(chunk)
                            if size > limits.max_stream_bytes:
                                raise ValueError("stream exceeds bound")
                            journal.record(
                                "raw_chunk",
                                rid,
                                chunk_index=chunk_index,
                                monotonic_ns=received_ns,
                                bytes_b64=base64.b64encode(chunk).decode("ascii"),
                            )
                            response.raise_for_status()
                            for frame in observer.feed(chunk):
                                if observer.tokens and not first:
                                    first = True
                                    journal.record(
                                        "first_token",
                                        rid,
                                        token_id=observer.tokens[0],
                                        monotonic_ns=received_ns,
                                        chunk_index=chunk_index,
                                    )
                                await sender.send(frame)
                        response.raise_for_status()
                        observer.complete()
                    confirmed = True
                    if (
                        time.monotonic_ns() - ingress_ns
                        >= limits.request_timeout_s * 1e9
                    ):
                        raise TimeoutError("deadline elapsed during cleanup")
                    status = "success"
            except TimeoutError:
                logger.exception("Completion deadline exceeded")
                status = "timeout"
                diagnostic = "deadline exceeded"
            except (ValueError, UnicodeError, httpx.HTTPError, RuntimeError) as exc:
                logger.exception("Completion failed")
                diagnostic = str(exc)
            except (asyncio.CancelledError, GeneratorExit):
                # Cancellation must retain accounting even while Starlette's
                # disconnect scope is cancelling this response generator.
                status = "cancelled"
                raise
            finally:
                await finalize()
            if confirmed:
                await sender.send(b"data: [DONE]\n\n")
            else:
                await sender.send(
                    b"data: "
                    + json.dumps(
                        {
                            "error": {
                                "request_id": rid,
                                "status": status,
                                "message": diagnostic,
                            }
                        }
                    ).encode()
                    + b"\n\n"
                )

        async def cleanup() -> None:
            nonlocal status
            if not finalized:
                status = "cancelled"
                await finalize()

        async def stream() -> AsyncGenerator[bytes, None]:
            sender, receiver = anyio.create_memory_object_stream[bytes | Exception](1)

            async def pump() -> None:
                try:
                    await produce(sender)
                except Exception as exc:
                    logger.exception("Stream producer failed")
                    await sender.send(exc)
                finally:
                    sender.close()

            task = asyncio.create_task(pump())
            try:
                async for item in receiver:
                    if isinstance(item, Exception):
                        raise item
                    yield item
                await task
            finally:
                task.cancel()
                with anyio.CancelScope(shield=True):
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    finally:
                        receiver.close()

        return _OwnedStreamingResponse(
            stream(), cleanup, rid, ingress_ns / 1e9 + limits.request_timeout_s
        )

    return app


def main(argv: list[str] | None = None) -> int:
    """Serve one static worker route using existing uvicorn dependencies.

    Args:
        argv: CLI arguments, or None for sys.argv.

    Returns:
        Zero after orderly server shutdown. Argument errors exit through argparse.

    Raises:
        OSError: Journal or server setup fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument(
        "--load-mode", choices=("native_load", "skip_external"), default="native_load"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)

    async def static_route(body: dict[str, object]) -> ProxyRoute:
        return ProxyRoute(args.worker_id, args.worker_url, args.load_mode, {})

    uvicorn.run(create_app(static_route, args.journal), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
