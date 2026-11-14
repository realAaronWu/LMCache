# SPDX-License-Identifier: Apache-2.0
"""CPU entry-point tests; these do not qualify vLLM or GPU execution."""

# Standard
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import asyncio
import json
import socket
import subprocess
import sys
import threading
import time

# Third Party
from starlette.requests import ClientDisconnect
import httpx
import pytest

# First Party
from benchmarks.pooled_kv_routing.proxy import (
    ProxyFinish,
    ProxyLimits,
    ProxyRoute,
    create_app,
)


def event(value: object) -> bytes:
    """Encode a fixture SSE event."""
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def valid_stream() -> bytes:
    """Return worker token evidence, including non-token events."""
    return b"".join(
        [
            event(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "token_ids": [],
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            event(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "hi",
                            "token_ids": [42],
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            event(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "token_ids": [],
                            "finish_reason": "length",
                        }
                    ]
                }
            ),
            event(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


def payload(request_id: str = "r1") -> dict[str, object]:
    """Return a token-addressed completion request."""
    return {
        "request_id": request_id,
        "model": "fixture",
        "prompt": [10, 11],
        "max_tokens": 1,
        "stream": True,
        "n": 1,
    }


async def exercise(
    tmp_path: Path, raw: bytes, body: dict[str, object] | None = None, status: int = 200
) -> tuple[
    httpx.Response, list[dict[str, object]], list[ProxyFinish], list[dict[str, object]]
]:
    """Drive the actual ASGI endpoint and read its durable journal."""
    sent: list[dict[str, object]] = []
    finished: list[ProxyFinish] = []

    async def worker(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            status, content=raw, headers={"content-type": "text/event-stream"}
        )

    async def route(body: dict[str, object]) -> ProxyRoute:
        return ProxyRoute(
            "w1", "http://worker/v1/completions", "skip_external", {"snapshot": 7}
        )

    async def finish(result: ProxyFinish) -> None:
        finished.append(result)

    path = tmp_path / "journal.jsonl"
    app = create_app(
        route, path, on_finish=finish, transport=httpx.MockTransport(worker)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/v1/completions", json=body if body is not None else payload()
        )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return response, sent, finished, records


def test_valid_entry_journals_actual_tokens_and_injects_mode(tmp_path: Path) -> None:
    response, sent, finished, records = asyncio.run(exercise(tmp_path, valid_stream()))
    assert response.status_code == 200
    assert response.content == valid_stream()
    assert sent[0]["prompt"] == [10, 11]
    assert sent[0]["kv_transfer_params"] == {"lmcache.skip_load": True}
    assert sent[0]["return_token_ids"] is True
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert "request_id" not in sent[0]
    assert finished[0].status == "success"
    assert finished[0].worker_terminal_confirmed is True
    assert [r["event"] for r in records].count("first_token") == 1
    terminal = records[-1]
    assert terminal["event"] == "end" and terminal["token_ids"] == [42]
    assert terminal["status"] == "success"
    assert terminal["accounting_status"] == "success"
    assert sum(r["event"] == "callback_attempt" for r in records) == 1
    assert [r["outcome"] for r in records if r["event"] == "callback_result"] == [
        "success"
    ]
    assert {r["event"] for r in records} >= {
        "ingress",
        "dispatch",
        "raw_chunk",
        "first_token",
        "end",
    }


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        valid_stream().replace(b"data: [DONE]\n\n", b""),
        valid_stream() + event({"error": {"message": "late"}}),
        valid_stream() + b"data: [DONE]\n\n",
        valid_stream().replace(b'"completion_tokens": 1', b'"completion_tokens": true'),
        valid_stream().replace(b'"token_ids": [42]', b'"token_ids": [true]'),
        valid_stream().replace(b'"token_ids": [42]', b'"token_ids": [NaN]'),
        valid_stream().replace(b'"completion_tokens": 1', b'"completion_tokens": 2'),
        valid_stream().replace(b'"finish_reason": "length"', b'"finish_reason": null'),
        valid_stream().replace(b"data: [DONE]\n\n", b'data: {"error": "crash"}\n\n'),
        valid_stream() + b'data: {"incomplete":',
    ],
)
def test_bad_evidence_never_finishes_success(tmp_path: Path, raw: bytes) -> None:
    response, _, finished, records = asyncio.run(exercise(tmp_path, raw))
    assert finished[0].status == "error"
    assert finished[0].worker_terminal_confirmed is False
    assert records[-1]["status"] == "error"
    assert b"data: [DONE]" not in response.content
    assert b'"error"' in response.content


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": [True]},
        {"prompt": "text"},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"stream": 1},
        {"n": True},
        {"n": 2},
        {"temperature": float("nan")},
        {"request_id": ""},
    ],
)
def test_invalid_input_is_rejected_before_dispatch(
    tmp_path: Path, change: dict[str, object]
) -> None:
    body = payload() | change

    # Encode explicitly so the NaN negative reaches the server's parser.
    async def run() -> None:
        async def route(body: dict[str, object]) -> ProxyRoute:
            raise AssertionError("invalid input reached routing")

        app = create_app(route, tmp_path / "journal.jsonl")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", content=json.dumps(body))
        assert response.status_code == 422

    asyncio.run(run())


def test_non_2xx_is_not_terminal_acknowledgement(tmp_path: Path) -> None:
    _, _, finished, records = asyncio.run(exercise(tmp_path, b"failure", status=503))
    assert finished[0].status == "error"
    assert not finished[0].worker_terminal_confirmed
    assert records[-1]["status"] == "error"
    assert any(record["event"] == "raw_chunk" for record in records)


def test_timeout_quarantines_worker_and_ids_are_not_reused(tmp_path: Path) -> None:
    async def run() -> None:
        finished: list[ProxyFinish] = []
        calls: list[str] = []

        async def worker(request: httpx.Request) -> httpx.Response:
            calls.append("dispatch")
            await asyncio.sleep(1)
            return httpx.Response(200, content=valid_stream())

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        async def finish(result: ProxyFinish) -> None:
            finished.append(result)

        app = create_app(
            route,
            tmp_path / "journal.jsonl",
            on_finish=finish,
            transport=httpx.MockTransport(worker),
            limits=ProxyLimits(request_timeout_s=0.03),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            await client.post("/v1/completions", json=payload())
            duplicate = await client.post("/v1/completions", json=payload())
            await client.post("/v1/completions", json=payload("r2"))
        assert duplicate.status_code == 409
        assert calls == ["dispatch"]
        assert finished[0].status == "timeout"
        assert not finished[0].worker_terminal_confirmed
        assert finished[1].status == "worker_tainted"

    asyncio.run(run())


def test_resource_limits_reject_without_worker(tmp_path: Path) -> None:
    async def run() -> None:
        async def route(body: dict[str, object]) -> ProxyRoute:
            raise AssertionError("oversized request reached routing")

        app = create_app(
            route, tmp_path / "journal.jsonl", limits=ProxyLimits(max_body_bytes=10)
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", json=payload())
        assert response.status_code == 413

    asyncio.run(run())


def test_disconnect_closes_worker_and_retains_taint(tmp_path: Path) -> None:
    """Cancel the real ASGI request after output, not a parser helper."""

    async def run() -> None:
        started = asyncio.Event()
        closed = asyncio.Event()
        finished: list[ProxyFinish] = []

        class WorkerStream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield event(
                    {
                        "choices": [
                            {"index": 0, "token_ids": [42], "finish_reason": None}
                        ]
                    }
                )
                started.set()
                await asyncio.sleep(60)

            async def aclose(self) -> None:
                await asyncio.sleep(0)
                closed.set()

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                stream=WorkerStream(),
                headers={"content-type": "text/event-stream"},
            )

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        async def finish(result: ProxyFinish) -> None:
            finished.append(result)

        path = tmp_path / "journal.jsonl"
        app = create_app(
            route, path, on_finish=finish, transport=httpx.MockTransport(worker)
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            task = asyncio.create_task(client.post("/v1/completions", json=payload()))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed.is_set()
            assert finished[0].status == "cancelled"
            assert not finished[0].worker_terminal_confirmed
            await client.post("/v1/completions", json=payload("r2"))
            assert finished[1].status == "worker_tainted"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records[2]["event"] == "worker_status"
        assert any(r["event"] == "first_token" for r in records)

    asyncio.run(run())


@pytest.mark.parametrize(
    "limit", [ProxyLimits(max_stream_bytes=10), ProxyLimits(max_event_bytes=10)]
)
def test_stream_bounds_fail_closed(tmp_path: Path, limit: ProxyLimits) -> None:
    async def run() -> None:
        finished: list[ProxyFinish] = []

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=valid_stream(),
                headers={"content-type": "text/event-stream"},
            )

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        async def finish(result: ProxyFinish) -> None:
            finished.append(result)

        app = create_app(
            route,
            tmp_path / "journal.jsonl",
            limits=limit,
            on_finish=finish,
            transport=httpx.MockTransport(worker),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", json=payload())
        assert b"data: [DONE]" not in response.content
        assert finished[0].status == "error"

    asyncio.run(run())


@pytest.mark.parametrize("fail_on", ["http.response.start", "http.response.body"])
def test_downstream_send_failure_finalizes_admitted_id(
    tmp_path: Path, fail_on: str
) -> None:
    async def run() -> None:
        finished: list[ProxyFinish] = []

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=valid_stream(),
                headers={"content-type": "text/event-stream"},
            )

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        async def finish(result: ProxyFinish) -> None:
            finished.append(result)

        app = create_app(
            route,
            tmp_path / "journal.jsonl",
            on_finish=finish,
            transport=httpx.MockTransport(worker),
        )
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "POST",
            "path": "/v1/completions",
            "root_path": "",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("proxy", 80),
            "client": ("client", 1),
        }

        async def receive() -> dict[str, object]:
            return {
                "type": "http.request",
                "body": json.dumps(payload()).encode(),
                "more_body": False,
            }

        async def send(message: dict[str, object]) -> None:
            if message["type"] == fail_on:
                raise OSError("client disconnected during send")

        with pytest.raises((OSError, ClientDisconnect)):
            await app(scope, receive, send)
        assert len(finished) == 1
        assert finished[0].status != "success"

    asyncio.run(run())


def test_cli_real_http_entry(tmp_path: Path) -> None:
    """Launch the standalone CLI against a CPU HTTP worker; no GPU qualification."""
    sent: list[dict[str, object]] = []

    class Worker(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            sent.append(
                json.loads(self.rfile.read(int(self.headers["content-length"])))
            )
            raw = valid_stream()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:
            pass

    worker = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    journal = tmp_path / "journal.jsonl"
    with (tmp_path / "proxy.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.pooled_kv_routing.proxy",
                "--worker-url",
                f"http://127.0.0.1:{worker.server_port}/v1/completions",
                "--worker-id",
                "cpu-fixture",
                "--journal",
                str(journal),
                "--port",
                str(port),
            ],
            cwd=Path(__file__).resolve().parents[3],
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False
            ) as client:
                deadline = time.monotonic() + 5
                while True:
                    assert process.poll() is None
                    try:
                        if client.get("/openapi.json").status_code == 200:
                            break
                    except httpx.ConnectError:
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(0.02)
                response = client.post("/v1/completions", json=payload())
            assert response.content == valid_stream()
            assert sent[0]["kv_transfer_params"] == {"lmcache.skip_load": False}
            assert (
                json.loads(journal.read_text().splitlines()[-1])["status"] == "success"
            )
        finally:
            process.terminate()
            process.wait(timeout=5)
            worker.shutdown()
            worker.server_close()
            thread.join(timeout=2)


@pytest.mark.parametrize("stride", [1, 17])
def test_fragmented_crlf_stream_preserves_token_evidence(
    tmp_path: Path, stride: int
) -> None:
    async def run() -> None:
        raw = valid_stream().replace(b"\n", b"\r\n")

        class Fragments(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                for start in range(0, len(raw), stride):
                    yield raw[start : start + stride]

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=Fragments(), headers={"content-type": "text/event-stream"}
            )

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        path = tmp_path / "journal.jsonl"
        app = create_app(route, path, transport=httpx.MockTransport(worker))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", json=payload())
        assert response.content == valid_stream()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records[-1]["status"] == "success"
        assert sum(r["event"] == "first_token" for r in records) == 1

    asyncio.run(run())


def test_deadline_during_automatic_eof_close_does_not_interrupt_cleanup(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        closed = asyncio.Event()
        finished: list[ProxyFinish] = []

        async def finish(result: ProxyFinish) -> None:
            finished.append(result)

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield valid_stream()
                await asyncio.sleep(0.02)

            async def aclose(self) -> None:
                await asyncio.sleep(0.02)
                closed.set()

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=Stream(), headers={"content-type": "text/event-stream"}
            )

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        app = create_app(
            route,
            tmp_path / "journal.jsonl",
            on_finish=finish,
            transport=httpx.MockTransport(worker),
            limits=ProxyLimits(request_timeout_s=0.03),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", json=payload())
        assert closed.is_set()
        assert b"data: [DONE]" not in response.content
        assert finished == [ProxyFinish("r1", "w1", "timeout", True)]
        end = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[-1])
        assert end["status"] == "timeout" and end["worker_terminal_confirmed"]

    asyncio.run(run())


@pytest.mark.parametrize("text", [{"wrong": "type"}, [], 1, True, None])
def test_recognized_text_requires_string(tmp_path: Path, text: object) -> None:
    raw = valid_stream().replace(
        b'"text": "hi"', b'"text": ' + json.dumps(text).encode()
    )
    response, _, finished, records = asyncio.run(exercise(tmp_path, raw))
    assert b"data: [DONE]" not in response.content
    assert finished[0].status == "error"
    assert records[-1]["status"] == "error"


@pytest.mark.parametrize("best_of", [True, 1.0])
def test_best_of_requires_exact_integer(tmp_path: Path, best_of: object) -> None:
    response, sent, finished, _ = asyncio.run(
        exercise(tmp_path, valid_stream(), payload() | {"best_of": best_of})
    )
    assert response.status_code == 422
    assert sent == [] and finished == []


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_finish_failure_is_durable_exact_once_and_stops_admission(
    tmp_path: Path, failure: str
) -> None:
    async def run() -> None:
        calls: list[ProxyFinish] = []

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=valid_stream(),
                headers={"content-type": "text/event-stream"},
            )

        async def finish(result: ProxyFinish) -> None:
            calls.append(result)
            if failure == "timeout":
                await asyncio.sleep(1)
            else:
                raise RuntimeError("ledger commit rejected")

        path = tmp_path / "journal.jsonl"
        app = create_app(
            route,
            path,
            on_finish=finish,
            transport=httpx.MockTransport(worker),
            limits=ProxyLimits(request_timeout_s=0.1),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", json=payload())
            assert b"data: [DONE]" not in response.content
            rejected = await client.post("/v1/completions", json=payload("r2"))
            assert rejected.status_code == 503
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert calls == [ProxyFinish("r1", "w1", "success", True)]
        assert sum(r["event"] == "callback_attempt" for r in records) == 1
        results = [r for r in records if r["event"] == "callback_result"]
        assert len(results) == 1 and results[0]["outcome"] == failure
        assert records[-1]["event"] == "end"
        assert records[-1]["status"] == failure
        assert records[-1]["completion_status"] == "success"
        assert records[-1]["accounting_status"] == failure
        assert records[-1]["worker_terminal_confirmed"] is True

    asyncio.run(run())


def test_ingress_slots_bound_readers_and_release_on_cancellation(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        readers: list[int] = []
        started = asyncio.Event()

        class SlowBody(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                readers.append(1)
                started.set()
                yield b'{"partial":'
                await asyncio.sleep(10)

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=valid_stream(),
                headers={"content-type": "text/event-stream"},
            )

        app = create_app(
            route,
            tmp_path / "journal.jsonl",
            limits=ProxyLimits(max_inflight=1),
            transport=httpx.MockTransport(worker),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            pending = asyncio.create_task(
                client.post("/v1/completions", content=SlowBody())
            )
            await asyncio.wait_for(started.wait(), 1)
            rejected = await asyncio.gather(
                *(client.post("/v1/completions", content=SlowBody()) for _ in range(3))
            )
            assert [r.status_code for r in rejected] == [503] * 3
            assert readers == [1]
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert (
                await client.post("/v1/completions", json=payload())
            ).content.endswith(b"data: [DONE]\n\n")

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["invalid", "oversize", "timeout", "disconnect"])
def test_ingress_slot_released_on_body_failures(tmp_path: Path, failure: str) -> None:
    async def run() -> None:
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                if failure == "timeout":
                    await asyncio.sleep(1)
                if failure == "disconnect":
                    raise ClientDisconnect()
                yield b"x" * (2048 if failure == "oversize" else 1)

        async def route(body: dict[str, object]) -> ProxyRoute:
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=valid_stream(),
                headers={"content-type": "text/event-stream"},
            )

        app = create_app(
            route,
            tmp_path / "journal.jsonl",
            limits=ProxyLimits(
                max_inflight=1, max_body_bytes=1024, request_timeout_s=0.1
            ),
            transport=httpx.MockTransport(worker),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            if failure == "disconnect":
                with pytest.raises(ClientDisconnect):
                    await client.post("/v1/completions", content=Body())
            else:
                response = await client.post("/v1/completions", content=Body())
                assert (
                    response.status_code
                    == {"invalid": 422, "oversize": 413, "timeout": 408}[failure]
                )
            assert (
                await client.post("/v1/completions", json=payload())
            ).content.endswith(b"data: [DONE]\n\n")

    asyncio.run(run())


def test_successful_callback_cannot_extend_success_past_request_deadline(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        calls: list[ProxyFinish] = []

        async def route(body: dict[str, object]) -> ProxyRoute:
            await asyncio.sleep(0.07)
            return ProxyRoute("w1", "http://worker/v1/completions", "native_load", {})

        async def worker(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=valid_stream(),
                headers={"content-type": "text/event-stream"},
            )

        async def finish(result: ProxyFinish) -> None:
            calls.append(result)
            await asyncio.sleep(0.06)

        path = tmp_path / "journal.jsonl"
        app = create_app(
            route,
            path,
            on_finish=finish,
            transport=httpx.MockTransport(worker),
            limits=ProxyLimits(request_timeout_s=0.1),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", json=payload())
        assert b"data: [DONE]" not in response.content
        end = json.loads(path.read_text().splitlines()[-1])
        assert end["status"] == "timeout"
        assert end["accounting_status"] == "success"
        assert calls == [ProxyFinish("r1", "w1", "success", True)]

    asyncio.run(run())
