# SPDX-License-Identifier: Apache-2.0
"""Cross-component contracts for routing, reservations and stream completion."""

# Standard
from pathlib import Path
import asyncio
import json

# Third Party
import httpx
import pytest

# First Party
from benchmarks.pooled_kv_routing.costs import CostEntry, CostKey, CostTable
from benchmarks.pooled_kv_routing.proxy import ProxyFinish, create_app
from benchmarks.pooled_kv_routing.runtime import RoutingRuntime
from benchmarks.pooled_kv_routing.schemas import (
    LoadMode,
    Policy,
    RouterConfig,
    WorkerSnapshot,
)
from benchmarks.pooled_kv_routing.telemetry import Telemetry


def components(completion_ms: float | None = 20) -> tuple[RoutingRuntime, Telemetry]:
    state = Telemetry(
        (WorkerSnapshot("A", "http://worker/v1/completions", True, 0, 0),),
        namespace="fixture",
    )
    modes: tuple[LoadMode, ...] = ("native_load", "skip_external")
    costs = CostTable(
        "fixture-only",
        tuple(
            CostEntry(CostKey(2, 1, None, None, 0, mode), 5, "fixture", completion_ms)
            for mode in modes
        ),
    )
    runtime = RoutingRuntime(
        state, costs, RouterConfig(0, policy="lew"), clock=lambda: 100
    )
    return runtime, state


def test_real_bridge_reserves_completion_not_queue_inclusive_ttft() -> None:
    async def run() -> None:
        runtime, state = components()
        for rid in ("one", "two"):
            decision = await runtime.route(
                {"request_id": rid, "prompt": [1, 2], "max_tokens": 1, "affinity": "A"}
            )
            assert decision.worker_id == "A"
        assert state.snapshot(100)[0].pending_work_ms == 40
        await runtime.finish(ProxyFinish("one", "A", "success", True))
        assert state.snapshot(100)[0].pending_work_ms == 20
        await runtime.finish(ProxyFinish("two", "A", "timeout", False))
        assert not state.snapshot(100)[0].healthy
        assert state.snapshot(100)[0].pending_work_ms == 20
        with pytest.raises(ValueError, match="healthy"):
            await runtime.route(
                {"request_id": "three", "prompt": [1, 2], "max_tokens": 1}
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "policy, mode", [("cost", "skip_external"), ("lew", "native_load")]
)
@pytest.mark.parametrize("complete", [True, False])
def test_proxy_runtime_end_to_end_accounting(
    tmp_path: Path, policy: Policy, mode: LoadMode, complete: bool
) -> None:
    """Same real entry path for both policies, including uncertain completion."""

    async def run() -> None:
        state = Telemetry(
            (
                WorkerSnapshot("A", "http://A/v1/completions", True, 0, 400),
                WorkerSnapshot("B", "http://B/v1/completions", True, 0, 0),
            ),
            namespace="fixture",
        )
        costs = CostTable(
            "fixture-only",
            (
                CostEntry(
                    CostKey(2, 1, None, None, 0, "native_load"), 100, "fixture", 150
                ),
                CostEntry(
                    CostKey(2, 1, None, None, 0, "skip_external"), 30, "fixture", 60
                ),
            ),
        )
        runtime = RoutingRuntime(
            state, costs, RouterConfig(0, policy=policy), clock=lambda: 100
        )
        sent = []

        async def worker(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "b"
            sent.append(json.loads(request.content))
            envelope = {
                "choices": [
                    {
                        "index": 0,
                        "text": "x",
                        "token_ids": [42],
                        "finish_reason": "length",
                    }
                ]
            }
            usage = {
                "choices": [],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }
            raw = b"".join(
                b"data: " + json.dumps(item).encode() + b"\n\n"
                for item in (envelope, usage)
            )
            if complete:
                raw += b"data: [DONE]\n\n"
            return httpx.Response(
                200, content=raw, headers={"content-type": "text/event-stream"}
            )

        journal = tmp_path / "journal.jsonl"
        app = create_app(
            runtime.route,
            journal,
            on_finish=runtime.finish,
            transport=httpx.MockTransport(worker),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://proxy"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "request_id": "one",
                    "model": "fixture",
                    "prompt": [1, 2],
                    "max_tokens": 1,
                    "stream": True,
                    "n": 1,
                },
            )
        assert sent[0]["kv_transfer_params"]["lmcache.skip_load"] is (
            mode == "skip_external"
        )
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        end = next(item for item in records if item["event"] == "end")
        assert end["status"] == ("success" if complete else "error")
        assert (b"[DONE]" in response.content) is complete
        snapshot = next(w for w in state.snapshot(100) if w.worker_id == "B")
        if complete:
            assert snapshot.pending_work_ms == 0 and snapshot.healthy
        else:
            assert snapshot.pending_work_ms == (60 if mode == "skip_external" else 150)
            assert not snapshot.healthy

    asyncio.run(run())


def test_unknown_completion_work_is_not_invented_or_reserved() -> None:
    async def run() -> None:
        runtime, state = components(None)
        with pytest.raises(ValueError, match="completion"):
            await runtime.route(
                {"request_id": "one", "prompt": [1, 2], "max_tokens": 1}
            )
        assert state.snapshot(100)[0].pending_requests == 0
        await runtime.finish(ProxyFinish("unrouted", "", "error", False))
        assert state.snapshot(100)[0].healthy

    asyncio.run(run())


@pytest.mark.parametrize("worker_id", ["B", ""])
@pytest.mark.parametrize("confirmed", [True, False])
def test_finish_rejects_crossed_worker_without_changing_reservation(
    worker_id: str, confirmed: bool
) -> None:
    async def run() -> None:
        runtime, state = components()
        await runtime.route({"request_id": "one", "prompt": [1, 2], "max_tokens": 1})
        before = state.snapshot(100)
        with pytest.raises(ValueError):
            await runtime.finish(
                ProxyFinish(
                    "one", worker_id, "success" if confirmed else "error", confirmed
                )
            )
        assert state.snapshot(100) == before
        await runtime.finish(ProxyFinish("one", "A", "success", True))
        assert state.snapshot(100)[0].pending_work_ms == 0

    asyncio.run(run())


def test_uncertainty_is_idempotent_until_real_terminal_confirmation() -> None:
    async def run() -> None:
        runtime, state = components()
        await runtime.route({"request_id": "one", "prompt": [1, 2], "max_tokens": 1})
        for _ in range(2):
            await runtime.finish(ProxyFinish("one", "A", "timeout", False))
            assert state.snapshot(100)[0].pending_work_ms == 20
        await runtime.finish(ProxyFinish("one", "A", "success", True))
        assert state.snapshot(100)[0].pending_work_ms == 0
        assert not state.snapshot(100)[0].healthy
        with pytest.raises(ValueError):
            await runtime.finish(ProxyFinish("one", "A", "success", True))

    asyncio.run(run())


def test_native_completion_reservation_uses_confirmed_fetch_path() -> None:
    async def run() -> None:
        worker = WorkerSnapshot("A", "http://worker/v1/completions", True, 0, 0)
        state = Telemetry((worker,), namespace="fixture")
        state.observe_worker(worker, epoch="one")
        state.confirm_prefix(
            "A",
            (1, 2),
            namespace="fixture",
            epoch="one",
            sampled_ns=0,
            fetch_path="remote_host",
        )
        costs = CostTable(
            "fixture-only",
            (
                CostEntry(
                    CostKey(2, 1, 2, None, 0, "native_load", "local_host"),
                    2,
                    "fixture",
                    10,
                ),
                CostEntry(
                    CostKey(2, 1, 2, None, 0, "native_load", "remote_host"),
                    3,
                    "fixture",
                    30,
                ),
            ),
        )
        runtime = RoutingRuntime(
            state, costs, RouterConfig(0, policy="lew"), clock=lambda: 100
        )
        decision = await runtime.route(
            {"request_id": "remote", "prompt": [1, 2], "max_tokens": 1}
        )
        assert decision.load_mode == "native_load"
        assert state.snapshot(100)[0].pending_work_ms == 30

    asyncio.run(run())
