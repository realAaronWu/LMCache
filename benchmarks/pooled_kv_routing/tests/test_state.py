# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pytest

# First Party
from benchmarks.pooled_kv_routing.schemas import RouteRequest, WorkerSnapshot
from benchmarks.pooled_kv_routing.telemetry import Telemetry


def test_reservation_terminal_and_taint() -> None:
    state = Telemetry((WorkerSnapshot("a", "http://a", True, 0, 0),), namespace="n")
    request = RouteRequest("r", (1, 2), 3, 0)
    state.reserve(request, "a", 10, 0)
    assert state.snapshot(1)[0].pending_work_ms == 10
    assert state.snapshot(1)[0].sampled_ns == 0
    with pytest.raises(ValueError, match="duplicate"):
        state.reserve(request, "a", 10, 1)
    state.cancel_unconfirmed("r", 2)
    snapshot = state.snapshot(2)[0]
    assert not snapshot.healthy
    assert snapshot.pending_work_ms == 10
    state.confirmed_complete("r", 3)
    assert state.snapshot(3)[0].pending_work_ms == 0
    assert not state.snapshot(3)[0].healthy
    with pytest.raises(ValueError, match="duplicate"):
        state.reserve(request, "a", 10, 3)
    with pytest.raises(ValueError, match="unavailable"):
        state.reserve(RouteRequest("new", (1,), 1, 0), "a", 10, 3)
    with pytest.raises(ValueError, match="terminal"):
        state.confirmed_complete("r", 4)


def test_concurrent_ingress_is_immediately_visible() -> None:
    state = Telemetry((WorkerSnapshot("a", "http://a", True, 0, 0),), namespace="n")

    def reserve(i: int) -> None:
        state.reserve(RouteRequest(str(i), (1,), 1, 0), "a", 2, 0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reserve, range(50)))
    snapshot = state.snapshot(0)[0]
    assert snapshot.pending_work_ms == 100
    assert snapshot.pending_requests == 50


def test_catalog_namespace_epoch_expiry_and_capacity() -> None:
    state = Telemetry(
        (WorkerSnapshot("a", "http://a", True, 0, 0),),
        namespace="n",
        max_catalog_entries=1,
    )
    worker = WorkerSnapshot("a", "http://a", True, 0, 0)
    with pytest.raises(ValueError, match="epoch"):
        state.observe_worker(worker, epoch="")
    state.observe_worker(worker, epoch="e1")
    with pytest.raises(ValueError, match="namespace"):
        state.confirm_prefix("a", (1,), namespace="other", epoch="e1", sampled_ns=0)
    state.confirm_prefix("a", (1,), namespace="n", epoch="e1", sampled_ns=0)
    request = RouteRequest("r", (1, 2), 1, 0)
    assert state.snapshot(0, request)[0].cached_prefix_tokens == 1
    assert state.snapshot(300_000_000, request)[0].cached_prefix_tokens == 1
    assert state.snapshot(301_000_000, request)[0].cached_prefix_tokens is None
    state.confirm_prefix("a", (2,), namespace="n", epoch="e1", sampled_ns=301_000_000)
    assert state.snapshot(301_000_000, request)[0].cached_prefix_tokens is None
    state.observe_worker(
        WorkerSnapshot("a", "http://a", True, 302_000_000, 0), epoch="e2"
    )
    with pytest.raises(ValueError, match="epoch"):
        state.confirm_prefix(
            "a", (1,), namespace="n", epoch="e1", sampled_ns=302_000_000
        )
    assert state.snapshot(302_000_000, request)[0].cached_prefix_tokens is None
    state.refresh_failed("a", 303_000_000)
    assert not state.snapshot(303_000_000)[0].healthy


def test_state_rejects_history_and_bounded_ledger_overflow() -> None:
    state = Telemetry(
        (WorkerSnapshot("a", "http://a", True, 0, 0),), namespace="n", max_requests=1
    )
    state.reserve(RouteRequest("r", (1,), 1, 0), "a", 1, 1)
    with pytest.raises(ValueError, match="time"):
        state.snapshot(0)
    with pytest.raises(ValueError, match="capacity"):
        state.reserve(RouteRequest("r2", (1,), 1, 0), "a", 1, 1)


@pytest.mark.parametrize("tokens", [(True,), (-1,), (1.5,)])
def test_token_schema_rejects_wrong_values(tokens: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        RouteRequest("r", tokens, 1, 0)


def test_catalog_eviction_restart_and_no_dispatch_inferred_apc() -> None:
    state = Telemetry(
        (WorkerSnapshot("a", "http://a", True, 0, 0, 1, 1),),
        namespace="n",
        max_catalog_entries=1,
    )
    request = RouteRequest("r", (1, 2), 1, 0)
    assert state.snapshot(0, request)[0].gpu_cached_prefix_tokens is None
    state.observe_worker(WorkerSnapshot("a", "http://a", True, 0, 0), epoch="e1")
    state.reserve(request, "a", 2, 0)
    assert state.snapshot(0, request)[0].cached_prefix_tokens is None
    state.confirmed_complete("r", 0)
    assert state.snapshot(0, request)[0].gpu_cached_prefix_tokens is None
    state.confirm_prefix("a", (1,), namespace="n", epoch="e1", sampled_ns=0)
    state.confirm_prefix("a", (2,), namespace="n", epoch="e1", sampled_ns=0)
    assert state.snapshot(0, request)[0].cached_prefix_tokens is None
    other = RouteRequest("other", (2, 3), 1, 0)
    assert state.snapshot(0, other)[0].cached_prefix_tokens == 1
    state.observe_worker(WorkerSnapshot("a", "http://a", True, 1, 0), epoch="e2")
    assert state.snapshot(1, other)[0].cached_prefix_tokens is None


def test_affinity_and_observation_time_cannot_be_bypassed() -> None:
    state = Telemetry((WorkerSnapshot("a", "http://a", True, 0, 0),), namespace="n")
    request = RouteRequest("r", (1,), 1, 0, "b")
    with pytest.raises(ValueError, match="affinity"):
        state.reserve(request, "a", 1, 0)
    state.observe_worker(WorkerSnapshot("a", "http://a", True, 1, 0), epoch="e")
    state.confirm_prefix("a", (1,), namespace="n", epoch="e", sampled_ns=2)
    with pytest.raises(ValueError, match="time"):
        state.snapshot(1)


def test_restart_with_active_work_does_not_forgive_reservation() -> None:
    state = Telemetry(
        (WorkerSnapshot("a", "http://a", True, 0, 0, epoch="e1"),), namespace="n"
    )
    state.reserve(RouteRequest("r", (1,), 1, 0), "a", 4, 0)
    state.observe_worker(WorkerSnapshot("a", "http://a", True, 1, 0), epoch="e2")
    snapshot = state.snapshot(1)[0]
    assert not snapshot.healthy
    assert snapshot.pending_work_ms == 4
