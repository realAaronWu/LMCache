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


@pytest.mark.parametrize(("observed_ms", "existing_work"), [(0, (1e308,)), (1e308, ())])
def test_overflowing_reserve_is_atomic(
    observed_ms: float, existing_work: tuple[float, ...]
) -> None:
    state = Telemetry(
        (WorkerSnapshot("a", "http://a", True, 0, observed_ms),), namespace="n"
    )
    for i, work_ms in enumerate(existing_work):
        state.reserve(RouteRequest(str(i), (1,), 1, 0), "a", work_ms, 0)
    before = state.snapshot(0)
    rejected = RouteRequest("rejected", (1,), 1, 0)
    with pytest.raises(ValueError, match="pending_work_ms"):
        state.reserve(rejected, "a", 1e308, 10)
    # Rejection must not consume an ID, add work, taint, or advance the clock.
    assert state.snapshot(0) == before
    state.reserve(rejected, "a", 1, 0)
    assert state.snapshot(0)[0].pending_requests == len(existing_work) + 1


def test_integer_aggregate_rejection_is_value_error_and_atomic() -> None:
    state = Telemetry((WorkerSnapshot("a", "http://a", True, 0, 0),), namespace="n")
    state.reserve(RouteRequest("first", (1,), 1, 0), "a", 10**308, 0)
    before = state.snapshot(0)
    retry = RouteRequest("retry", (1,), 1, 0)
    with pytest.raises(ValueError, match="pending_work_ms"):
        state.reserve(retry, "a", 10**308, 10)
    assert state.snapshot(0) == before
    state.confirmed_complete("first", 0)
    state.reserve(retry, "a", 10**308, 0)
    assert state.snapshot(0)[0].pending_requests == 1
    assert state.snapshot(0)[0].pending_work_ms == before[0].pending_work_ms


def test_huge_integer_worker_work_rejected_as_value_error() -> None:
    with pytest.raises(ValueError, match="pending_work_ms"):
        WorkerSnapshot("a", "http://a", True, 0, 10**400)


@pytest.mark.parametrize("uncertain", [False, True], ids=["complete", "cancel"])
@pytest.mark.parametrize("wrong_worker", ["b", ""], ids=["other", "empty"])
def test_terminal_worker_guard_is_atomic(uncertain: bool, wrong_worker: str) -> None:
    state = Telemetry(
        (
            WorkerSnapshot("a", "http://a", True, 0, 0),
            WorkerSnapshot("b", "http://b", True, 0, 0),
        ),
        namespace="n",
    )
    state.reserve(RouteRequest("r", (1,), 1, 0), "a", 7, 0)
    before = state.snapshot(0)
    operation = state.cancel_unconfirmed if uncertain else state.confirmed_complete
    with pytest.raises(ValueError, match="worker_id"):
        operation("r", 10, worker_id=wrong_worker)
    assert state.snapshot(0) == before
    with pytest.raises(KeyError):
        operation("unknown", 10, worker_id="a")
    assert state.snapshot(0) == before
    operation("r", 0, worker_id="a")
    worker = state.snapshot(0)[0]
    assert worker.healthy is not uncertain
    assert worker.pending_work_ms == (7 if uncertain else 0)
    if uncertain:
        operation("r", 0, worker_id="a")
        assert state.snapshot(0)[0] == worker


def test_overflowing_observation_preserves_epoch_catalog_and_clock() -> None:
    state = Telemetry(
        (WorkerSnapshot("a", "http://a", True, 0, 0, epoch="e1"),), namespace="n"
    )
    request = RouteRequest("r", (1,), 1, 0)
    state.confirm_prefix("a", (1,), namespace="n", epoch="e1", sampled_ns=0)
    state.reserve(request, "a", 1e308, 0)
    before = state.snapshot(0, request)
    with pytest.raises(ValueError, match="pending_work_ms"):
        state.observe_worker(
            WorkerSnapshot("a", "http://a", True, 10, 1e308), epoch="e2"
        )
    assert state.snapshot(0, request) == before
    state.confirmed_complete("r", 0)
    state.observe_worker(WorkerSnapshot("a", "http://a", True, 10, 1e308), epoch="e1")
    assert state.snapshot(10)[0].pending_work_ms == 1e308


def test_confirmed_fetch_path_is_request_specific_and_expires() -> None:
    state = Telemetry(
        (WorkerSnapshot("a", "http://a", True, 0, 0, epoch="e"),), namespace="n"
    )
    request = RouteRequest("r", (1, 2, 3), 1, 0)
    state.confirm_prefix(
        "a", (1,), namespace="n", epoch="e", sampled_ns=0, fetch_path="remote_host"
    )
    state.confirm_prefix(
        "a", (1, 2), namespace="n", epoch="e", sampled_ns=0, fetch_path="local_host"
    )
    assert state.snapshot(0, request)[0].fetch_path == "local_host"
    assert state.snapshot(0)[0].fetch_path is None
    state.observe_worker(WorkerSnapshot("a", "http://a", True, 1, 0), epoch="new")
    assert state.snapshot(1, request)[0].fetch_path is None
    state.confirm_prefix(
        "a",
        (1,),
        namespace="n",
        epoch="new",
        sampled_ns=1,
        fetch_path="remote_host",
    )
    assert state.snapshot(300_000_001, request)[0].fetch_path == "remote_host"
    assert state.snapshot(300_000_002, request)[0].fetch_path is None
