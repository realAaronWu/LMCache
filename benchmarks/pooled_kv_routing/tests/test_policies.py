# SPDX-License-Identifier: Apache-2.0
# Standard
import json

# Third Party
import pytest

# First Party
from benchmarks.pooled_kv_routing.costs import CostEntry, CostKey, CostTable
from benchmarks.pooled_kv_routing.policies import choose
from benchmarks.pooled_kv_routing.schemas import (
    LoadMode,
    Policy,
    RouterConfig,
    RouteRequest,
    WorkerSnapshot,
)


def test_cost_choice_baselines_and_serialization() -> None:
    request = RouteRequest("r", (1, 2), 3, 0)
    snapshots = (
        WorkerSnapshot("a", "http://a", True, 0, 10, 2, 0),
        WorkerSnapshot("b", "http://b", True, 0, 0, 0, 0),
    )
    conditions: list[tuple[int | None, LoadMode, float]] = [
        (2, "native_load", 1),
        (None, "skip_external", 20),
        (0, "native_load", 30),
    ]
    table = CostTable(
        "v",
        tuple(
            CostEntry(CostKey(2, 3, prefix, 0, 0, mode), ms, "raw")
            for prefix, mode, ms in conditions
        ),
    )
    cost = choose(request, snapshots, table, RouterConfig(0))
    assert cost.worker_id == "a"
    assert cost.predicted_ms == 11
    assert (
        choose(request, snapshots, table, RouterConfig(0, policy="lew")).worker_id
        == "b"
    )
    assert (
        choose(
            request,
            snapshots,
            table,
            RouterConfig(0, policy="static", alpha_ms_per_token=10),
        ).worker_id
        == "a"
    )
    assert (
        json.loads(json.dumps(cost.as_dict(), allow_nan=False))["cost_version"] == "v"
    )
    assert len(cost.candidates) == 4


@pytest.mark.parametrize("policy", ["cost", "lew", "static"])
def test_stale_affinity_and_unavailable_are_safe(policy: Policy) -> None:
    table = CostTable("v", ())
    request = RouteRequest("r", (1,), 1, 0, "a")
    snapshots = (
        WorkerSnapshot("a", "http://a", True, 0, 99, 1),
        WorkerSnapshot("b", "http://b", True, 301_000_000, 0),
    )
    decision = choose(
        request, snapshots, table, RouterConfig(301_000_000, policy=policy)
    )
    assert decision.worker_id == "a"
    assert decision.load_mode == "skip_external"
    assert decision.predicted_ms is None
    assert decision.fallback_reason == "stale_snapshot"
    with pytest.raises(ValueError, match="eligible"):
        choose(request, snapshots[1:], table, RouterConfig(301_000_000, policy=policy))


def test_unknown_cost_falls_back_without_zero_and_ties_are_stable() -> None:
    table = CostTable("v", ())
    snapshots = (
        WorkerSnapshot("a", "http://a", True, 0, 0),
        WorkerSnapshot("b", "http://b", True, 0, 0),
    )
    winners = set()
    for i in range(20):
        request = RouteRequest(str(i), (1,), 1, 0)
        decision = choose(request, snapshots, table, RouterConfig(0))
        assert decision == choose(request, snapshots[::-1], table, RouterConfig(0))
        assert decision.predicted_ms is None
        assert decision.fallback_reason == "unknown_cost_bucket"
        winners.add(decision.worker_id)
    assert winners == {"a", "b"}


def test_future_snapshots_not_used() -> None:
    with pytest.raises(ValueError, match="future"):
        choose(
            RouteRequest("r", (1,), 1, 0),
            (WorkerSnapshot("a", "http://a", True, 1, 0),),
            CostTable("v", ()),
            RouterConfig(0),
        )


def test_frozen_baseline_mode_and_gain_threshold() -> None:
    request = RouteRequest("r", (1,), 1, 0)
    workers = (WorkerSnapshot("a", "http://a", True, 0, 0, 1, 0),)
    table = CostTable(
        "v",
        (
            CostEntry(CostKey(1, 1, 1, 0, 0, "native_load"), 10, "raw"),
            CostEntry(CostKey(1, 1, None, 0, 0, "skip_external"), 8, "raw"),
        ),
    )
    assert choose(request, workers, table, RouterConfig(0)).load_mode == "skip_external"
    limited = choose(request, workers, table, RouterConfig(0, min_gain_ms=2))
    assert limited.load_mode == "native_load"
    assert limited.fallback_reason == "insufficient_gain"
    baseline = choose(
        request,
        workers,
        table,
        RouterConfig(0, policy="lew", baseline_load_mode="skip_external"),
    )
    assert baseline.load_mode == "skip_external"


@pytest.mark.parametrize("policy", ["cost", "lew", "static"])
def test_freshness_boundary_and_health_apply_to_every_policy(policy: Policy) -> None:
    request = RouteRequest("r", (1,), 1, 0)
    workers = (
        WorkerSnapshot("a", "http://a", True, 0, 0, 1),
        WorkerSnapshot("b", "http://b", True, 300_000_001, 10, 0),
        WorkerSnapshot("c", "http://c", False, 300_000_001, 0, 1),
    )
    decision = choose(
        request, workers, CostTable("v", ()), RouterConfig(300_000_001, policy=policy)
    )
    assert decision.worker_id == "b"
    assert {c.worker_id for c in decision.candidates} == {"a", "b"}
    assert all(
        c.reason == "stale_snapshot" for c in decision.candidates if c.worker_id == "a"
    )
    boundary = choose(
        request,
        workers[:1],
        CostTable("v", ()),
        RouterConfig(300_000_000, policy=policy),
    )
    assert boundary.load_mode == "native_load"
