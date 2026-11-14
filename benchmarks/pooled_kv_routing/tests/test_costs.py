# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import FrozenInstanceError

# Third Party
import pytest

# First Party
from benchmarks.pooled_kv_routing.costs import CostEntry, CostKey, CostTable, estimate
from benchmarks.pooled_kv_routing.schemas import RouteRequest, WorkerSnapshot


def test_measured_cost_and_unknown_bucket() -> None:
    request = RouteRequest("r", (1, 2), 3, 0)
    worker = WorkerSnapshot("a", "http://a", True, 0, 7)
    key = CostKey(2, 3, None, None, 0, "native_load")
    table = CostTable("v1", (CostEntry(key, 11, "raw.json#trial1"),))
    assert estimate(request, worker, "native_load", table).predicted_ms == 18
    missing = estimate(request, worker, "skip_external", table)
    assert missing.predicted_ms is None
    assert missing.reason == "unknown_cost_bucket"
    with pytest.raises(FrozenInstanceError):
        table.version = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_measured_cost(value: float) -> None:
    with pytest.raises(ValueError):
        CostEntry(CostKey(2, 3, None, None, 0, "native_load"), value, "raw")


def test_provenance_and_duplicate_buckets_required() -> None:
    key = CostKey(2, 3, None, None, 0, "native_load")
    with pytest.raises(ValueError):
        CostEntry(key, 1, "")
    entry = CostEntry(key, 1, "raw")
    with pytest.raises(ValueError):
        CostTable("v", (entry, entry))


def test_skip_external_keeps_gpu_reuse_not_external_reuse() -> None:
    request = RouteRequest("r", (1, 2), 3, 0)
    worker = WorkerSnapshot("a", "http://a", True, 0, 0, 2, 1)
    table = CostTable(
        "v", (CostEntry(CostKey(2, 3, None, 1, 0, "skip_external"), 5, "raw"),)
    )
    assert estimate(request, worker, "skip_external", table).predicted_ms == 5
    concurrent = WorkerSnapshot("a", "http://a", True, 0, 0, 2, 1, 1)
    assert estimate(request, concurrent, "skip_external", table).predicted_ms is None


def test_measured_completion_work_excludes_queue_and_may_be_absent() -> None:
    key = CostKey(2, 3, None, None, 0, "native_load")
    table = CostTable("v", (CostEntry(key, 5, "raw", completion_ms=20),))
    assert table.lookup_work(key) == 20
    worker = WorkerSnapshot("a", "http://a", True, 0, 100)
    assert (
        estimate(
            RouteRequest("r", (1, 2), 3, 0), worker, "native_load", table
        ).predicted_ms
        == 105
    )
    assert CostTable("v", (CostEntry(key, 5, "raw"),)).lookup_work(key) is None
    assert table.lookup_work(CostKey(2, 3, None, None, 1, "native_load")) is None


@pytest.mark.parametrize("completion", [0, 4, float("inf"), float("nan"), True])
def test_invalid_completion_measurement(completion: float) -> None:
    with pytest.raises(ValueError):
        CostEntry(
            CostKey(2, 3, None, None, 0, "native_load"),
            5,
            "raw",
            completion_ms=completion,
        )


def test_queue_plus_service_overflow_rejected() -> None:
    request = RouteRequest("r", (1,), 1, 0)
    worker = WorkerSnapshot("a", "http://a", True, 0, 1e308)
    table = CostTable(
        "v", (CostEntry(CostKey(1, 1, None, None, 0, "native_load"), 1e308, "raw"),)
    )
    with pytest.raises(ValueError, match="predicted_ms"):
        estimate(request, worker, "native_load", table)


def test_measured_fetch_paths_are_distinct_and_skip_ignores_path() -> None:
    request = RouteRequest("r", (1, 2), 1, 0)
    local_key = CostKey(2, 1, 2, 0, 0, "native_load", fetch_path="local_host")
    remote_key = CostKey(2, 1, 2, 0, 0, "native_load", fetch_path="remote_host")
    skip_key = CostKey(2, 1, None, 0, 0, "skip_external")
    table = CostTable(
        "v",
        (
            CostEntry(local_key, 5, "local-raw", completion_ms=20),
            CostEntry(remote_key, 15, "remote-raw", completion_ms=30),
            CostEntry(skip_key, 25, "skip-raw"),
        ),
    )
    local = WorkerSnapshot("a", "http://a", True, 0, 0, 2, 0, fetch_path="local_host")
    remote = WorkerSnapshot("b", "http://b", True, 0, 0, 2, 0, fetch_path="remote_host")
    unknown = WorkerSnapshot("c", "http://c", True, 0, 0, 2, 0)
    assert estimate(request, local, "native_load", table).predicted_ms == 5
    assert estimate(request, remote, "native_load", table).predicted_ms == 15
    assert estimate(request, unknown, "native_load", table).predicted_ms is None
    assert estimate(request, local, "skip_external", table).predicted_ms == 25
    assert estimate(request, remote, "skip_external", table).predicted_ms == 25
    assert table.lookup_work(local_key) == 20
    assert table.lookup_work(remote_key) == 30
    with pytest.raises(ValueError):
        CostKey(2, 1, None, 0, 0, "skip_external", fetch_path="remote_host")
