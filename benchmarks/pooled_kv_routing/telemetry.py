# SPDX-License-Identifier: Apache-2.0
"""Thread-safe confirmed observations and bounded ingress accounting.

No network collection or inferred cache residency. One monotonic clock domain
is required. The lifetime request capacity is deliberate: keeping every used ID
until the run ends detects duplicate ingress without an unbounded tombstone set.
Create a new instance for a new run; journal terminal results outside this module.
"""

# Standard
from collections import OrderedDict
from dataclasses import dataclass, replace
from itertools import chain
import threading

# Local
from .schemas import (
    STALE_NS,
    FetchPath,
    RouteRequest,
    WorkerSnapshot,
    finite_ms,
    nonnegative_int,
)


@dataclass(frozen=True)
class _Reservation:
    worker_id: str
    estimated_ms: float


@dataclass(frozen=True)
class _Prefix:
    sampled_ns: int
    gpu_tokens: int | None
    fetch_path: FetchPath | None


class Telemetry:
    """A run-scoped synchronized ledger, plus at most 16 confirmed prefixes.

    Worker observations must exclude this ledger's reservations (otherwise they
    are double counted). Catalog identity is namespace + token tuple + worker
    epoch; an epoch change invalidates all residency on that worker. Unconfirmed
    cancellation or refresh failure makes the worker unhealthy for this run.

    Args:
        workers (tuple[WorkerSnapshot, ...]): Confirmed initial worker inputs.
        namespace (str): Fixed model/tokenizer/cache namespace for this run.
        max_catalog_entries (int): Global observed prefix bound, at most 16.
        max_requests (int): Lifetime ingress ID bound; overflow rejects admission.
    Raises:
        ValueError: Invalid bounds, namespace, or duplicate workers.
    """

    def __init__(
        self,
        workers: tuple[WorkerSnapshot, ...],
        *,
        namespace: str,
        max_catalog_entries: int = 16,
        max_requests: int = 100000,
    ) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be nonempty")
        nonnegative_int(max_catalog_entries, "max_catalog_entries")
        nonnegative_int(max_requests, "max_requests")
        if not 1 <= max_catalog_entries <= 16 or max_requests == 0:
            raise ValueError("invalid capacity")
        if not workers or len({w.worker_id for w in workers}) != len(workers):
            raise ValueError("workers must be nonempty and unique")
        self._lock = threading.Lock()
        self._namespace = namespace
        self._workers = {
            w.worker_id: replace(
                w,
                cached_prefix_tokens=None,
                gpu_cached_prefix_tokens=None,
                fetch_path=None,
            )
            for w in workers
        }
        self._catalog: OrderedDict[tuple[str, tuple[int, ...]], _Prefix] = OrderedDict()
        self._reservations: dict[str, _Reservation] = {}
        self._completed: set[str] = set()
        self._tainted: set[str] = set()
        self._max_catalog_entries = max_catalog_entries
        self._max_requests = max_requests
        self._now_ns = max(w.sampled_ns for w in workers)

    def observe_worker(self, worker: WorkerSnapshot, *, epoch: str) -> None:
        """Apply confirmed queue/health evidence, excluding local reservations.

        Args:
            worker (WorkerSnapshot): Confirmed observation; prefix/path fields ignored.
            epoch (str): Nonempty worker incarnation from the observer.
        Raises:
            ValueError: Missing epoch, URL drift, time reversal or aggregate overflow.
            KeyError: Unknown worker.
        """
        if not isinstance(epoch, str) or not epoch:
            raise ValueError("epoch must be nonempty")
        with self._lock:
            previous = self._workers[worker.worker_id]
            if worker.url != previous.url:
                raise ValueError("worker URL changed")
            self._pending_work_ms(worker)
            self._advance(worker.sampled_ns)
            if previous.epoch != epoch:
                self._invalidate(worker.worker_id)
                if any(
                    r.worker_id == worker.worker_id for r in self._reservations.values()
                ):
                    self._tainted.add(worker.worker_id)
            self._workers[worker.worker_id] = replace(
                worker,
                epoch=epoch,
                cached_prefix_tokens=None,
                gpu_cached_prefix_tokens=None,
                fetch_path=None,
            )

    def confirm_prefix(
        self,
        worker_id: str,
        token_ids: tuple[int, ...],
        *,
        namespace: str,
        epoch: str,
        sampled_ns: int,
        gpu_cached_prefix_tokens: int | None = None,
        fetch_path: FetchPath | None = None,
    ) -> None:
        """Record actual residency evidence, never a prediction after dispatch.

        Args:
            worker_id (str): Registered worker.
            token_ids (tuple[int, ...]): Confirmed complete prefix contents.
            namespace (str): Must match this run's model/tokenizer namespace.
            epoch (str): Must match a confirmed worker incarnation.
            sampled_ns (int): Local observation time.
            gpu_cached_prefix_tokens (int | None): Independently verified APC count.
            fetch_path (FetchPath | None): Confirmed native host path, not an
                inference that a peer exists or is reachable. None means unknown.
        Raises:
            ValueError: Invalid tokens, namespace/epoch mismatch or time reversal.
            KeyError: Unknown worker.
        """
        RouteRequest("prefix-validation", token_ids, 1, sampled_ns)
        if fetch_path not in (None, "local_host", "remote_host"):
            raise ValueError("invalid fetch_path")
        if gpu_cached_prefix_tokens is not None:
            nonnegative_int(gpu_cached_prefix_tokens, "gpu_cached_prefix_tokens")
            if gpu_cached_prefix_tokens > len(token_ids):
                raise ValueError("GPU prefix exceeds confirmed prefix")
        with self._lock:
            worker = self._workers[worker_id]
            if namespace != self._namespace:
                raise ValueError("namespace mismatch")
            if not epoch or epoch != worker.epoch:
                raise ValueError("epoch mismatch")
            self._advance(sampled_ns)
            key = (worker_id, token_ids)
            self._catalog[key] = _Prefix(
                sampled_ns, gpu_cached_prefix_tokens, fetch_path
            )
            self._catalog.move_to_end(key)
            if len(self._catalog) > self._max_catalog_entries:
                self._catalog.popitem(last=False)

    def reserve(
        self, request: RouteRequest, worker_id: str, estimated_ms: float, now_ns: int
    ) -> None:
        """Reserve ingress work atomically before dispatch, without cache claims.

        Args:
            request (RouteRequest): Unique run-scoped request with observed arrival.
            worker_id (str): Selected healthy worker, respecting request affinity.
            estimated_ms (float): Independently calibrated completion work,
                excluding already queued work (CostTable.lookup_work). Never
                reserve a queue-inclusive TTFT prediction or prefill-only cost.
            now_ns (int): Current observation clock.
        Raises:
            ValueError: Duplicate, capacity, unavailable worker, affinity, time or
                unrepresentable aggregate work. Rejection leaves state unchanged.
            KeyError: Unknown worker.
        """
        finite_ms(estimated_ms, "estimated_ms")
        with self._lock:
            if (
                request.request_id in self._reservations
                or request.request_id in self._completed
            ):
                raise ValueError("duplicate request_id")
            if len(self._reservations) + len(self._completed) >= self._max_requests:
                raise ValueError("request ledger capacity exceeded")
            worker = self._workers[worker_id]
            if worker_id in self._tainted or not worker.healthy:
                raise ValueError("worker unavailable")
            if request.affinity is not None and request.affinity != worker_id:
                raise ValueError("affinity mismatch")
            if now_ns < request.arrival_ns:
                raise ValueError("future request")
            self._pending_work_ms(worker, estimated_ms)
            self._advance(now_ns)
            self._reservations[request.request_id] = _Reservation(
                worker_id, estimated_ms
            )

    def confirmed_complete(self, request_id: str, now_ns: int) -> None:
        """Release work only after confirmed worker termination (any status).

        Args:
            request_id (str): Previously reserved nonterminal request.
            now_ns (int): Confirmation time, not client disconnect time.
        Raises:
            ValueError: Duplicate terminal or time reversal.
            KeyError: Unknown request ID.
        """
        with self._lock:
            if request_id in self._completed:
                raise ValueError("duplicate terminal")
            self._reservations[request_id]
            self._advance(now_ns)
            del self._reservations[request_id]
            self._completed.add(request_id)

    def cancel_unconfirmed(self, request_id: str, now_ns: int) -> None:
        """Taint a worker and retain all reserved work pending terminal evidence.

        Args:
            request_id (str): Active reservation with unconfirmed cancellation.
            now_ns (int): Time cancellation became uncertain.
        Raises:
            ValueError: Already terminal request or time reversal.
            KeyError: Unknown request ID.
        """
        with self._lock:
            if request_id in self._completed:
                raise ValueError("already terminal")
            reservation = self._reservations[request_id]
            self._advance(now_ns)
            self._tainted.add(reservation.worker_id)
            self._invalidate(reservation.worker_id)

    def refresh_failed(self, worker_id: str, now_ns: int) -> None:
        """Invalidate residency and stop admission after a failed state refresh.

        Args:
            worker_id (str): Registered worker with missing evidence.
            now_ns (int): Failure observation time.
        Raises:
            ValueError: Time reversal.
            KeyError: Unknown worker.
        """
        with self._lock:
            self._workers[worker_id]
            self._advance(now_ns)
            self._tainted.add(worker_id)
            self._invalidate(worker_id)

    def snapshot(
        self, now_ns: int, request: RouteRequest | None = None
    ) -> tuple[WorkerSnapshot, ...]:
        """Return atomic request-specific confirmed state, never future history.

        Args:
            now_ns (int): Current local clock, monotonic over all calls.
            request (RouteRequest | None): Prefix lookup input, absent means unknown.
        Returns:
            tuple[WorkerSnapshot, ...]: Stable worker order with pending ingress.
        Raises:
            ValueError: Time reversal or request arriving in the future.
        """
        with self._lock:
            if request is not None and request.arrival_ns > now_ns:
                raise ValueError("future request")
            self._advance(now_ns)
            result = []
            for worker_id, worker in sorted(self._workers.items()):
                active = [
                    r for r in self._reservations.values() if r.worker_id == worker_id
                ]
                prefix: int | None = None
                gpu: int | None = None
                fetch_path: FetchPath | None = None
                if request is not None:
                    for (owner, tokens), entry in self._catalog.items():
                        if (
                            owner == worker_id
                            and now_ns - entry.sampled_ns <= STALE_NS
                            and request.token_ids[: len(tokens)] == tokens
                        ):
                            if prefix is None or len(tokens) > prefix:
                                prefix = len(tokens)
                                fetch_path = entry.fetch_path
                            if entry.gpu_tokens is not None:
                                gpu = max(gpu or 0, entry.gpu_tokens)
                result.append(
                    replace(
                        worker,
                        healthy=worker.healthy and worker_id not in self._tainted,
                        pending_work_ms=self._pending_work_ms(worker),
                        pending_requests=worker.pending_requests + len(active),
                        cached_prefix_tokens=prefix,
                        gpu_cached_prefix_tokens=gpu,
                        fetch_path=fetch_path,
                    )
                )
            return tuple(result)

    def _pending_work_ms(self, worker: WorkerSnapshot, extra_ms: float = 0.0) -> float:
        """Compute the snapshot aggregate before committing admission/observation.

        Caller holds the lock. Appending the proposed work uses the same summation
        order as the eventual snapshot; individual finite values can sum to inf.
        """
        reserved = sum(
            chain(
                (
                    r.estimated_ms
                    for r in self._reservations.values()
                    if r.worker_id == worker.worker_id
                ),
                (extra_ms,),
            )
        )
        total = worker.pending_work_ms + reserved
        finite_ms(total, "pending_work_ms")
        return total

    def _advance(self, now_ns: int) -> None:
        nonnegative_int(now_ns, "time")
        if now_ns < self._now_ns:
            raise ValueError("time reversal: historical state is not retained")
        self._now_ns = now_ns

    def _invalidate(self, worker_id: str) -> None:
        self._catalog = OrderedDict(
            (key, value) for key, value in self._catalog.items() if key[0] != worker_id
        )
