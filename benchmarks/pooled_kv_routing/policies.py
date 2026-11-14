# SPDX-License-Identifier: Apache-2.0
"""Pure fair-input LEW, static cache/load, and measured-cost routing."""

# Standard
import hashlib

# Local
from .costs import CostTable, estimate
from .schemas import (
    STALE_NS,
    Candidate,
    Estimate,
    LoadMode,
    RouteDecision,
    RouterConfig,
    RouteRequest,
    WorkerSnapshot,
)


def _tie(request_id: str, worker_id: str, mode: LoadMode) -> bytes:
    """Return a process-independent tie key with unambiguous field boundaries."""
    return hashlib.sha256(repr((request_id, worker_id, mode)).encode()).digest()


def choose(
    request: RouteRequest,
    snapshots: tuple[WorkerSnapshot, ...],
    costs: CostTable,
    config: RouterConfig,
) -> RouteDecision:
    """Select a worker with common eligibility/freshness across all policies.

    Stale healthy workers can receive affinity-bound traffic in skip mode, but
    no stale queue/cache values enter scoring. Prefer fresh workers otherwise.
    Missing any candidate bucket falls back to the configured strong baseline;
    no prediction is invented. Baselines use the same calibrated reservations.

    Args:
        request (RouteRequest): Observed input and mandatory external affinity.
        snapshots (tuple[WorkerSnapshot, ...]): Same inputs for every strategy.
        costs (CostTable): Frozen exact measured conditions.
        config (RouterConfig): Current monotonic time and frozen policy settings.
    Returns:
        RouteDecision: Executable choice and candidate diagnostics.
    Raises:
        ValueError: Future inputs, duplicate IDs, or no healthy eligible worker.
    """
    if config.now_ns < request.arrival_ns or any(
        w.sampled_ns > config.now_ns for w in snapshots
    ):
        raise ValueError("future request or snapshot")
    if len({w.worker_id for w in snapshots}) != len(snapshots):
        raise ValueError("duplicate worker_id")
    eligible = tuple(
        w
        for w in snapshots
        if w.healthy and (request.affinity is None or w.worker_id == request.affinity)
    )
    if not eligible:
        raise ValueError("no healthy eligible worker (including affinity)")
    for worker in eligible:
        if any(
            n is not None and n > len(request.token_ids)
            for n in (worker.cached_prefix_tokens, worker.gpu_cached_prefix_tokens)
        ):
            raise ValueError("prefix exceeds prompt")
    fresh = tuple(w for w in eligible if config.now_ns - w.sampled_ns <= STALE_NS)
    diagnostics: list[Candidate] = []
    modes: tuple[LoadMode, ...] = ("native_load", "skip_external")
    for worker in sorted(eligible, key=lambda w: w.worker_id):
        for mode in modes:
            prediction = (
                Estimate(None, "stale_snapshot")
                if config.now_ns - worker.sampled_ns > STALE_NS
                else estimate(request, worker, mode, costs)
            )
            diagnostics.append(
                Candidate(
                    worker.worker_id,
                    mode,
                    prediction.predicted_ms,
                    prediction.reason,
                    worker.sampled_ns,
                )
            )
    candidates = tuple(diagnostics)
    if not fresh:
        winner = min(
            eligible,
            key=lambda w: _tie(request.request_id, w.worker_id, "skip_external"),
        )
        return RouteDecision(
            winner.worker_id,
            winner.url,
            "skip_external",
            None,
            "stale_snapshot",
            candidates,
            costs.version,
            winner.sampled_ns,
        )
    baseline_policy = config.policy if config.policy != "cost" else config.baseline
    baseline_mode = config.baseline_load_mode
    baseline = min(
        fresh,
        key=lambda w: (
            w.pending_work_ms
            - (
                config.alpha_ms_per_token
                * (
                    (
                        w.cached_prefix_tokens
                        if baseline_mode == "native_load"
                        else w.gpu_cached_prefix_tokens
                    )
                    or 0
                )
                if baseline_policy == "static"
                else 0
            ),
            _tie(request.request_id, w.worker_id, baseline_mode),
        ),
    )
    chosen = next(
        c
        for c in candidates
        if c.worker_id == baseline.worker_id and c.load_mode == baseline_mode
    )
    reason = None
    if config.policy == "cost":
        usable = tuple(c for c in candidates if c.reason != "stale_snapshot")
        scored = [(c.predicted_ms, c) for c in usable if c.predicted_ms is not None]
        if len(scored) != len(usable) or chosen.predicted_ms is None:
            reason = "unknown_cost_bucket"
        else:
            best_ms, best = min(
                scored,
                key=lambda pair: (
                    pair[0],
                    _tie(request.request_id, pair[1].worker_id, pair[1].load_mode),
                ),
            )
            if chosen.predicted_ms - best_ms > config.min_gain_ms:
                chosen = best
            else:
                reason = "insufficient_gain"
    winner = next(w for w in fresh if w.worker_id == chosen.worker_id)
    return RouteDecision(
        winner.worker_id,
        winner.url,
        chosen.load_mode,
        chosen.predicted_ms,
        reason,
        candidates,
        costs.version,
        winner.sampled_ns,
    )
