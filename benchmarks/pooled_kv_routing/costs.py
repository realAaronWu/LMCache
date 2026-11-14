# SPDX-License-Identifier: Apache-2.0
"""Frozen exact-bucket measured costs; no fitting, interpolation or collection."""

# Standard
from dataclasses import dataclass

# Local
from .schemas import (
    Estimate,
    LoadMode,
    RouteRequest,
    WorkerSnapshot,
    finite_ms,
    nonnegative_int,
)


@dataclass(frozen=True)
class CostKey:
    """Measured service condition, including output budget and concurrency.

    None denotes a specifically measured unknown-prefix condition, not zero.
    skip_external keys require cached_prefix_tokens=None but retain GPU reuse.
    """

    prompt_tokens: int
    max_tokens: int
    cached_prefix_tokens: int | None
    gpu_cached_prefix_tokens: int | None
    pending_requests: int
    load_mode: LoadMode

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "max_tokens", "pending_requests"):
            nonnegative_int(getattr(self, name), name)
        if not self.prompt_tokens or not self.max_tokens:
            raise ValueError("prompt_tokens and max_tokens must be positive")
        for count in (self.cached_prefix_tokens, self.gpu_cached_prefix_tokens):
            if count is not None:
                nonnegative_int(count, "prefix_tokens")
                if count > self.prompt_tokens:
                    raise ValueError("prefix exceeds prompt")
        if self.load_mode not in ("native_load", "skip_external"):
            raise ValueError("invalid load_mode")
        if self.load_mode == "skip_external" and self.cached_prefix_tokens is not None:
            raise ValueError("skip_external ignores external prefix")


@dataclass(frozen=True)
class CostEntry:
    """Measured whole service-to-first-token ms with a raw-evidence reference.

    service_ms excludes the separately accounted pending_work_ms. It is not a
    sum of overlapping transfer/prefill means. Provenance is an opaque nonempty
    reference; verifying raw measurements belongs to calibration/preflight.
    completion_ms includes decode through termination but excludes queued work;
    None means this entry cannot supply a reservation work estimate.
    """

    key: CostKey
    service_ms: float
    provenance: str
    completion_ms: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, CostKey):
            raise ValueError("key must be CostKey")
        finite_ms(self.service_ms, "service_ms")
        if self.service_ms == 0:
            raise ValueError("service_ms must be positive")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("provenance must reference measured evidence")
        if self.completion_ms is not None:
            finite_ms(self.completion_ms, "completion_ms")
            if self.completion_ms < self.service_ms:
                raise ValueError("completion_ms must be >= service_ms")


@dataclass(frozen=True)
class CostTable:
    """Immutable versioned measured entries; absent keys are unsupported."""

    version: str
    entries: tuple[CostEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be nonempty")
        if type(self.entries) is not tuple or any(
            not isinstance(entry, CostEntry) for entry in self.entries
        ):
            raise ValueError("entries must be tuple[CostEntry, ...]")
        if len({entry.key for entry in self.entries}) != len(self.entries):
            raise ValueError("duplicate cost bucket")

    def lookup(self, key: CostKey) -> float | None:
        """Look up an exact condition; return None when not measured.

        Args:
            key (CostKey): Complete measured service condition.
        Returns:
            float | None: Positive measured ms or unsupported.
        """
        return next(
            (entry.service_ms for entry in self.entries if entry.key == key), None
        )

    def lookup_work(self, key: CostKey) -> float | None:
        """Look up measured completion work, excluding all queued work.

        Args:
            key (CostKey): Exact measured condition used for this admission.
        Returns:
            float | None: Whole-request completion ms, or None when unmeasured.
                Never substitute lookup() or a queue-inclusive TTFT prediction.
        """
        return next(
            (entry.completion_ms for entry in self.entries if entry.key == key), None
        )


def estimate(
    request: RouteRequest, worker: WorkerSnapshot, mode: LoadMode, costs: CostTable
) -> Estimate:
    """Predict queued work plus measured service without extrapolation.

    Args:
        request (RouteRequest): Current observed request.
        worker (WorkerSnapshot): Request-specific prefix and remaining work.
        mode (LoadMode): Native loading or external-load bypass, not APC bypass.
        costs (CostTable): Frozen service costs for this environment.
    Returns:
        Estimate: Finite prediction or an explicit unknown-bucket diagnostic.
    Raises:
        ValueError: Invalid mode or prefix count exceeding the prompt.
    """
    key = CostKey(
        len(request.token_ids),
        request.max_tokens,
        worker.cached_prefix_tokens if mode == "native_load" else None,
        worker.gpu_cached_prefix_tokens,
        worker.pending_requests,
        mode,
    )
    service_ms = costs.lookup(key)
    if service_ms is None:
        return Estimate(None, "unknown_cost_bucket")
    total = worker.pending_work_ms + service_ms
    finite_ms(total, "predicted_ms")
    return Estimate(total)
