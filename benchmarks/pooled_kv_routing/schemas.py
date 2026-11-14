# SPDX-License-Identifier: Apache-2.0
"""Validated immutable inputs and JSON-safe routing diagnostics."""

# Standard
from dataclasses import asdict, dataclass
from typing import Literal
import math

LoadMode = Literal["native_load", "skip_external"]
Policy = Literal["cost", "lew", "static"]
STALE_NS = 300_000_000


def nonnegative_int(value: int, name: str) -> None:
    """Validate an integer boundary value; raise ValueError for bool or negatives."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def finite_ms(value: float, name: str) -> None:
    """Validate finite nonnegative milliseconds; raise ValueError on bad input."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite nonnegative milliseconds")


@dataclass(frozen=True)
class RouteRequest:
    """Observed request input; affinity is a required worker ID, not a hint."""

    request_id: str
    token_ids: tuple[int, ...]
    max_tokens: int
    arrival_ns: int
    affinity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be nonempty")
        if type(self.token_ids) is not tuple or not self.token_ids:
            raise ValueError("token_ids must be a nonempty tuple")
        for token in self.token_ids:
            nonnegative_int(token, "token_id")
        nonnegative_int(self.max_tokens, "max_tokens")
        if not self.max_tokens:
            raise ValueError("max_tokens must be positive")
        nonnegative_int(self.arrival_ns, "arrival_ns")
        if self.affinity is not None and (
            not isinstance(self.affinity, str) or not self.affinity
        ):
            raise ValueError("affinity must be a worker ID")


@dataclass(frozen=True)
class WorkerSnapshot:
    """Confirmed worker observation; unknown prefix counts remain None.

    pending_work_ms/pending_requests include ingress reservations. Prefix counts
    describe the current request only; a caller must not reuse them for another.
    sampled_ns is the observation time, never refreshed by local reservation.
    """

    worker_id: str
    url: str
    healthy: bool
    sampled_ns: int
    pending_work_ms: float
    cached_prefix_tokens: int | None = None
    gpu_cached_prefix_tokens: int | None = None
    pending_requests: int = 0
    epoch: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.worker_id, str)
            or not self.worker_id
            or not isinstance(self.url, str)
            or not self.url
        ):
            raise ValueError("worker_id and url must be nonempty")
        if type(self.healthy) is not bool:
            raise ValueError("healthy must be bool")
        nonnegative_int(self.sampled_ns, "sampled_ns")
        finite_ms(self.pending_work_ms, "pending_work_ms")
        nonnegative_int(self.pending_requests, "pending_requests")
        for name in ("cached_prefix_tokens", "gpu_cached_prefix_tokens"):
            value = getattr(self, name)
            if value is not None:
                nonnegative_int(value, name)
        if self.epoch is not None and (
            not isinstance(self.epoch, str) or not self.epoch
        ):
            raise ValueError("epoch must be nonempty")


@dataclass(frozen=True)
class RouterConfig:
    """Frozen development-set parameters; now_ns uses the snapshot clock."""

    now_ns: int
    policy: Policy = "cost"
    baseline: Literal["lew", "static"] = "lew"
    alpha_ms_per_token: float = 0.0
    min_gain_ms: float = 0.0
    baseline_load_mode: LoadMode = "native_load"

    def __post_init__(self) -> None:
        nonnegative_int(self.now_ns, "now_ns")
        if self.policy not in ("cost", "lew", "static") or self.baseline not in (
            "lew",
            "static",
        ):
            raise ValueError("unsupported policy or baseline")
        finite_ms(self.alpha_ms_per_token, "alpha_ms_per_token")
        finite_ms(self.min_gain_ms, "min_gain_ms")
        if self.baseline_load_mode not in ("native_load", "skip_external"):
            raise ValueError("unsupported baseline_load_mode")


@dataclass(frozen=True)
class Estimate:
    """Unknown estimates carry None and a diagnostic, never a synthetic zero."""

    predicted_ms: float | None
    reason: str | None = None


@dataclass(frozen=True)
class Candidate:
    """One legal worker/mode estimate, tied to its observation time."""

    worker_id: str
    load_mode: LoadMode
    predicted_ms: float | None
    reason: str | None
    sampled_ns: int


@dataclass(frozen=True)
class RouteDecision:
    """Executable selection and immutable evidence for the parent journal."""

    worker_id: str
    url: str
    load_mode: LoadMode
    predicted_ms: float | None
    fallback_reason: str | None
    candidates: tuple[Candidate, ...]
    cost_version: str
    sampled_ns: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe fields, including all legal candidate diagnostics.

        Returns:
            dict[str, object]: Dataclass fields; candidates are JSON arrays.
        """
        result = asdict(self)
        result["candidates"] = [asdict(item) for item in self.candidates]
        return result
