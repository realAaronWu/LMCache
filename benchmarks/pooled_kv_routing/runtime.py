# SPDX-License-Identifier: Apache-2.0
"""Connect pure routing to the stream proxy without owning device resources."""

# Standard
from collections.abc import Callable
from dataclasses import replace
from typing import cast
import time

# Local
from .costs import CostKey, CostTable
from .policies import choose
from .proxy import ProxyFinish, ProxyRoute
from .schemas import RouterConfig, RouteRequest
from .telemetry import Telemetry


class RoutingRuntime:
    """Single-event-loop bridge with atomic choose/reserve before dispatch.

    The caller supplies confirmed worker observations and measured cost entries.
    This class does not infer residency or refresh observation timestamps. It
    must not run across event loops or threads. No device cancellation, P2P
    safety, calibration validity or performance qualification is implied.

    Args:
        telemetry: Run-scoped ledger and confirmed external observations.
        costs: Shared frozen measurements, including completion work.
        config: Frozen policy parameters; now_ns is set from the local clock.
        clock: Local monotonic nanoseconds, injectable for deterministic tests.
    """

    def __init__(
        self,
        telemetry: Telemetry,
        costs: CostTable,
        config: RouterConfig,
        *,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._telemetry = telemetry
        self._costs = costs
        self._config = config
        self._clock = clock

    async def route(self, body: dict[str, object]) -> ProxyRoute:
        """Select and reserve measured completion work for validated proxy input.

        Args:
            body: Proxy completion JSON, including its validated request ID.

        Returns:
            ProxyRoute containing the executable endpoint and full diagnostics.

        Raises:
            ValueError: Invalid input, no eligible worker or unknown work cost.
            KeyError: Missing required request fields.

        Notes:
            Work uses the isolated (zero-pending) completion measurement, not
            queue-inclusive predicted TTFT. Existing pending work is reserved
            separately. Client arrival_ns is not this host's clock.
        """
        now = self._clock()
        request = RouteRequest(
            cast(str, body["request_id"]),
            tuple(cast(list[int], body["prompt"])),
            cast(int, body["max_tokens"]),
            now,
            cast(str | None, body.get("affinity")),
        )
        snapshots = self._telemetry.snapshot(now, request)
        decision = choose(
            request, snapshots, self._costs, replace(self._config, now_ns=now)
        )
        worker = next(w for w in snapshots if w.worker_id == decision.worker_id)
        work_key = CostKey(
            len(request.token_ids),
            request.max_tokens,
            worker.cached_prefix_tokens
            if decision.load_mode == "native_load"
            else None,
            worker.gpu_cached_prefix_tokens,
            0,
            decision.load_mode,
            worker.fetch_path if decision.load_mode == "native_load" else None,
        )
        work_ms = self._costs.lookup_work(work_key)
        if work_ms is None:
            raise ValueError("no measured isolated completion work for admission")
        self._telemetry.reserve(request, decision.worker_id, work_ms, now)
        return ProxyRoute(
            decision.worker_id,
            decision.url,
            decision.load_mode,
            decision.as_dict() | {"reserved_completion_ms": work_ms},
        )

    async def finish(self, result: ProxyFinish) -> None:
        """Settle only confirmed completion; retain and taint uncertain work.

        Args:
            result: The proxy's exactly-once terminal callback.

        Raises:
            ValueError: Invalid identity/confirmation, owner mismatch or
                duplicate confirmed terminal.
            KeyError: A selected request was never reserved.

        Notes:
            Uncertain notifications are idempotent. Later real terminal
            evidence may release retained work but does not clear taint.
            An unselected failed route with no reservation is a no-op.
        """
        if (
            not isinstance(result.request_id, str)
            or not result.request_id
            or not isinstance(result.worker_id, str)
            or type(result.worker_terminal_confirmed) is not bool
        ):
            raise ValueError("invalid completion identity or confirmation type")
        now = self._clock()
        if result.worker_terminal_confirmed:
            if not result.worker_id:
                raise ValueError("confirmed completion requires worker identity")
            self._telemetry.confirmed_complete(
                result.request_id, now, worker_id=result.worker_id
            )
        else:
            try:
                self._telemetry.cancel_unconfirmed(
                    result.request_id, now, worker_id=result.worker_id
                )
            except KeyError:
                if result.worker_id:
                    raise
