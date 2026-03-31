# Design: Profiling-Based KV Cache Orchestration Plugin

## 1. Goals and Constraints

### Goals

1. Add a **Profiling Engine** that observes KV cache access patterns at runtime.
2. Add a **Cache Orchestrator** that uses profiling data to make cost-aware eviction, placement, and prefetch decisions.
3. Both components are **optional plugins** -- when not configured, LMCache behaves identically to today with zero overhead.

### Constraints

- **Minimize changes to existing code.** The existing `StorageManager`, `LocalCPUBackend`, `LMCacheEngine`, and all storage backends must not be restructured.
- **Zero overhead when disabled.** No extra imports, no background threads, no per-access cost beyond a single null-pointer check.
- **Bounded overhead when enabled.** Per-access instrumentation must cost < 100 ns. Background aggregation must not contend with the inference hot path.
- **Follow existing patterns.** Use the same extension mechanisms LMCache already provides: `BaseCachePolicy` for eviction, `extra_config` for plugin settings, lazy imports for optional modules.

## 2. Why a Plugin, Not a Core Change

| Alternative considered | Why rejected |
|---|---|
| Modify `StorageManager` to always profile | Adds overhead for users who don't need it. Violates the principle that LMCache should be lightweight by default. |
| Add a new `StorageBackendInterface` wrapper | Storage backends have diverse interfaces (allocator vs. non-allocator, sync vs. async). Wrapping all of them introduces fragile coupling. |
| Fork `LocalCPUBackend` into a profiling variant | Code duplication. Any future fix to `LocalCPUBackend` must be applied twice. |
| **Use `BaseCachePolicy` as the plugin surface** | **Chosen.** The eviction policy is already a pluggable interface. `LocalCPUBackend` calls `self.cache_policy.get_evict_candidates()` -- replacing this policy is the narrowest possible integration point for controlling eviction. |

The `BaseCachePolicy` interface is the natural insertion point because:
- It is already a factory-based abstraction (`get_cache_policy(name)`).
- `LocalCPUBackend` holds a reference to it and calls it under `cpu_lock` -- thread safety is handled by the caller, not the policy.
- Every eviction decision flows through `get_evict_candidates()` -- replacing this method is sufficient to change all eviction behavior.

For placement and prefetch (which `BaseCachePolicy` does not cover), we add thin opt-in hooks in `StorageManager` guarded by null checks. This follows the same pattern as the existing `_bypass_lock` / `_bypassed_backends` mechanism -- a field that is `None` or empty by default and only activated when configured.

## 3. Architecture

```
                  LMCacheEngine
                       |
                       v
               StorageManager
              /       |        \
             /        |         \
    LocalCPU    LocalDisk    RemoteBackend ...
    Backend      Backend
       |
       |  cache_policy interface
       v
  ┌─────────────────────────────┐
  │  ProfilingCachePolicy       │  <-- new, implements BaseCachePolicy
  │                             │
  │  ┌───────────────────────┐  │
  │  │   Profiling Engine    │  │  <-- new, runs background aggregation
  │  └───────────┬───────────┘  │
  │              │ profiling    │
  │              v  data        │
  │  ┌───────────────────────┐  │
  │  │   Cache Orchestrator  │  │  <-- new, scoring + prefetch + placement
  │  └───────────────────────┘  │
  └─────────────────────────────┘
```

### Why the Profiling Engine and Orchestrator live inside the policy object

- **Single owner.** The policy is created by `LocalCPUBackend.__init__()` via `get_cache_policy()`. If the policy owns the engine and orchestrator, their lifetime is tied to the backend's lifetime -- no separate init/shutdown coordination needed.
- **No circular references.** The policy holds the engine and orchestrator. The `StorageManager` holds a *weak reference* (or a thin callback interface) to the orchestrator for placement and prefetch hooks. The orchestrator never holds a reference back to `StorageManager`.
- **Testable in isolation.** Unit tests can create a `ProfilingCachePolicy` with a mock engine, without instantiating `StorageManager` or any storage backend.

## 4. Component Design

### 4.1 Profiling Engine

**Responsibility:** Collect and aggregate per-chunk access telemetry.

**Data structures (per chunk):**

| Field | Type | Size | Updated on |
|---|---|---|---|
| `access_count` | `int` | 8 bytes | Every hit/put (atomic increment) |
| `last_access_ts` | `float` | 8 bytes | Every hit (timestamp) |
| `recomputation_cost` | `float` | 8 bytes | First put only (token_count * per_token_latency) |
| `first_seen_ts` | `float` | 8 bytes | First put only |

Total: 32 bytes per chunk. At 1 million cached chunks, this is 32 MB -- negligible relative to the KV cache itself.

**Background aggregation thread:**

Runs every `aggregation_interval` seconds (default: 1.0). Computes:
- **Concentration ratio:** Fraction of total accesses going to the top-K% chunks.
- **New-chunk rate:** Count of chunks first seen in this window.
- **Phase label:** "high-reuse" or "sequential", determined by thresholding the concentration ratio.

**Design choice: Why a background thread, not inline computation.**

Inline aggregation (computing histograms on every access) would add O(N) work to the hot path. The background thread decouples collection (O(1) per access: increment counter, update timestamp) from aggregation (O(N) per interval: scan all counters). At 1-second intervals, the aggregation thread processes ~1M chunks in < 10 ms, well within budget.

**Design choice: Why sampling for reuse distance.**

Exact reuse distance tracking requires maintaining a stack of all accesses -- O(N) space and O(log N) per access. We sample 1 in every K accesses (default K=100) and compute reuse distance only for sampled accesses using a compact sketch. This bounds the per-access overhead to ~10 ns (a modulo check + conditional branch) at the cost of approximate reuse distance distributions, which is sufficient for phase detection and eviction scoring.

### 4.2 Cache Orchestrator

**Responsibility:** Use profiling data to make eviction, placement, and prefetch decisions.

**Eviction scoring (called from `get_evict_candidates`):**

```
score(chunk) = w1 * recency(chunk)
             + w2 * frequency(chunk)
             + w3 * recomp_cost(chunk)
             + w4 * size(chunk)
```

Where `w1..w4` are adjusted on phase transitions:

| Weight | High-reuse phase | Sequential phase |
|---|---|---|
| w1 (recency) | 0.2 | 0.4 |
| w2 (frequency) | 0.4 | 0.1 |
| w3 (recomp cost) | 0.3 | 0.4 |
| w4 (size) | 0.1 | 0.1 |

Candidates with the **highest** score (least valuable) are evicted first.

**Design choice: Why a weighted linear combination, not a learned model.**

A linear scorer is O(1) per candidate, deterministic, explainable, and has no training data requirement. It runs inside `LocalCPUBackend.cpu_lock` -- any overhead here directly blocks inference. A learned model (e.g., neural predictor) would add milliseconds of latency per eviction decision, which is unacceptable. The linear scorer with 4 multiplications and 3 additions costs < 20 ns per candidate.

**Design choice: Why weights are phase-switched, not continuously optimized.**

Continuous online optimization (e.g., gradient descent on hit rate) requires a differentiable objective and introduces instability during convergence. Phase-switching uses two fixed weight vectors selected by the phase detector. This is simpler, more predictable, and sufficient for the two dominant workload patterns (high-reuse vs. sequential). Additional phase profiles can be added as static configurations without changing the scoring logic.

**Tier-aware demotion (called from `get_evict_candidates`):**

When the orchestrator selects an eviction candidate, it checks:

```
if chunk.recomputation_cost > demotion_threshold:
    return DemoteToNVMe(chunk)    # write to lower tier, don't discard
else:
    return Discard(chunk)          # free memory, chunk can be recomputed
```

**Design choice: Why demotion is decided at eviction time, not placement time.**

At placement time (put), we don't yet know how frequently the chunk will be accessed. Deciding at eviction time uses the accumulated profiling data, giving a more informed decision. This also avoids slowing down the put path with demotion logic.

**Prefetch scheduling:**

The orchestrator maintains a priority queue of chunks that are likely to be accessed soon, ranked by `P(access) / fetch_latency(current_tier)`. A background thread drains this queue and issues `get_blocking` + `submit_put_task` to promote chunks from slow tiers to DRAM.

**Design choice: Why prefetch runs in a separate thread, not in the aggregation thread.**

Prefetch involves blocking I/O (disk reads, network fetches). Running it in the aggregation thread would delay phase detection. A separate daemon thread isolates prefetch latency from profiling latency.

### 4.3 ProfilingCachePolicy

**Responsibility:** Implement `BaseCachePolicy` and bridge the existing interface to the Profiling Engine and Cache Orchestrator.

```python
class ProfilingCachePolicy(BaseCachePolicy[KeyType, OrderedDict[KeyType, Any]]):

    def __init__(self, config: dict):
        self.engine = ProfilingEngine(config)
        self.orchestrator = CacheOrchestrator(self.engine, config)

    def init_mutable_mapping(self) -> OrderedDict:
        return OrderedDict()

    def update_on_hit(self, key, cache_dict):
        cache_dict.move_to_end(key)               # preserve LRU ordering as fallback
        self.engine.record_hit(key)                # update counters

    def update_on_put(self, key):
        self.engine.record_put(key)

    def update_on_force_evict(self, key):
        self.engine.record_evict(key)

    def get_evict_candidates(self, cache_dict, num_candidates=1):
        return self.orchestrator.select_eviction_candidates(
            cache_dict, num_candidates
        )
```

**Design choice: Why the policy preserves LRU ordering (`move_to_end`) alongside profiling.**

If the orchestrator's scoring data is stale (e.g., during the first few seconds before the first aggregation), the `OrderedDict` ordering provides a sound LRU fallback. The orchestrator's `select_eviction_candidates` can optionally fall back to iterating the dict in LRU order when profiling data is insufficient, exactly as the existing `LRUCachePolicy` does.

## 5. Integration Points

### 5.1 Cache Policy Factory (1 change)

**File:** `lmcache/v1/storage_backend/cache_policy/__init__.py`

**Change:** Add a lazy-import branch for `"PROFILING"`. No new static imports.

```python
def get_cache_policy(policy_name: str, config: Optional[dict] = None) -> BaseCachePolicy:
    upper = policy_name.upper()
    if upper == "PROFILING":
        from lmcache.v1.cache_orchestration.profiling_policy import (
            ProfilingCachePolicy,
        )
        return ProfilingCachePolicy(config or {})
    # ... existing code unchanged ...
```

**Why lazy import:** The `cache_orchestration` package may have heavier dependencies (e.g., `sortedcontainers` for the priority queue). Lazy importing ensures these are never loaded when profiling is not requested.

**Why pass `config`:** The profiling policy needs tuning parameters (aggregation interval, sample rate, weight vectors). These come from `extra_config` and must be threaded through the factory. The existing policies ignore this parameter (they take no constructor args), so adding it as `Optional[dict] = None` is backward-compatible.

### 5.2 StorageManager Access Hooks (2 changes)

**File:** `lmcache/v1/storage_backend/storage_manager.py`

**Change 1: Add an optional profiling callback field.**

In `__init__`, add:
```python
self._on_access: Optional[Callable] = None    # set by ProfilingCachePolicy.register()
```

**Change 2: Call the callback in `batched_get` after line 507.**

```python
if self._on_access is not None:
    self._on_access(keys, backend_name)
```

**Why a callback, not a direct reference to ProfilingEngine:**

A callback decouples `StorageManager` from the profiling module. `StorageManager` never imports anything from `cache_orchestration`. The callback is set by the profiling policy during initialization via a `register(storage_manager)` call. If the policy is not profiling-aware, `_on_access` remains `None` and the check costs < 1 ns.

**Why only in `batched_get`, not `batched_put`:**

The `update_on_put` hook in `BaseCachePolicy` already captures put events. Adding a redundant hook in `batched_put` would double-count. The `batched_get` hook captures which backend served each request (hit tier information), which the policy's `update_on_hit` does not have visibility into.

### 5.3 Prefetch Thread (1 change)

**File:** `lmcache/v1/storage_backend/storage_manager.py`

**Change:** In `__init__`, conditionally start a prefetch thread:

```python
self._prefetch_thread: Optional[Thread] = None

def enable_prefetch(self, orchestrator, interval: float = 0.5):
    self._orchestrator = orchestrator
    self._prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
    self._prefetch_thread.start()
```

**Why a daemon thread:** Daemon threads are automatically killed when the process exits, avoiding shutdown coordination. The prefetch loop uses a `threading.Event` for clean early termination on `close()`.

**Why the orchestrator calls `enable_prefetch`, not `StorageManager.__init__`:**

`StorageManager.__init__` runs before the cache policy is created (the policy is created inside `LocalCPUBackend.__init__`, which is called from `CreateStorageBackends`, which is called from `StorageManager.__init__`). By the time the policy exists and can register, `__init__` has already completed. The `enable_prefetch` method allows deferred registration after construction.

### 5.4 Placement Routing (1 change, optional)

**File:** `lmcache/v1/storage_backend/storage_manager.py`

**Change:** In `batched_put`, optionally filter target backends:

```python
# Before the backend loop:
skip_backends = set()
if self._orchestrator is not None:
    skip_backends = self._orchestrator.get_skip_backends(keys)

# Inside the loop, add:
if backend_name in skip_backends:
    continue
```

**Why `skip_backends` rather than `select_backends`:**

The default behavior (write to all backends) is correct and safe. The orchestrator only needs to *exclude* backends for cold chunks that should bypass DRAM. A skip-set is simpler and preserves the existing iteration order. If the orchestrator returns an empty set, behavior is identical to today.

**Why this is optional:**

This hook is the lowest-priority integration. Eviction scoring (5.1) and prefetch (5.3) provide the majority of the value. Placement routing can be added in a follow-up without changing the design.

## 6. Overhead Analysis

### When disabled (`cache_policy: "LRU"`)

| Path | Additional cost | Source |
|---|---|---|
| `batched_get` hot path | `if self._on_access is not None:` | 1 comparison, < 1 ns |
| `batched_put` hot path | `if backend_name in skip_backends:` | Only if placement routing is added; 1 set lookup, < 5 ns |
| Eviction path | None | `LRUCachePolicy` is used, unchanged |
| Background threads | None | No threads started |
| Memory | None | No profiling data structures allocated |
| Imports | None | `cache_orchestration` package never imported |

**Total: < 2 ns per request on any hot path. Effectively zero.**

### When enabled (`cache_policy: "profiling"`)

| Path | Additional cost | Source |
|---|---|---|
| `update_on_hit` (per chunk access) | ~50 ns | Atomic counter increment + timestamp write + 1% chance of reuse distance sample |
| `update_on_put` (per chunk store) | ~30 ns | Counter increment + timestamp + recomputation cost annotation |
| `get_evict_candidates` (per eviction) | ~20 ns per candidate | 4 multiplications + 3 additions for composite score |
| `batched_get` callback | ~100 ns per batch | Record backend name + key list into ring buffer |
| Background aggregation thread | ~10 ms per 1M chunks | Scans all chunk metadata every `aggregation_interval` (1s) |
| Background prefetch thread | Variable | Bounded by `prefetch_budget` (configurable bytes/sec) |
| Memory | ~32 bytes per chunk | Per-chunk metadata (counters + timestamps + cost) |

**Hot-path budget:** The inference-critical path is `update_on_hit` at ~50 ns per chunk. A typical retrieve operation touches 10-50 chunks, adding 0.5-2.5 us total. For comparison, a single GPU kernel launch is ~5 us and an RDMA operation is ~2 us. The profiling overhead is well within noise.

**Background thread budget:** The aggregation thread runs every 1 second and takes ~10 ms for 1M chunks. This consumes < 1% of one CPU core. The prefetch thread issues I/O at a rate-limited budget, using bandwidth that would otherwise be idle.

## 7. Configuration

All profiling settings go under `extra_config`, following the existing LMCache pattern:

```yaml
cache_policy: "profiling"
extra_config:
  # Profiling Engine
  profiling.aggregation_interval: 1.0       # seconds between background scans
  profiling.reuse_distance_sample_rate: 100  # sample 1 in K accesses
  profiling.per_token_prefill_latency: 0.001 # seconds, calibrated at startup

  # Phase Detection
  profiling.phase.concentration_threshold: 0.3  # delta to trigger phase switch
  profiling.phase.high_reuse_weights: "0.2,0.4,0.3,0.1"  # w1,w2,w3,w4
  profiling.phase.sequential_weights: "0.4,0.1,0.4,0.1"

  # Prefetch
  profiling.prefetch.enabled: true
  profiling.prefetch.interval: 0.5            # seconds between prefetch sweeps
  profiling.prefetch.budget_mb_per_sec: 500   # rate limit

  # Demotion
  profiling.demotion.enabled: true
  profiling.demotion.cost_threshold: 10.0     # seconds of recomp cost to trigger demotion
```

**Why `extra_config` rather than top-level config fields:**

Top-level fields require changes to `LMCacheEngineConfig` and its validation logic. `extra_config` is a pass-through dict that any plugin can read without modifying the config schema. This is the same pattern used by `enable_nixl_storage`, `audit_backend_enabled`, and all storage plugins.

## 8. New Files

```
lmcache/v1/cache_orchestration/            # new package
├── __init__.py                            # empty, marks package
├── profiling_engine.py                    # ProfilingEngine class
├── cache_orchestrator.py                  # CacheOrchestrator class
└── profiling_policy.py                    # ProfilingCachePolicy (BaseCachePolicy impl)
```

## 9. Changed Files

| File | Change | Lines added | Lines modified |
|---|---|---|---|
| `cache_policy/__init__.py` | Lazy-import branch for `"PROFILING"` | ~5 | 0 |
| `storage_manager.py` | `_on_access` callback field + null-guarded call in `batched_get` | ~6 | 0 |
| `storage_manager.py` | `enable_prefetch()` method + daemon thread | ~15 | 0 |
| `storage_manager.py` | (Optional) `skip_backends` check in `batched_put` | ~4 | 0 |
| **Total** | | **~30** | **0** |

No existing lines are modified. All changes are additive: new fields, new methods, new branches.

## 10. Testing Strategy

### Unit tests (no infrastructure required)

- `ProfilingEngine`: Test counter increments, reuse distance sampling, phase detection triggers.
- `CacheOrchestrator`: Test eviction scoring with known inputs, weight switching on phase change, demotion threshold logic.
- `ProfilingCachePolicy`: Test `BaseCachePolicy` interface compliance, fallback to LRU ordering when profiling data is empty.

### Integration tests (with mocked backends)

- Verify that `cache_policy: "LRU"` produces zero profiling overhead (no threads, no imports).
- Verify that `cache_policy: "profiling"` creates the engine and orchestrator, starts background threads, and shuts down cleanly.
- Verify that eviction candidates differ from pure LRU when chunks have varying recomputation costs.
- Verify that prefetch thread promotes a hot chunk from a mock disk backend to the CPU backend.

### Performance tests

- Measure per-access overhead of `update_on_hit` with profiling enabled vs. LRU baseline.
- Measure eviction scoring latency for 1, 10, 100 candidates.
- Measure background thread CPU utilization at 100K, 1M, 10M cached chunks.
