# Patent Draft: Profiling-Based KV Cache Orchestration Across a Disaggregated Memory-Storage Hierarchy

**Status:** DRAFT -- Attorney-Client Privileged / Work Product

---

## TITLE OF THE INVENTION

**System and Method for Profiling-Driven Orchestration of Key-Value Cache in Large Language Model Inference Across a Disaggregated, Multi-Tier Memory-Storage Hierarchy**

---

## I. FIELD OF THE INVENTION

The present invention relates generally to memory management in distributed computing systems, and more particularly to methods and systems for dynamically orchestrating key-value (KV) cache placement, migration, prefetching, and eviction across a multi-tier, disaggregated memory and storage hierarchy in large language model (LLM) inference environments, using runtime workload profiling to drive cache management decisions.

---

## II. BACKGROUND OF THE INVENTION

### A. The Memory Wall in LLM Inference

Transformer-based large language models (LLMs) rely on a key-value (KV) cache to accelerate autoregressive token generation. During the prefill phase, the model computes attention key and value tensors for all input tokens; during the decode phase, these cached tensors are reused to avoid redundant computation. The KV cache size grows linearly with both the sequence length and the model dimensionality. For contemporary models, 1,000 tokens of context may require approximately 4.5 GB of KV cache, and production workloads with long documents or shared corpora routinely produce KV caches exceeding 2 TB in aggregate.

Even the most advanced GPU accelerators (e.g., NVIDIA Blackwell B200 with 192 GB HBM3e, or the forthcoming Rubin architecture with approximately 288 GB HBM4) cannot hold the full KV cache for large-scale serving workloads. This fundamental mismatch between KV cache demand and GPU memory capacity -- the "memory wall" -- necessitates offloading temporarily unused KV cache to lower-cost, higher-capacity storage tiers.

### B. Multi-Tier Storage Hierarchy

Modern LLM inference deployments employ a hierarchical storage architecture:

1. **Tier 0 -- GPU HBM:** Highest bandwidth (~8 TB/s on HBM3e), lowest capacity (192--288 GB), highest cost per GB. Holds the active working set of KV cache required for ongoing token generation.

2. **Tier 1 -- CPU DRAM:** High bandwidth (~200 GB/s DDR5), moderate capacity (512 GB -- 2 TB per node), moderate cost. Serves as the primary offload target for evicted KV cache pages, using pinned (page-locked) memory for efficient DMA transfers.

3. **Tier 2 -- NVMe SSD:** Moderate bandwidth (~14 GB/s per Gen5 drive, scalable via striping), high capacity (tens of TB per node), low cost. Provides persistent storage for KV cache that cannot fit in DRAM.

4. **Tier 3 -- Remote Object Storage (S3-compatible):** Shared, persistent, virtually unlimited capacity, but higher latency. Provides long-term storage for shared context caches (e.g., enterprise document corpora).

### C. Disaggregated Prefill-Decode Architecture

Modern LLM serving increasingly adopts a disaggregated architecture in which prefill (prompt processing) and decode (token generation) phases are executed on separate GPU pools. This separation allows each pool to be independently scaled and optimized: prefill nodes are compute-bound and benefit from high parallelism, while decode nodes are memory-bound and benefit from large KV cache capacity.

In this architecture, KV cache produced by a prefill node must be transferred to a decode node before generation can begin. The efficiency of this transfer directly impacts Time To First Token (TTFT), a critical quality-of-service metric.

### D. Remote Direct Memory Access (RDMA) for KV Cache Transfer

RDMA technologies (InfiniBand, RoCEv2) enable zero-copy data transfer between nodes at wire speed (up to 400 Gb/s on NDR InfiniBand) with microsecond-scale latency, bypassing the operating system kernel and CPU involvement in the data path. RDMA is the natural transport for high-performance KV cache sharing across nodes, as it eliminates the TCP/IP overhead that would otherwise bottleneck cache transfer at scale.

---

## III. LIMITATIONS OF THE STATE OF THE ART

Despite significant advances in KV cache management for LLM inference, the following critical limitations persist in existing systems:

### Limitation 1: Static, Policy-Blind Cache Management

Existing systems (vLLM PagedAttention, SGLang RadixAttention, LMCache, Mooncake, InfiniStore) employ fixed eviction policies -- typically Least Recently Used (LRU), Least Frequently Used (LFU), or First-In-First-Out (FIFO) -- that are configured at system initialization and remain unchanged throughout operation. These policies treat all KV cache entries identically, without regard to:

- **Recomputation cost asymmetry:** Evicting a KV cache entry for a 100K-token document incurs orders of magnitude more recomputation cost (potentially minutes of GPU time) than evicting a 256-token entry, yet LRU treats both equally based solely on access recency.

- **Access pattern heterogeneity:** Production LLM workloads exhibit highly non-uniform access patterns. System prompts and shared document prefixes are accessed by every request in a session, while tail tokens are accessed only once. Static policies cannot distinguish these patterns without runtime profiling.

- **Workload phase transitions:** Serving workloads exhibit temporal patterns (e.g., business-hours spikes with high prefix reuse, overnight batch processing with sequential access). Static policies cannot adapt to these phase transitions.

### Limitation 2: No Cross-Tier Placement Intelligence

Existing multi-tier KV cache systems employ a fixed, sequential tier hierarchy: GPU -> CPU -> Disk -> Remote. Data is placed in tiers based on the order it was written, not based on any analysis of where it would be most beneficially placed. Specifically:

- **No cost-aware tier selection:** When writing a KV cache entry, the system does not consider the access latency, bandwidth, and cost characteristics of each tier relative to the expected access pattern of that entry.

- **No demotion/promotion intelligence:** When evicting from a higher tier, data is either discarded or written to the next tier regardless of whether it is likely to be accessed again. There is no mechanism to selectively promote hot entries from lower tiers or retain high-value entries in higher tiers.

- **No global view:** Each node manages its own tier hierarchy independently. There is no cluster-wide view of cache placement that would enable, for example, preferentially storing a shared document corpus on the node closest to where it will next be needed.

### Limitation 3: Reactive-Only Prefetching

Existing systems perform prefetching only in response to explicit prefix-match lookups: when a request arrives, the system searches for matching cached prefixes across tiers and loads them on demand. This approach is purely reactive and incurs the full latency of the slowest tier where data resides. There is no mechanism for:

- **Predictive prefetching:** Proactively loading KV cache entries into higher tiers before a request arrives, based on observed access patterns or scheduled workloads.

- **Hotspot-aware staging:** Identifying entries with high reuse probability and proactively staging them in GPU or CPU memory to eliminate fetch latency from the critical path.

### Limitation 4: Suboptimal Network Bandwidth Utilization

KV cache in GPU-based inference engines is managed at the page level, with page sizes typically 64 KB or smaller (e.g., 16 tokens x num_heads x head_dim x dtype_size). Transferring individual pages over RDMA or TCP networks results in poor bandwidth utilization due to:

- **Per-message overhead:** Each RDMA send/receive operation incurs fixed overhead (~1 us for signaled completions). Sending thousands of 64 KB pages individually wastes a significant fraction of available bandwidth on overhead.

- **PCIe transaction inefficiency:** Small DMA transfers do not saturate PCIe bus bandwidth. Modern NIC hardware achieves peak throughput only with transfer sizes >= 1--2 MB.

- **No adaptive grouping:** Existing systems use fixed chunk sizes regardless of network conditions, workload characteristics, or the specific access pattern of the data being transferred.

### Limitation 5: No Unified Memory-Storage Pooling

Existing systems treat each node's DRAM and NVMe storage as local resources with independent management. There is no mechanism to:

- **Present a global namespace:** Provide a single, cluster-wide view of all available DRAM and NVMe capacity across nodes, allowing any inference engine to transparently access KV cache residing on any node.

- **Decouple capacity from locality:** Allow a node with excess DRAM to serve as overflow storage for a neighboring node with high cache pressure, without requiring the application to be aware of the physical data location.

---

## IV. SUMMARY OF THE INVENTION

The present invention provides a system and method for **profiling-driven KV cache orchestration** across a disaggregated, multi-tier memory-storage hierarchy. The invention introduces a closed-loop control system in which runtime workload profiling continuously informs cache placement, migration, prefetching, and eviction decisions, replacing the static, policy-blind approaches of the prior art.

### Core Innovation: The Profiling-Orchestration Feedback Loop

The central novelty is a **runtime profiling engine** that continuously collects, analyzes, and acts on KV cache access telemetry to drive a **cache orchestrator** that makes dynamic, cost-aware decisions across the entire memory-storage hierarchy. This forms a closed feedback loop:

```
                   +------------------+
                   |  Inference Engine |
                   |  (vLLM / SGLang) |
                   +--------+---------+
                            |
                   KV cache access events
                   (get, put, hit, miss, evict)
                            |
                            v
                   +------------------+
                   | Profiling Engine  |
                   |                  |
                   | - Access freq.   |
                   | - Reuse distance |
                   | - Cost model     |
                   | - Phase detector |
                   +--------+---------+
                            |
                   Placement / eviction /
                   prefetch directives
                            |
                            v
                   +------------------+
                   | Cache Orchestrator|
                   |                  |
                   | - Tier selector  |
                   | - Migration mgr  |
                   | - Prefetch sched |
                   | - Eviction policy|
                   +--------+---------+
                            |
            +-------+-------+-------+-------+
            |       |       |       |       |
            v       v       v       v       v
          GPU     CPU     NVMe   Remote   Remote
          HBM    DRAM     SSD    DRAM(*)  S3/Obj
         Tier 0  Tier 1  Tier 2  Tier 1'  Tier 3
                        (*) via RDMA mesh
```

### Key Inventive Aspects

**Aspect 1: Multi-Dimensional Access Profiling**

The profiling engine maintains per-chunk telemetry that goes beyond simple access counts:

- **Access frequency histogram:** Per-chunk access count over configurable time windows (short-term for burst detection, long-term for steady-state characterization).

- **Reuse distance distribution:** The number of distinct intervening chunk accesses between consecutive accesses to the same chunk. Short reuse distances indicate hot data; long reuse distances indicate cold data that can be safely demoted.

- **Recomputation cost annotation:** Each KV cache chunk is annotated with the estimated GPU time required to recompute it from scratch. This cost is proportional to the number of tokens in the chunk and the model's computational profile, and is computed once at prefill time. This annotation enables cost-aware eviction: the orchestrator avoids evicting high-recomputation-cost entries when lower-cost alternatives exist.

- **Temporal phase detection:** The profiling engine identifies workload phase transitions (e.g., shift from interactive chat with high prefix reuse to batch document processing with sequential access) using change-point detection on the access frequency time series. Phase transitions trigger re-evaluation of the active orchestration strategy.

- **Cross-request correlation:** The profiling engine tracks which KV cache entries are co-accessed across requests, building a co-occurrence graph. This enables group prefetching: when one entry in a frequently co-accessed group is requested, the entire group is prefetched.

**Aspect 2: Cost-Aware Tier Placement**

The cache orchestrator uses a cost model to determine the optimal tier for each KV cache chunk. The cost model considers:

- **Access probability:** Estimated from the profiling engine's reuse distance distribution.
- **Access latency per tier:** Measured at runtime (GPU HBM: ~100 ns, CPU DRAM: ~300 ns, NVMe: ~10 us, Remote DRAM via RDMA: ~2 us, S3: ~10 ms).
- **Recomputation cost:** The GPU time to regenerate the entry if evicted entirely.
- **Storage cost:** The opportunity cost of occupying capacity in each tier (e.g., a byte in GPU HBM is worth more than a byte on NVMe because it displaces active model weights or KV cache).

The placement decision minimizes the expected total cost:

```
E[cost] = P(access) * latency(tier) + (1 - P(access)) * 0
        + P(eviction_needed) * recomputation_cost
        + storage_cost(tier) * size * holding_time
```

This cost model naturally produces the intuitive behavior: frequently accessed, expensive-to-recompute chunks are placed in the fastest tier, while infrequently accessed, cheap-to-recompute chunks are demoted to slower, cheaper tiers.

**Aspect 3: Predictive Prefetching with Hotspot Detection**

The profiling engine identifies **hotspot chunks** -- entries whose access frequency exceeds a configurable threshold within a sliding time window. For these chunks, the orchestrator issues proactive prefetch directives:

- **Tier promotion:** Hot chunks residing in NVMe or remote storage are preemptively promoted to CPU DRAM.
- **GPU staging:** Chunks identified as likely to be needed within the next N inference requests (estimated from the co-occurrence graph and recent request patterns) are preemptively transferred to GPU HBM.
- **Cross-node prefetching:** In a disaggregated prefill-decode architecture, when the prefill node produces KV cache for a request, the profiling engine consults its access model to determine which decode node is most likely to need the data, and initiates RDMA transfer to that node's DRAM before the decode request arrives.

**Aspect 4: Adaptive Eviction with Recomputation Cost Awareness**

When a tier reaches its capacity threshold and eviction is necessary, the orchestrator selects eviction candidates using a composite scoring function:

```
eviction_score(chunk) = w1 * recency_score(chunk)
                      + w2 * frequency_score(chunk)
                      + w3 * recomputation_cost_score(chunk)
                      + w4 * size_score(chunk)
```

Where:
- `recency_score`: Inverse of time since last access (LRU component).
- `frequency_score`: Inverse of access frequency (LFU component).
- `recomputation_cost_score`: Inverse of recomputation cost (prefer evicting cheap-to-recompute entries).
- `size_score`: Larger entries free more capacity per eviction (favor evicting large, cold, cheap entries).
- `w1..w4`: Weights dynamically adjusted by the phase detector based on the current workload characteristics.

Critically, eviction is **tier-aware**: before discarding a chunk entirely, the orchestrator considers **demotion** to a lower tier. A chunk evicted from CPU DRAM may be written to NVMe rather than discarded, if its recomputation cost exceeds the NVMe storage cost. This demotion decision is also driven by the cost model.

---

## V. DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS

### A. System Architecture

The preferred embodiment comprises a cluster of N GPU nodes, each containing:
- One or more GPUs with HBM (Tier 0)
- CPU DRAM, page-locked and RDMA-registered (Tier 1)
- One or more NVMe SSDs (Tier 2)
- A network interface card (NIC) capable of RDMA (InfiniBand or RoCEv2)

The nodes are interconnected via a high-speed RDMA fabric. An S3-compatible object storage service (Tier 3) is accessible from all nodes.

### B. Disaggregated Memory-Storage Pool (Implementation Approach)

#### B.1 Cross-Node DRAM Pooling via RDMA

The CPU DRAM on each node is registered with the RDMA subsystem as a single contiguous memory region at system initialization. This registration enables any node in the cluster to read from or write to any other node's DRAM via one-sided RDMA operations (RDMA_READ, RDMA_WRITE), without involving the remote node's CPU.

A **global memory directory service** maintains a mapping from KV cache chunk identifiers to their physical locations (node ID, memory address, size). When an inference engine on Node A requires a KV cache chunk that resides in Node B's DRAM, the directory service provides the RDMA address, and Node A performs a direct RDMA_READ to retrieve the data. This forms a **memory mesh** in which the effective DRAM capacity available to any single node is the aggregate DRAM of the entire cluster.

#### B.2 Cross-Node NVMe Pooling

NVMe SSDs across nodes are organized into a distributed storage pool. Each node exposes its local NVMe capacity to the cluster via an RDMA-accessible storage service. KV cache chunks that cannot fit in the aggregate DRAM pool are partitioned and distributed across the NVMe drives of multiple nodes, with an index maintained by the global directory service.

For KV caches exceeding 1 TB (e.g., a shared document corpus), the system partitions the cache into chunks and stripes them across NVMe drives on different nodes. This provides both aggregate capacity and aggregate bandwidth: loading a 2 TB cache from 16 nodes' NVMe drives simultaneously achieves 16x the bandwidth of a single node.

Data movement between NVMe and DRAM is performed via RDMA combined with Storage Performance Development Kit (SPDK) for kernel-bypass NVMe access, minimizing CPU involvement and latency.

#### B.3 S3-Compatible Object Storage with RDMA Data Plane

For long-term persistent storage and sharing of very large KV caches, the system employs S3-compatible object storage with an RDMA-accelerated data plane. Using NVIDIA's cuObject library, the system registers pinned CPU memory with the RDMA subsystem and generates per-request RDMA tokens. The S3 server performs direct RDMA_READ (for PUT) or RDMA_WRITE (for GET) operations against the registered memory, bypassing the HTTP body entirely. This eliminates one data copy per transfer compared to the traditional TCP-based S3 path.

### C. Adaptive Chunk Coalescing for Bandwidth Optimization (Implementation Approach)

#### C.1 The Small-Page Problem

GPU inference engines (vLLM, SGLang) manage KV cache at the page level using PagedAttention, with typical page sizes of 16--64 tokens. At 2 bytes per element with 32 attention heads and 128 head dimensions, a 16-token page is only 16 * 32 * 128 * 2 * 2 (K+V) = 512 KB. While adequate for GPU memory management, this granularity is too fine for efficient network transfer.

#### C.2 Chunk Coalescing Strategy

The system groups contiguous KV cache pages into larger **transfer chunks** of configurable size (default 2 MB, tunable from 512 KB to 16 MB). The coalescing process:

1. **Contiguity detection:** Identifies sequences of KV cache pages that are logically contiguous (consecutive token positions within the same request/prefix).
2. **Chunk formation:** Groups contiguous pages into chunks up to the target size.
3. **Single RDMA operation:** Each chunk is transferred as a single RDMA operation, amortizing the per-message overhead across many pages.

This coalescing is adaptive: the system monitors achieved network throughput and adjusts the chunk size to maximize bandwidth utilization. If the NIC is underutilized (indicating chunks are too small), the target size is increased. If DRAM fragmentation limits the formation of large contiguous chunks, the target size is decreased.

### D. Profiling Engine Implementation

#### D.1 Lightweight Instrumentation

The profiling engine instruments the KV cache access path with minimal overhead:

- **Inline counters:** Each chunk's metadata includes a 64-bit access counter and a 64-bit last-access timestamp, updated atomically on every access. This adds < 1 ns overhead per access.
- **Sampling-based reuse distance:** Rather than tracking reuse distance for every access (which would require O(N) space for N chunks), the system samples 1 in every K accesses (configurable, default K=100) and computes reuse distance for the sampled accesses using a compact hash-based sketch data structure.
- **Background aggregation:** A dedicated background thread aggregates per-chunk counters into the access frequency histogram and reuse distance distribution at configurable intervals (default: every 1 second). This avoids impacting the latency-critical inference path.

#### D.2 Phase Detection

The phase detector monitors the aggregate access frequency distribution over time using an online change-point detection algorithm. When a statistically significant shift in the distribution is detected (e.g., the fraction of accesses to the top-10% most-accessed chunks changes by more than a configurable threshold), the phase detector signals the orchestrator to re-evaluate its strategy.

Phase transitions trigger:
- Re-computation of the cost model's weight parameters (w1..w4 in the eviction score).
- Re-evaluation of the chunk coalescing target size.
- Re-assessment of prefetch aggressiveness (more aggressive during high-reuse phases, less during sequential-access phases).

#### D.3 Recomputation Cost Estimation

At prefill time, when a KV cache chunk is first produced, the system records:
- The number of tokens in the chunk.
- The model identity and tensor parallelism degree.
- The measured prefill latency per token (calibrated at system startup).

From these, the recomputation cost is estimated as:

```
recomputation_cost(chunk) = num_tokens * per_token_prefill_latency
```

This estimate is stored in the chunk's metadata and used by the eviction scoring function. For very large shared caches (e.g., a 2 TB document corpus), the aggregate recomputation cost may be on the order of minutes of GPU time, making eviction extremely expensive and motivating demotion to NVMe rather than discard.

### E. Cache Orchestrator Implementation

#### E.1 Placement Decision Pipeline

When a new KV cache chunk is produced (e.g., after prefill), the orchestrator determines its initial placement:

1. **Classify chunk:** Use the profiling engine's access model to classify the chunk as hot (high predicted reuse), warm (moderate reuse), or cold (low reuse). For the first occurrence of a new chunk, classification is based on the chunk's properties (token count, prefix depth, association with known hot prefixes).

2. **Select tier:** Hot chunks -> CPU DRAM (Tier 1), with proactive staging to GPU HBM (Tier 0) if predicted to be needed within the next few requests. Warm chunks -> CPU DRAM with eligibility for NVMe demotion under memory pressure. Cold chunks -> NVMe (Tier 2) or S3 (Tier 3) directly, bypassing DRAM to avoid polluting the cache.

3. **Select node:** In a multi-node deployment, select the node whose inference engine is most likely to need the chunk (based on the request routing history and the co-occurrence model). Default: the local node.

#### E.2 Migration Controller

A background migration controller continuously scans chunk metadata and issues migration operations when the current placement no longer matches the optimal placement:

- **Promotion:** A chunk in NVMe with increasing access frequency is promoted to DRAM.
- **Demotion:** A chunk in DRAM with decreasing access frequency is demoted to NVMe.
- **Cross-node migration:** A chunk in Node A's DRAM that is increasingly accessed by Node B's inference engine is migrated to Node B's DRAM (or replicated, if both nodes access it).

Migration is rate-limited to avoid bandwidth contention with active inference traffic. The migration budget (bytes/second) is dynamically adjusted based on observed network utilization.

#### E.3 Prefetch Scheduler

The prefetch scheduler maintains a priority queue of chunks predicted to be needed soon, ranked by:

```
prefetch_priority(chunk) = P(access within T) / fetch_latency(current_tier)
```

Where T is a configurable lookahead horizon (default: 100 inference requests). Chunks with high access probability and high fetch latency (i.e., residing in slow tiers) are prefetched first. The scheduler issues prefetch operations as background RDMA transfers, using available bandwidth without blocking active requests.

---

## VI. ADVANTAGES OVER THE PRIOR ART

### Advantage 1: Reduced GPU Recomputation Waste

By incorporating recomputation cost into eviction decisions, the system avoids the pathological behavior of LRU-based systems that evict expensive-to-recompute entries (e.g., 100K-token document caches) in favor of retaining cheap-to-recompute entries (e.g., short system prompts). In production workloads with mixed context lengths, this can reduce GPU recomputation by 40--60% compared to pure LRU, directly translating to higher serving throughput.

### Advantage 2: Improved Cache Hit Rate via Predictive Prefetching

By proactively staging hot KV cache entries in fast tiers before they are needed, the system reduces cache miss latency from the perspective of the inference engine. For workloads with identifiable access patterns (e.g., rotating system prompts, frequently queried documents), predictive prefetching can eliminate 80--90% of cold-tier fetches from the critical path.

### Advantage 3: Optimal Utilization of the Memory-Storage Hierarchy

The cost-aware placement model ensures that each tier is used for the data that best matches its performance and cost characteristics. GPU HBM holds only the most critical, imminently-needed entries. CPU DRAM serves as a high-bandwidth cache for hot and warm entries. NVMe provides cost-effective persistent storage for cold entries that are expensive to recompute. S3 provides unlimited capacity for shared context. This produces significantly higher effective cache capacity compared to systems that use fixed, sequential tier placement.

### Advantage 4: Adaptive Response to Workload Changes

The phase detection mechanism enables the system to automatically adjust its caching strategy when workload characteristics change, without human intervention. This is critical for production environments where workloads vary significantly between business hours (interactive, high reuse) and off-hours (batch processing, sequential access).

### Advantage 5: Maximized Network Bandwidth Utilization

Adaptive chunk coalescing ensures that RDMA transfers saturate available network bandwidth regardless of the underlying KV page size. By monitoring achieved throughput and dynamically adjusting chunk sizes, the system achieves near-peak bandwidth utilization (> 90% of theoretical NIC line rate) compared to the 30--50% typical of per-page transfers.

### Advantage 6: Cluster-Wide Cache Efficiency

The global DRAM and NVMe pooling, combined with cross-node migration, transforms the cluster's memory resources from isolated per-node caches into a unified, globally-optimized cache. This enables workloads that exceed a single node's capacity to benefit from the aggregate capacity of the cluster, while the orchestrator ensures data is physically located where it is most likely to be needed.

---

## VII. CLAIMS (OUTLINE)

### Independent Claims

**Claim 1 (System):** A system for managing key-value cache in large language model inference, comprising:
- a multi-tier storage hierarchy comprising GPU memory, CPU memory, non-volatile storage, and remote storage;
- a profiling engine that continuously monitors per-chunk access telemetry including access frequency, reuse distance, and recomputation cost;
- a cache orchestrator that uses output of the profiling engine to dynamically determine placement, migration, prefetching, and eviction of KV cache chunks across the multi-tier hierarchy based on a cost model that considers access probability, tier latency, recomputation cost, and storage cost.

**Claim 2 (Method):** A method for orchestrating key-value cache across a disaggregated memory hierarchy, comprising:
- instrumenting KV cache access paths to collect per-chunk access telemetry;
- computing a composite eviction score incorporating access recency, access frequency, recomputation cost, and entry size, with dynamically adjusted weights;
- selecting eviction candidates and determining whether to demote to a lower tier or discard based on the cost model;
- proactively prefetching chunks predicted to be accessed within a configurable lookahead horizon.

**Claim 3 (Method):** A method for predictive prefetching of KV cache in LLM inference, comprising:
- maintaining a co-occurrence graph of KV cache chunks based on cross-request access correlation;
- identifying hotspot chunks whose access frequency exceeds a threshold within a sliding time window;
- issuing proactive prefetch directives to stage hotspot chunks in a faster tier before they are requested.

### Dependent Claims

**Claim 4:** The system of Claim 1, wherein the profiling engine performs temporal phase detection using change-point detection on the access frequency time series, and the orchestrator adjusts its strategy in response to detected phase transitions.

**Claim 5:** The system of Claim 1, wherein the multi-tier hierarchy includes a disaggregated memory pool formed by RDMA-registering CPU DRAM across multiple nodes, providing a global memory namespace accessible to any node via one-sided RDMA operations.

**Claim 6:** The system of Claim 5, further comprising a cross-node NVMe pool formed by exposing local NVMe capacity via RDMA-accessible storage services, with KV cache chunks striped across NVMe drives on different nodes.

**Claim 7:** The method of Claim 2, wherein recomputation cost for each chunk is estimated at prefill time as the product of the chunk's token count and a calibrated per-token prefill latency.

**Claim 8:** The method of Claim 2, further comprising adaptive chunk coalescing that groups contiguous KV cache pages into transfer chunks whose size is dynamically adjusted based on monitored network throughput to maximize RDMA bandwidth utilization.

**Claim 9:** The system of Claim 1, wherein remote storage access employs an RDMA-accelerated data plane for S3-compatible object storage, generating per-request RDMA tokens from a pool-registered memory region to enable server-initiated RDMA_READ or RDMA_WRITE operations that bypass the HTTP body.

**Claim 10:** The method of Claim 3, wherein cross-node prefetching in a disaggregated prefill-decode architecture initiates RDMA transfer of KV cache from a prefill node to a predicted decode node before the decode request arrives, based on request routing history.

---

## VIII. ABSTRACT

A system and method for profiling-driven orchestration of key-value (KV) cache in large language model (LLM) inference across a multi-tier, disaggregated memory-storage hierarchy. A profiling engine continuously collects per-chunk access telemetry -- including access frequency, reuse distance, recomputation cost, and cross-request co-occurrence -- and feeds this data into a cache orchestrator. The orchestrator dynamically determines KV cache chunk placement, tier migration, predictive prefetching, and cost-aware eviction using a composite cost model that jointly considers access probability, tier latency, recomputation cost, and storage cost. The system pools CPU DRAM and NVMe storage across nodes via RDMA into a globally-addressable namespace, employs adaptive chunk coalescing to maximize network bandwidth utilization, and supports RDMA-accelerated S3 object storage for persistent cache tiers. Phase detection enables automatic strategy adaptation when workload characteristics change. The result is a closed-loop control system that continuously optimizes KV cache placement across the entire memory-storage hierarchy, reducing GPU recomputation waste, improving cache hit rates, and maximizing cluster-wide resource utilization.

---

*This document is a preliminary draft for internal review. All claims are subject to revision following prior art search and freedom-to-operate analysis.*
