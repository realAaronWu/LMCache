# T0 Preflight Contract (Schema 2)

Offline checker, not a launcher or attestation service. Requirements remain
the 2026-11-15 pooled-KV benchmark plan. Public APIs remain `preflight(config:
dict) -> dict` and `main(argv: list[str] | None = None) -> int`, with no sibling
or runtime imports. Schema 1 summaries are rejected, never silently upgraded.

## Frozen Configuration

All objects reject missing/unknown fields. No scalar coercion is performed.
Every artifact reference is exactly `{path, sha256}`: absolute local path and
64 lowercase hex SHA256. Files must resolve to nonempty bounded regular files
and match hashes. Symlinks to regular files are allowed.

Top-level fields:

- `schema_version`: integer 2, not bool.
- `run_id`: nonblank ID unique to this collection. Reuse is a collector error;
  it is not an authentication/freshness timestamp.
- `scope`: `functional_single_gpu` or `primary_two_node_rdma`.
- `source`: `{base, head, patch_sha256, artifact}`. Full lowercase 40-hex git
  revisions, 64-hex patch digest, reference to actual collected source archive.
  Base is `bcafb031401d404a5dabae4853f5e23bf9e5685c`. Collector records the
  actual running patched tree; checker does not reconstruct git history.
- `model`: `{id, revision, tokenizer_revision, files}`. Nonblank ID, full 40-hex
  revisions and 1-256 actual model/tokenizer payload references. Collector owns
  complete file discovery. Listed hashes are recomputed by streaming.
- `environment`: exactly `{python, vllm, torch, cuda, driver, nixl,
  image_digest, artifact}`. Fixed numeric-leading version strings, no aliases.
  Image is `sha256:<64 lowercase hex>` or null for bare-metal. Artifact is the
  authoritative environment capture below, not an opaque lock blob.
- `nodes`: 1-2 references to authoritative inventory captures.
- `native_evidence`: reference to a native case index, or null for NOT RUN.

## Authoritative Captures

Each capture is bounded JSON with exactly:
`{schema_version, role, identity, terminal, returncode, observation, logs}`.
Schema is integer 2. Role is environment/inventory/reference/probe as required
by the reference position. Identity matches the exact expected object.
Returncode is actual child exit, integer zero only. Terminal must be the
string `completed`, emitted by the trusted collector ONLY after child exit
and complete structured output collection. Timeout/partial/interrupted
collection must never emit it. Logs retain 1-16 diagnostic artifact references.
PASS text cannot override exit23, incomplete terminal state, missing fields,
wrong identity, outputs or counters. Hashes alone cannot qualify an opaque or
truncated JSON/log as a capture. No universal stdout parser is provided.

Collector writes one actual structured record, not a success summary alongside
a contradictory record. Every required role is checked. Captures alone supply
observations; config/index values are expected identities/requirements, never
substitutes for actual exits or output. Logs are diagnostics, not success proof.

### Environment And Inventory

Environment identity: `{run_id, source_head}`. Observation has the seven config
environment fields other than artifact, plus `binaries` (1-256 unique installed
binary/wheel references) and `argv` (1-256 nonblank launch arguments). Versions
and image match config exactly. Binary bytes are hash-checked; collector selects
the relevant installed binaries and build inputs.

Inventory identity: `{run_id, source_head}`. Observation is exactly
`{host_id, gpu_uuid, gpu_model, gpu_memory_bytes, rdma_device, rdma_gid,
link_gbps, source, model, environment}`. These are actual observed fields, not
a second summary. Source/model/environment match config. Host is stable physical
identity, not a DNS alias; UUID is the actual GPU, not a process index. Collector
retains NUMA/PCIe/NIC diagnostics in logs. GPU memory is integer [1,2^50];
device/GID are nonblank strings or null; speed is numeric (not bool) in
(0,10000] Gbps or null (thus NaN/infinity are invalid).

Functional requires exactly one inventory and permits null RDMA fields.
Primary requires two different hosts, GPU UUIDs and GIDs, same GPU model,
non-null device/GID/speed. Same NIC names across hosts are permitted.

### Native Index And Outputs

Index is exactly `{provenance_sha256, cases}`. Digest is SHA256 of UTF-8 JSON of
config's `{run_id, source, model, environment, nodes}` serialized with Python
`sort_keys=True, separators=(",", ":")` and default ensure_ascii. Cases are
exactly `{name, request_id, host_ids, reference, probe}`; last two are artifact
references. No exit/token/counter summaries are accepted. Names/request IDs
are unique; case inventory is exact, not a subset.

Reference/probe identity is exactly
`{run_id, provenance_sha256, name, request_id, host_ids}` matching index/config
and observed inventory. Reference is the no-external-cache control with the
same input/output budget; probe is the native run. Reference observation is
exactly `{token_ids, token_count}`; probe observation is exactly
`{token_ids, token_count, transport, remote_bytes, external_fetches,
loader_extra_args}`. Counts are integers [1,65536] equal to list lengths.
IDs are integers [0,2^31-1], never bool or re-tokenized text. Reference/probe IDs
must match exactly. Missing/truncated output cannot be represented by omitting
a role, shortening its declared token inventory or completing collection early.
Transfer/fetch counts are integers [0,2^63-1].

Functional requires exactly `single_gpu`, local transport, inventory host,
zero remote bytes/fetches and empty loader_extra_args. This is an output smoke,
not streaming timing, cancellation, P2P or resource-lifetime qualification.

Primary requires exactly `A_to_B`, `B_to_A`, `shared_prefix`, `partial_hit`,
`miss`, `skip_load`. Hosts follow inventory order, reversed only for B_to_A.
Transport is observed `rdma`, not merely configured UCX/NIXL (TCP is invalid).
First four require positive completed remote bytes/fetches; miss/skip require
zero. Only skip has loader_extra_args `{"lmcache.skip_load": true}`, observed
at the default loader, not client; others have `{}`. Flag receipt alone cannot
pass without zero-fetch observations, valid output and successful exits.

## Results And Boundaries

Report fields: schema_version (2), evidence_status, functional_status,
eligible_for_primary, performance_status, checks, diagnostics. Statuses are
PASS/FAIL/NOT RUN. Unattempted checks may be absent after earlier failure.
Null native evidence is NOT RUN; missing, inconsistent, failed, partial,
misassociated, malformed or hash-tampered referenced captures are FAIL.

Primary eligibility remains always false: pinned source releases timed-out
P2P buffers without transfer quiescence. Other heads lack independent resource
safety qualification (NOT RUN). Evidence completeness can PASS while resource
safety FAILs; no config flag/report overrides this gate. Performance is always
NOT RUN. Functional PASS never implies RDMA or >5% performance PASS.

CLI: `python -m benchmarks.pooled_kv_routing.preflight --config FILE --output FILE`.
Writes report; returns 0 only for functional evidence PASS. Unqualified primary,
invalid inputs and NOT RUN return 1. Argparse/output-write errors are normal
nonzero CLI errors. Parent owns overall AC0-AC6 acceptance.

Limits: JSON/config <=1 MiB; logs/source <=64 MiB each; model/environment binary
<=64 GiB each. All lists bounded above; streaming buffers <=1 MiB. No network,
shell, install or GPU actions. Trusted collector owns accurate observation,
input equivalence and complete source/model/binary discovery. Malicious forgery
of every structured record and matching artifact is outside threat model.
Hashes identify bytes, not authenticity or quiescence. CPU fixtures test only
the checker; real GPU/RDMA evidence remains separate.
