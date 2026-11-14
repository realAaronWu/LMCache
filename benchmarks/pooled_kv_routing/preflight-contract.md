# T0 Preflight Contract

This is an offline evidence checker, not a launcher or an attestation service.
The authoritative requirements remain the 2026-11-15 pooled-KV benchmark plan.
`preflight(config: dict) -> dict` and `main(argv: list[str] | None = None) -> int`
have no sibling or LMCache runtime imports. Parent `run.py` may delegate to main.

## Input

All objects reject unknown/missing fields. No scalar coercion is performed.
Every artifact reference is exactly `{ "path": absolute_local_path,
"sha256": lowercase_64_hex }`. References must resolve to nonempty regular
files with matching hashes. Raw files are retained, not interpreted as PASS
messages. Symlinks to regular files are allowed; their target bytes are hashed.

Top-level fields:

- `schema_version`: integer 1 (not true).
- `scope`: `functional_single_gpu` or `primary_two_node_rdma`.
- `source`: `{base, head, patch_sha256, artifact}`; full lowercase 40-hex git
  revisions, 64-hex patch digest, and a reference to the collected source
  snapshot/archive. Base must be `bcafb031401d404a5dabae4853f5e23bf9e5685c`.
  The collector records the actual running source, including any patch; the
  checker does not reconstruct git history from an archive.
- `model`: `{id, revision, tokenizer_revision, files}`; nonempty model ID,
  full 40-hex revisions and 1-256 references to the actual model/tokenizer
  payload files. Hashes are recomputed by streaming. The collector owns the
  complete inventory; this checker cannot infer omitted model files.
- `environment`: exactly `{python, vllm, torch, cuda, driver, nixl,
  image_digest, artifact}`. Versions are nonempty numeric-leading fixed strings,
  not floating aliases; image identity is `sha256:<64 lowercase hex>` or null for
  bare-metal execution. The required artifact reference retains the collected
  environment lock, binary/wheel hashes, build metadata and actual launch argv.
- `nodes`: 1-2 references to observed JSON inventories (schema below).
- `native_evidence`: a reference to observed native JSON, or null for NOT RUN.

Each node inventory is exactly:
`{host_id, gpu_uuid, gpu_model, gpu_memory_bytes, rdma_device, rdma_gid,
link_gbps, source, model, environment, raw}`. The provenance objects must equal
the frozen config. `raw` is 1-16 artifact references to retained device,
environment and launch captures. Host ID is a stable physical machine identity,
not a DNS alias; GPU UUID is the actual observed device, not its process index.
GPU memory is a positive integer <= 2^50 bytes. Device/GID are nonblank strings
or null; link speed is a finite positive number <= 10000 Gbps or null.
Functional scope requires one inventory and permits null RDMA fields.
Primary requires two distinct hosts, GPU UUIDs, GIDs, matching GPU model,
non-null RDMA device/GID and observed link speed. Same NIC names on different
hosts are allowed. The collector must capture NUMA/PCIe and NIC topology in raw.

Native JSON is exactly `{provenance_sha256, cases}`. The provenance digest is
SHA256 of UTF-8 JSON of config's `{source, model, environment, nodes}`, serialized
with `sort_keys=True, separators=(",", ":")` (Python default ensure_ascii).
It binds the probe to the frozen inventory without trusting a caller PASS flag.

Each case is exactly:
`{name, returncode, request_id, host_ids, transport, remote_bytes,
external_fetches, expected_token_ids, actual_token_ids, loader_extra_args, raw}`.
Return code must be integer zero; counts must be integers in [0, 2^63-1].
Expected IDs must come from the no-external-cache reference run; actual IDs
from real native output, never text re-tokenization. Both contain 1-65536
nonnegative integer token IDs <= 2^31-1 and must match exactly. Request IDs and
case names must be unique. Raw references retain reference and probe outputs,
exit/terminal records, device/backend/transfer counters and loader receipt.
Missing or truncated captures must not be summarized as completed observations.

Functional scope requires exactly `single_gpu`, local transport, the inventory
host ID, zero remote bytes/fetches, and empty loader_extra_args. This is only an
output smoke qualification, not cancellation, streaming timing or P2P safety.

Primary requires exactly `A_to_B`, `B_to_A`, `shared_prefix`, `partial_hit`,
`miss`, `skip_load`. host_ids are inventory order, reversed only for B_to_A.
Transport must be the observed `rdma`, not merely configured UCX/NIXL (which
can use TCP). The first four cases require positive completed remote bytes
and fetches. Miss and skip_load require zero bytes/fetches. Only skip_load has
`loader_extra_args={"lmcache.skip_load": true}`, captured at the default loader,
not the client; all other cases have `{}`. Skip's received flag alone cannot
pass without the zero-fetch observation, valid output, successful exit and raw.

Limits: JSON/config <= 1 MiB, raw/source artifact <= 64 MiB each, each model
file <= 64 GiB. Lists are bounded as above; reads stream in <= 1 MiB chunks.
No shell commands, downloads, installs, GPU sessions or network requests occur.

## Result And Safety Gate

Report fields: `schema_version`, `evidence_status`, `functional_status`,
`eligible_for_primary`, `performance_status`, `checks`, `diagnostics`.
Checks record configuration, inventory, native, RDMA and resource safety.
Unattempted checks may be absent after an earlier failure. Status is PASS,
FAIL or NOT RUN. Missing native evidence is NOT RUN; malformed/missing/tampered
referenced evidence is FAIL. Evidence PASS means complete consistent submitted
observations, not overall acceptance. Performance is always NOT RUN.

Primary eligibility is **always false in this implementation**: the pinned
head has a source-confirmed unsafe buffer release path. Other heads have not
been independently qualified, so their resource safety is NOT RUN, never PASS.
A later source fix and independently reviewed qualification must extend this
gate in code; a config boolean or supplied report cannot override it. Functional
scope does not exercise this P2P failure behavior and cannot imply RDMA PASS.

CLI: `python -m benchmarks.pooled_kv_routing.preflight --config FILE --output FILE`.
It writes JSON and returns 0 only for functional evidence PASS; unqualified
primary, NOT RUN, malformed config and bad evidence return 1. Argparse and output
write errors remain ordinary nonzero CLI errors. Parent owns overall AC0-AC6.

Trust boundary: collector and its observed records are trusted; missing,
inconsistent or wrong-typed evidence and self-asserted boolean certificates are
not. Malicious forgery of every record and matching file/hash is out of scope.
Hashes identify retained bytes, not authenticity, physical topology by
themselves, complete source/model inventory, or NIXL quiescence. Runtime
qualification must use real collector artifacts; tests use synthetic CPU
fixtures and do not establish hardware support or >5% performance.
