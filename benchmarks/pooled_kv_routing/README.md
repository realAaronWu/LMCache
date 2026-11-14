# Pooled KV routing experiment components

These opt-in tools do not modify LMCache's engine, storage policy or transport.
They are infrastructure for a routing experiment, not a qualified distributed
benchmark or a claim of lower TTFT.

## Entry Points

Run from a source checkout with the repository test dependencies installed:

```bash
python -m benchmarks.pooled_kv_routing.run --help
python -m benchmarks.pooled_kv_routing.run preflight --help
python -m benchmarks.pooled_kv_routing.run generate --help
python -m benchmarks.pooled_kv_routing.run analyze --help
python -m benchmarks.pooled_kv_routing.run serve --help
python -m benchmarks.pooled_kv_routing.run smoke --help
```

`serve` is a one-worker, static-route functional probe. It does not collect
distributed telemetry or execute the cost policy. Bind it to loopback for local
testing. Completion requests need a unique `request_id`, integer-token `prompt`,
positive `max_tokens`, `stream=true`, and a single output. The journal path must
not already exist. Consult each command's help for its required paths.

`smoke` runs on the CUDA host and compares deterministic direct/proxy token
streams. Its direct endpoint must expose vLLM's development server-info receipt;
use an isolated loopback test service, not a production deployment. Receipt
collection is filtered; do not persist arbitrary server configuration or secrets.
Functional PASS does not imply that a native LMCache connector was installed.

## Integration Boundary

```text
confirmed observations --> Telemetry ------+
                                          |
frozen CostTable --> RoutingRuntime.route -> proxy -> worker completion stream
                         |                   |
                    reserve work       terminal evidence
                         ^                   |
                         +--- runtime.finish-+
```

Applications may supply `RoutingRuntime.route` and `.finish` to
`proxy.create_app`. Keep both callbacks on one event loop. The bridge calls
`choose`, looks up isolated completion work, and reserves it before dispatch.
The whole routing/reservation section contains no await, so competing ingress
cannot select against an unreserved decision on that event loop. It is not a
multi-process transaction or a cross-thread API.

TTFT prediction and reservation work are deliberately different: queue-inclusive
predicted TTFT must not be reserved again. A completion cost bucket always uses
zero pending requests, includes the decode budget, and preserves native
`local_host` versus `remote_host`. Missing completion cost rejects admission;
neither an invented zero nor an interpolation is used.

Only explicit observations can establish worker health, incarnation, prefix
contents, path or GPU residency. Sending a request establishes none of these.
Reservations do not refresh observations. Stale/unknown data follow the pure
policy's fallback rules; unconfirmed execution retains work and taints its
worker. HTTP closure is not native-transfer cancellation. Complete stream
protocol evidence confirms this experiment's worker completion, not RDMA
quiescence or buffer lifetime safety.

## Module Ownership

| Module | Responsibility | Explicitly not provided |
|---|---|---|
| `preflight` | Retained structured captures and source/safety gates | Transfer execution or hardware safety certification |
| `schemas`, `costs`, `policies` | Validated snapshots, exact measured buckets, pure Cost/LEW/Static selection | Online fitting, cost collection or transport decisions |
| `telemetry` | Bounded confirmed-prefix catalog and work ledger | Directory polling, reachability discovery or inferred APC |
| `proxy` | Bounded streaming, raw journal and lifecycle callback | Engine cancellation RPC or native resource ownership |
| `runtime` | Selection, reservation and completion-accounting bridge | Collector or calibration validity |
| `workloads` | Seeded traces from a supplied tokenized corpus | Corpus licensing or tokenizer provenance attestation |
| `analysis` | Strict interchange-packet validation and paired statistics | Raw-journal adapter, original-clock proof or full AC qualification |
| `gpu_smoke` | Real CUDA/direct/proxy functional receipts | Multi-node/P2P qualification or performance conclusions |

## Validation And Remaining Work

Focused CPU tests exercise the actual ASGI/CLI entries as well as pure policies:

```bash
python -m pytest --noconftest -q benchmarks/pooled_kv_routing/tests tests/test_gpu_smoke.py
```

`--noconftest` intentionally avoids unrelated native allocator fixtures for these
standalone tools. This is not a substitute for the standard LMCache suite or
hosted CI. The root suite requires a working native installation.

The experiment still needs a real directory/worker collector, environment-bound
cost calibration, open-loop arm orchestration with restart/warmup barriers, and
a faithful raw-journal-to-analysis adapter. The analysis packet's external gates
remain NOT RUN without independent deployment, mechanism and delivery evidence.
Neither CPU fixtures nor the single-GPU smoke complete those tasks.

Before any primary P2P efficacy run, independently qualify native transfer
timeouts, cancellation and buffer/lock release ordering on the target hardware.
The pinned P2P baseline is not presumed safe merely because a timeout is rare.
Only a separately frozen held-out paired benchmark can establish a TTFT gain;
do not select seeds or report cold-start smoke ratios as optimization results.
