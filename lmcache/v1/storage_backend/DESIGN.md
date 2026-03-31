# cuObject S3-over-RDMA Connector — Design

## Overview

The cuObject S3 connector adds RDMA-accelerated data transfer to LMCache's
existing CRT-based S3 storage path. When a cuObject-enabled S3 endpoint is
available, the connector bypasses the HTTP body for PUT and GET operations:
the server reads from (or writes to) the client's pinned memory directly over
RDMA, eliminating TCP/IP stack overhead and kernel copies.

The design goal is **minimal, surgical integration**: override only the
data-plane methods (`_s3_upload` / `_s3_download`) while inheriting the full
control-plane infrastructure (auth, signing, TLS, circuit breaker, priority
queue, object size caching, batched operations) from the existing
`S3Connector`.

--- 

## Background: The cuObject RDMA Protocol

NVIDIA's cuObject library (`libcuobjclient.so`, shipping with CUDA Toolkit
>= 13.1.1) provides two usage patterns for S3-over-RDMA, both documented
in the cuObjClient API Specification v1.0.0:

**Callback-based I/O** (spec sections 1.2, 1.6, 1.10): The library
orchestrates the full I/O lifecycle.  `cuObjPut()` / `cuObjGet()` invoke
user-supplied callbacks that receive RDMA descriptors and are expected to
perform the entire server communication synchronously within the callback.

**Manual RDMA Token Management** (spec section 1.12.4): The caller
generates tokens directly via `cuMemObjGetRDMAToken()` and uses them in
its own HTTP request flow.  Token lifetime is caller-managed (freed via
`cuMemObjPutRDMAToken()`).

LMCache uses the **Manual RDMA Token Management** pattern (see
[design decision #3](#3-cumemobjgetrdmatoken-for-direct-token-generation)
for rationale).  The end-to-end flow is:

1. The client registers a contiguous pinned memory region with cuObject
   via `cuMemObjGetDescriptor()`.
2. For each PUT/GET, the client calls `cuMemObjGetRDMAToken()`, which
   generates an **RDMA token** — an opaque string encoding the memory
   address, RDMA keys, and connection info for a sub-region of the
   registered pool.
3. The client sends a standard S3 HTTP request but includes the token as
   an `x-amz-rdma-token` header.
4. The cuObject-enabled S3 server reads the token and performs
   `RDMA_READ` (PUT) or `RDMA_WRITE` (GET) directly to/from the client's
   pinned memory. No data travels through the HTTP body.
5. The server responds with an `x-amz-rdma-reply` header indicating
   completion status.
6. The client frees the token string via `cuMemObjPutRDMAToken()`.

This protocol is transparent to the S3 control plane: the request is still a
valid `PutObject` / `GetObject`; only the data plane changes.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  CuObjectS3ConnectorAdapter                      │  URL routing
│  Matches "cuobj+s3://" URLs                      │
│  Strips prefix, creates CuObjectS3Connector      │
└───────────────────────┬──────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────┐
│  CuObjectS3Connector (extends S3Connector)       │  Data-plane override
│  Overrides _s3_upload / _s3_download only        │
│  Injects x-amz-rdma-token into HTTP headers      │
│  Falls back to parent HTTP body on any failure    │
└───────────────────────┬──────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────┐
│  CuObjClientWrapper (cuobject_bindings.py)       │  Python wrapper
│  Pool registration, RDMA token generation,        │
│  RDMA reply parsing (JSON / numeric / keyword)    │
└───────────────────────┬──────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────┐
│  CuObjectClient (C++ pybind11 extension)         │  Build-time linked
│  Links against libcuobjclient.so at build time    │
│  std::unique_ptr<cuObjClient> instance            │
│  cuMemObjGetRDMAToken / cuMemObjPutRDMAToken      │
│  Pool tracking (base + size for offset calc)      │
│  All methods release GIL                          │
└──────────────────────────────────────────────────┘
```

### Module Layout

```
csrc/storage_backends/cuobject/
├── cuobject_api.h            # #include <cuobjclient.h> + namespace constants
├── cuobject_client.h         # CuObjectClient C++ class declaration
├── cuobject_client.cpp       # Implementation: token generation, pool tracking
└── pybind.cpp                # pybind11 module: lmcache.lmcache_cuobject

lmcache/v1/storage_backend/connector/
├── cuobject_bindings.py      # CuObjClientWrapper: pool mgmt, reply parsing
├── cuobject_s3_connector.py  # CuObjectS3Connector: data-plane override with RDMA headers
└── cuobject_s3_adapter.py    # URL-based factory: "cuobj+s3://" -> CuObjectS3Connector
```

---

## Build System

The cuObject C++ extension (`lmcache.lmcache_cuobject`) is **conditionally
built** by `setup.py`. At build time, `_find_cuobject_paths()` searches for
the cuObjClient SDK (`cuobjclient.h` and `libcuobjclient.so`):

1. Check environment variables `CUOBJECT_INCLUDE_DIR` / `CUOBJECT_LIB_DIR`.
2. Search standard CUDA Toolkit paths (e.g. `/usr/local/cuda/include`).

If the SDK is found, the extension is compiled and linked against
`-lcuobjclient`. If not, the extension is silently skipped — LMCache
builds and runs normally without cuObject support.

```
Build-time SDK found?
  ├─ Yes → compile lmcache_cuobject, link -lcuobjclient
  └─ No  → skip extension, log warning
```

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `CUOBJECT_INCLUDE_DIR` | Path to directory containing `cuobjclient.h` |
| `CUOBJECT_LIB_DIR` | Path to directory containing `libcuobjclient.so` |
| `NO_CUOBJECT` | Set to `1` to skip the extension even if SDK is present |

---

## End-to-End Workflow

### Initialisation

```
  CuObjectS3Connector            CuObjectClient (C++)           libcuobjclient
  ────────────────────            ────────────────────           ──────────────
         │                                │                            │
         │  CuObjectClient(proto)         │                            │
         │──────────────────────────────▶ │                            │
         │                                │  make_unique<cuObjClient>  │
         │                                │  (ops={}, proto)           │
         │                                │──────────────────────────▶ │
         │                                │          cuObjClient*      │
         │                                │◀──────────────────────────│
         │            client              │                            │
         │◀──────────────────────────────│                            │
         │                                │                            │
         │  register_pool(base, size)     │                            │
         │──────────────────────────────▶ │                            │
         │                                │  cuMemObjGetDescriptor     │
         │                                │  (base, size)              │
         │                                │──────────────────────────▶ │
         │                                │   Registers memory for     │
         │                                │   RDMA, sets up keys       │
         │                                │◀──────────────────────────│
         │                                │  store pool_base_, size_   │
         │          (base, size)          │                            │
         │◀──────────────────────────────│                            │
```

### Upload (PUT)

```
  CuObjectS3Connector     CuObjectClient (C++)     libcuobjclient         CRT           S3 Server
  ────────────────────     ────────────────────     ──────────────     ───────────     ────────────
         │                         │                       │               │                │
    ┌────┴────┐                    │                       │               │                │
    │_rdma_   │                    │                       │               │                │
    │upload() │                    │                       │               │                │
    └────┬────┘                    │                       │               │                │
         │                         │                       │               │                │
   1.    │  prepare_put(ptr, size) │                       │               │                │
         │────────────────────────▶│                       │               │                │
         │                         │  offset = ptr - base  │               │                │
         │                         │                       │               │                │
   2.    │                         │  cuMemObjGetRDMAToken │               │                │
         │                         │  (base, size, offset, │               │                │
         │                         │   CUOBJ_PUT, &desc)   │               │                │
         │                         │──────────────────────▶│               │                │
         │                         │     desc_str (alloc)  │               │                │
         │                         │◀──────────────────────│               │                │
         │                         │                       │               │                │
   3.    │                         │  token = copy(desc)   │               │                │
         │                         │  cuMemObjPutRDMAToken │               │                │
         │                         │  (desc)  [free alloc] │               │                │
         │                         │──────────────────────▶│               │                │
         │                         │◀──────────────────────│               │                │
         │           token         │                       │               │                │
         │◀────────────────────────│                       │               │                │
         │                         │                       │               │                │
   4.    │  PUT /<key>             │                       │               │                │
         │  x-amz-rdma-token: <token>                     │               │                │
         │  Content-Length: <size>  │                       │               │                │
         │  (no body)              │                       │               │                │
         │─────────────────────────────────────────────────────────────── ▶│                │
         │                         │                       │    SigV4 sign │                │
         │                         │                       │    TLS send   │                │
         │                         │                       │               │───────────────▶│
         │                         │                       │               │                │
   5.    │                         │                       │               │      RDMA_READ │
         │  ◀ ═══════════════════════════ RDMA ══════════════════════════════════════════  │
         │     Server reads data directly from client's pinned memory      │                │
         │     (no data in HTTP body)                      │               │                │
         │                         │                       │               │                │
   6.    │                         │                       │               │  HTTP 200      │
         │                         │                       │               │  x-amz-rdma-   │
         │                         │                       │               │  reply: ok     │
         │                         │                       │               │◀───────────────│
         │              on_headers(200, reply)              │               │                │
         │◀─────────────────────────────────────────────────────────────── │                │
         │                         │                       │               │                │
   7.    │  parse_rdma_reply("ok") │                       │               │                │
         │  → True ✓               │                       │               │                │
```

### Download (GET)

```
  CuObjectS3Connector     CuObjectClient (C++)     libcuobjclient         CRT           S3 Server
  ────────────────────     ────────────────────     ──────────────     ───────────     ────────────
         │                         │                       │               │                │
    ┌────┴────┐                    │                       │               │                │
    │_rdma_   │                    │                       │               │                │
    │download │                    │                       │               │                │
    └────┬────┘                    │                       │               │                │
         │                         │                       │               │                │
   1.    │  prepare_get(ptr, size) │                       │               │                │
         │────────────────────────▶│                       │               │                │
         │                         │  offset = ptr - base  │               │                │
         │                         │                       │               │                │
   2.    │                         │  cuMemObjGetRDMAToken │               │                │
         │                         │  (base, size, offset, │               │                │
         │                         │   CUOBJ_GET, &desc)   │               │                │
         │                         │──────────────────────▶│               │                │
         │                         │     desc_str (alloc)  │               │                │
         │                         │◀──────────────────────│               │                │
         │                         │                       │               │                │
   3.    │                         │  token = copy(desc)   │               │                │
         │                         │  cuMemObjPutRDMAToken │               │                │
         │                         │  (desc)  [free alloc] │               │                │
         │                         │──────────────────────▶│               │                │
         │                         │◀──────────────────────│               │                │
         │           token         │                       │               │                │
         │◀────────────────────────│                       │               │                │
         │                         │                       │               │                │
   4.    │  GET /<key>             │                       │               │                │
         │  x-amz-rdma-token: <token>                     │               │                │
         │  (no body, no on_body callback)                 │               │                │
         │─────────────────────────────────────────────────────────────── ▶│                │
         │                         │                       │    SigV4 sign │                │
         │                         │                       │    TLS send   │                │
         │                         │                       │               │───────────────▶│
         │                         │                       │               │                │
   5.    │                         │                       │               │     RDMA_WRITE │
         │  ◀ ═══════════════════════════ RDMA ══════════════════════════════════════════  │
         │     Server writes data directly into client's pinned memory     │                │
         │     Data is now in the MemoryObj buffer                         │                │
         │                         │                       │               │                │
   6.    │                         │                       │               │  HTTP 200      │
         │                         │                       │               │  x-amz-rdma-   │
         │                         │                       │               │  reply: ok     │
         │                         │                       │               │◀───────────────│
         │              on_headers(200, reply)              │               │                │
         │◀─────────────────────────────────────────────────────────────── │                │
         │                         │                       │               │                │
   7.    │  parse_rdma_reply("ok") │                       │               │                │
         │  → True ✓               │                       │               │                │
```

### Cleanup

```
  CuObjectS3Connector            CuObjectClient (C++)           libcuobjclient
  ────────────────────            ────────────────────           ──────────────
         │                                │                            │
         │  deregister_pool(base)         │                            │
         │──────────────────────────────▶ │                            │
         │                                │  cuMemObjPutDescriptor     │
         │                                │  (base)                    │
         │                                │──────────────────────────▶ │
         │                                │   Releases RDMA keys       │
         │                                │◀──────────────────────────│
         │◀──────────────────────────────│                            │
         │                                │                            │
         │  close()                       │                            │
         │──────────────────────────────▶ │                            │
         │                                │  client_.reset()           │
         │                                │  → ~cuObjClient()          │
         │                                │──────────────────────────▶ │
         │                                │◀──────────────────────────│
         │◀──────────────────────────────│                            │
```

### Fallback

Both `_s3_upload` and `_s3_download` use a two-level fallback:

1. **Init-time**: If the C++ extension is not available or cuObject client
   initialisation fails (RDMA hardware unavailable, library not linked),
   `_rdma_enabled` is set to `False` and the connector operates as a plain
   `S3Connector` for its entire lifetime.
2. **Per-request**: If `prepare_put` / `prepare_get` or RDMA reply
   verification fails, the method catches the exception and delegates to
   `super()._s3_upload()` / `super()._s3_download()` (standard HTTP body).

```python
def _s3_upload(self, key_str, memory_obj):
    if not self._rdma_enabled:
        return super()._s3_upload(key_str, memory_obj)
    try:
        return self._rdma_upload(key_str, memory_obj)
    except Exception:
        return super()._s3_upload(key_str, memory_obj)
```

---

## Key Design Decisions

### 1. Inheritance over composition

`CuObjectS3Connector` extends `S3Connector` and overrides only `_s3_upload`
and `_s3_download`. Everything else — CRT client setup, AWS credential
provider chain, TLS/ALPN negotiation, S3Express signing, circuit breaker,
object size cache, `AsyncPQExecutor` with prioritised PEEK/PREFETCH/GET/PUT
queues, `get()`, `put()`, `batched_get()`, `exists()`, and `close()` — is
inherited unchanged.

**Rationale**: RDMA only changes the data plane; the control plane stays the
same. Overriding two methods is the minimal surface area.

### 2. Pool-level RDMA registration

The entire pinned CPU memory pool backing `local_cpu_backend.memory_allocator`
is registered with cuObject once at init time:

```python
base_ptr, size_bytes = _get_allocator_buffer_info(allocator)
self._rdma_pool_handle = self._cuobj_client.register_pool(base_ptr, size_bytes)
```

Individual `MemoryObj` buffers are sub-regions of this pool.  The per-request
`prepare_put` / `prepare_get` calls generate a token for just that sub-region
by computing `buffer_offset = data_ptr - pool_base`.

**Rationale**: LMCache's memory allocators (`MixedMemoryAllocator`,
`PinMemoryAllocator`, `LazyMemoryAllocator`) all use a single contiguous
pinned buffer. Registering once amortises the RDMA registration cost over
all requests.

### 3. cuMemObjGetRDMAToken for direct token generation

RDMA tokens are generated via `cuMemObjGetRDMAToken()` — a direct API that
returns the token string without callbacks or I/O operations:

```cpp
cuObjErr_t rc = client_->cuMemObjGetRDMAToken(
    pool_base_ptr, size, buffer_offset, CUOBJ_PUT, &desc_str);
std::string token(desc_str);
client_->cuMemObjPutRDMAToken(desc_str);  // free library allocation
```

**Rationale**: The cuObjClient API Specification v1.0.0 documents two
distinct usage patterns:

**Pattern A — Callback-based I/O** (spec sections 1.2, 1.6, 1.10,
1.12.1–1.12.2): `cuObjPut()` / `cuObjGet()` are synchronous I/O operations
that invoke user-supplied `CUObjOps_t` callbacks.  The callback receives a
`cufileRDMAInfo_t*` containing the RDMA descriptor and is expected to perform
the **entire server communication** synchronously — send the token, wait for
the RDMA transfer, and return bytes transferred.  `cuObjPut()` blocks until
the callback returns.

**Pattern B — Manual RDMA Token Management** (spec section 1.12.4):
`cuMemObjGetRDMAToken()` generates an RDMA token string with caller-managed
lifetime.  The caller uses the token in its own request flow and frees it
via `cuMemObjPutRDMAToken()` when done.

We use **Pattern B** for three reasons:

1. **Descriptor lifetime constraint** (spec section 1.15, Best Practices):
   "RDMA descriptors are only valid during the callback invocation."  In
   Pattern A, the `rdma_info` descriptor is valid only while the callback
   is executing.  Since CRT sends S3 requests asynchronously via an event
   loop, the token would need to outlive the callback — violating this
   constraint.  Pattern B's `cuMemObjGetRDMAToken()` returns a token whose
   lifetime we control explicitly (free with `cuMemObjPutRDMAToken()`).

2. **CRT async model incompatibility**: Pattern A requires the callback to
   do the complete server round-trip synchronously.  CRT's S3 request
   lifecycle is asynchronous (signing, TLS negotiation, HTTP/2
   multiplexing all happen on CRT's I/O event loop).  Blocking inside a
   C callback waiting for CRT completion would deadlock CRT's threads or
   defeat its async design.

3. **Separation of concerns**: LMCache already has a full-featured S3
   control plane (CRT client, AWS credential chain, SigV4 signing,
   circuit breaker, priority-queue executor).  We only need the RDMA
   token — not the I/O orchestration that Pattern A provides.  Pattern B
   gives us exactly the token and nothing more.

### 4. Build-time linking against libcuobjclient

The C++ extension is compiled and linked against `libcuobjclient.so` at
build time. The `CuObjectClient` class holds a
`std::unique_ptr<cuObjClient>` and calls the cuObjClient C++ API directly.

**Rationale**:
- **Type safety**: Direct method calls with compiler-checked signatures.
- **Simpler code**: No `dlopen`/`dlsym` boilerplate.
- **Conditional build**: `setup.py` only compiles the extension when the SDK
  is found. Users without the SDK get a clean build with no cuObject support;
  a clear `ImportError` is raised at runtime if needed.

### 5. pybind11 extension over ctypes

The C++ wrapper is exposed to Python via pybind11 (`lmcache.lmcache_cuobject`)
rather than pure-Python ctypes bindings.

**Rationale**:
- **GIL release**: All methods use `py::call_guard<py::gil_scoped_release>()`
  so RDMA operations run concurrently with Python threads.
- **Type safety**: pybind11 handles Python/C++ type conversion and exception
  propagation automatically.

### 6. CRT `S3RequestType.DEFAULT` instead of `PUT_OBJECT` / `GET_OBJECT`

RDMA requests use `S3RequestType.DEFAULT` rather than the CRT-managed
`PUT_OBJECT` / `GET_OBJECT` types.

**Rationale**: The CRT's managed request types expect to handle the HTTP body
themselves (multipart upload, parallel download). RDMA requests have no HTTP
body — we need CRT only for signing, TLS, and connection management. Using
`DEFAULT` gives us full control over the request while still getting CRT's
auth and transport.

---

## C++ Extension Internals

### Token Generation (`generate_token`)

The core of the C++ extension is a single private method that both
`prepare_put` and `prepare_get` delegate to:

```cpp
std::string CuObjectClient::generate_token(uintptr_t data_ptr,
                                           size_t size, int op_type) {
    // 1. Validate data_ptr is within the registered pool
    size_t buffer_offset = data_ptr - pool_base_;

    // 2. Generate RDMA token
    char *desc_str = nullptr;
    client_->cuMemObjGetRDMAToken(pool_base_, size, buffer_offset,
                                   op_type, &desc_str);

    // 3. Copy and free
    std::string token(desc_str);
    client_->cuMemObjPutRDMAToken(desc_str);

    return token;
}
```

The pool base address and size are stored during `register_pool()` so that
the Python caller can pass the `MemoryObj`'s data pointer directly — the
C++ layer computes the buffer offset automatically.

### Thread Safety

`cuMemObjGetRDMAToken()` writes to a caller-provided `char**` output
pointer, so concurrent calls from different threads each get their own
descriptor allocation. No internal mutex is needed (unlike the previous
callback-based approach which required a mutex to protect shared
`captured_descriptor_` state).

### Token Lifetime

After `cuMemObjPutRDMAToken()` frees the library-allocated string, the
**token content** (as a copied `std::string`) remains valid for use in the
S3 request. The token encodes RDMA memory addresses and keys that reference
the RDMA memory registration made by `cuMemObjGetDescriptor()` — freeing
the string does not affect the RDMA hardware state. The registration
remains active until `cuMemObjPutDescriptor()` is called during cleanup.

---

## RDMA Reply Verification

The `x-amz-rdma-reply` response header is parsed by
`CuObjClientWrapper.parse_rdma_reply()`. It handles three formats:

| Format | Success | Failure |
|--------|---------|---------|
| **JSON** | `{"status": "ok"}`, `"success"`, `"complete"`, `"completed"`, `"done"` | `{"error": "..."}` or unrecognised status |
| **Numeric** | `"0"` (maps to `CUOBJ_SUCCESS`) | Any non-zero integer |
| **Keyword** | `"ok"`, `"success"`, `"complete"`, `"completed"`, `"done"` | Prefixes: `"error"`, `"fail"`, `"fault"` |

Any unrecognised, non-empty value is treated as failure to prevent silent data
corruption. This is a deliberate fail-safe choice.

---

## Why cuObject Connector Instead of NIXL's Built-in S3-over-RDMA

LMCache already uses NIXL's OBJ backend for S3 via the
`NixlDynamicStorageBackend` (in `nixl_storage_backend.py`). The cuObject
connector adds RDMA to the **separate, connector-based S3 path**. These two
paths serve different user populations and deployment models. Below are the
specific reasons for not routing the connector path through NIXL for RDMA.

### Different storage pipeline architectures

LMCache has two parallel S3 storage pipelines:

```
Pipeline 1 (connector-based):
  S3Connector → RemoteStorageBackend → CacheEngine
  Uses: MixedMemoryAllocator / PinMemoryAllocator / LazyMemoryAllocator
  API:  get(key) → MemoryObj,  put(key, MemoryObj)

Pipeline 2 (NIXL-based):
  NixlDynamicStorageBackend (with OBJ backend)
  Uses: PagedTensorMemoryAllocator
  API:  storage_to_mem(keys) via NIXL descriptor/transfer workflow
```

The cuObject connector adds RDMA to Pipeline 1. Using NIXL for this purpose
would mean either rewriting Pipeline 1 to speak NIXL's API (losing its
accumulated features) or building a complex bridge between the two allocator
and transfer models.

### Incompatible descriptor lifecycle

NIXL's OBJ backend requires S3 object keys to be baked into **storage
descriptors at registration time**. Every batch of keys in
`NixlDynamicStorageBackend` goes through a six-step ceremony:

```python
# For every batch:
descs = [NixlDesc(device_id=i, meta_info=format_key(key)) for i, key in ...]
storage_reg_descs, storage_xfer_handler = agent.create_batched_storage_handler(descs, page_size)
handle = agent.get_mem_to_storage_handle(mem_indices, storage_xfer_handler, storage_indices)
agent.post_blocking(handle)
agent.release_handle(handle)
agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)
```

Six NIXL API calls per batch just for the transfer ceremony. The cuObject
approach is one call: `cuMemObjGetRDMAToken(ptr, size, offset, op, &token)`.

### Loss of S3Connector control-plane features

The `S3Connector` has accumulated production-grade features that would be lost
or need reimplementation if the data path were routed through NIXL:

| Feature | S3Connector | NixlDynamicStorageBackend |
|---------|-------------|--------------------------|
| AWS credential chain | Full CRT provider (env, profile, IMDS, ECS) | Simple key/secret params |
| Circuit breaker | 3-strike auto-disable with recovery | None |
| Object size caching | In-memory `dict` avoids repeated HEADs | Per-key `query_memory` |
| Priority queue executor | 4-level priorities (PEEK/PREFETCH/GET/PUT) | None |
| HTTP/2 multiplexing | TLS ALPN `h2` negotiation | Controlled by NIXL plugin |
| S3 Express support | Dedicated `V4_S3EXPRESS` signing config | None |
| Batched async operations | `asyncio.gather` with `return_exceptions` | `asyncio.to_thread` batches |
| Zero-copy upload | `MemoryViewStream` over `memoryview` | Via NIXL transfer |

### Per-request fallback vs all-or-nothing

`CuObjectS3Connector` falls back to HTTP body transfer **per-request**. If a
single `prepare_put` call fails (e.g. RDMA resource exhaustion), that one
request is retried via HTTP; all other requests continue using RDMA.

NIXL's factory chooses one engine at init time. If RDMA fails at the plugin
level, the transfer raises `nixlBackendError` with no per-request fallback
to HTTP.

### Summary

| Concern | cuObject connector | NIXL OBJ backend |
|---------|-------------------|------------------|
| Integration target | Connector-based S3 pipeline | Separate NIXL storage pipeline |
| Per-request fallback | Yes (catch + `super()`) | No (factory-level) |
| Control-plane features | Inherited from `S3Connector` | Own implementation |
| Descriptor overhead | One `cuMemObjGetRDMAToken` call | 6 API calls per batch |
| Build dependency | cuObjClient SDK (conditional) | Full NIXL build |
| RDMA guarantee | Explicit (`_rdma_enabled` flag) | Depends on factory selection |

The two approaches coexist because they serve different deployment models.
The cuObject connector is the minimal, targeted way to add RDMA to the
connector-based pipeline without disrupting the NIXL-based pipeline or
sacrificing the S3Connector's mature feature set.

---

## Configuration

Enable via the `cuobj+s3://` URL scheme in the LMCache config:

```yaml
remote_url: "cuobj+s3://my-bucket.us-east-1.amazonaws.com"
extra_config:
  s3_region: "us-east-1"
  # Optional cuObject settings:
  cuobj_nic_device: "mlx5_0"          # RDMA NIC (default: auto-select)
```

Alternatively, use a standard `s3://` URL with `enable_cuobject: true` in
`extra_config`.

### Required

| Key | Description |
|-----|-------------|
| `s3_region` | AWS region (e.g. `us-east-1`) |

### Optional

| Key | Default | Description |
|-----|---------|-------------|
| `cuobj_nic_device` | auto | RDMA NIC device name (e.g. `mlx5_0`) |
| `aws_access_key_id` | env | Override AWS access key |
| `aws_secret_access_key` | env | Override AWS secret key |
| `s3_num_io_threads` | 64 | CRT I/O thread pool size |
| `s3_prefer_http2` | true | Negotiate HTTP/2 via ALPN |
| `disable_tls` | false | Disable TLS (for testing) |

---

## Prerequisites

- **CUDA Toolkit >= 13.1.1** with cuObject client SDK
  (`cuobjclient.h` and `libcuobjclient.so`)
- **LMCache built with the cuObject extension** — the SDK must be present at
  build time (set `CUOBJECT_INCLUDE_DIR` / `CUOBJECT_LIB_DIR` if not in
  standard paths). Do not use `NO_CUOBJECT=1` or `NO_CUDA_EXT=1`.
- **Pinned CPU memory** — the `local_cpu_backend` must use
  `MixedMemoryAllocator`, `PinMemoryAllocator`, or `LazyMemoryAllocator`
  (all provide a contiguous pinned buffer)
- **RDMA NIC** — InfiniBand or RoCEv2 capable
- **cuObject-enabled S3 endpoint** — e.g. Dell ObjectScale with RDMA, or an
  NVIDIA-accelerated S3 server
