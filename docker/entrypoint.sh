#!/bin/bash
# ---------------------------------------------------------------------------
# entrypoint.sh -- LMCache + vLLM + cuObject S3 RDMA
#
# Default: starts vLLM with LMCache KV connector.
# Override: pass any command to run instead (e.g., bash, python, etc.)
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Env defaults (override via docker run -e) ----------------------------
: "${MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${PORT:=8000}"
: "${TP_SIZE:=1}"
: "${LOAD_FORMAT:=dummy}"
: "${KV_CONNECTOR:=LMCacheConnectorV1}"
: "${KV_ROLE:=kv_both}"
: "${PYTHONHASHSEED:=0}"
: "${LMCACHE_MAX_LOCAL_CPU_SIZE:=66}"

export PYTHONHASHSEED
export LMCACHE_MAX_LOCAL_CPU_SIZE

# ---- Verify cuObject (optional, non-fatal) --------------------------------
if python -c "from lmcache.lmcache_cuobject import CuObjectClient" 2>/dev/null; then
    echo "[entrypoint] cuObject extension: available"
else
    echo "[entrypoint] cuObject extension: NOT available (RDMA will fall back to HTTP)"
fi

# ---- If user passes a custom command, run it instead ----------------------
if [ $# -gt 0 ]; then
    exec "$@"
fi

# ---- Default: start vLLM with LMCache ------------------------------------
echo "[entrypoint] Starting vLLM server..."
echo "[entrypoint]   Model:    ${MODEL}"
echo "[entrypoint]   Port:     ${PORT}"
echo "[entrypoint]   TP size:  ${TP_SIZE}"
echo "[entrypoint]   KV role:  ${KV_ROLE}"

exec vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --load-format "${LOAD_FORMAT}" \
    --kv-transfer-config \
    "{\"kv_connector\": \"${KV_CONNECTOR}\", \"kv_role\": \"${KV_ROLE}\"}"
