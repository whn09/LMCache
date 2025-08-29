#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Launch vLLM with LMCache using environment variables
# This approach is often more reliable than config files

MODEL="${1:-deepseek-ai/DeepSeek-V2-Lite}"
PORT="${2:-8000}"

echo "========================================"
echo "Starting vLLM with LMCache (ENV method)"
echo "========================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "========================================"

# Set LMCache environment variables directly
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=20
export LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_SAVE_PREFIX_CACHE=True
export LMCACHE_LOG_LEVEL=info

# For MLA models
export LMCACHE_REMOTE_SERDE=naive
export LMCACHE_USE_LAYERWISE=False

# Debug: Show LMCache is enabled
export LMCACHE_ENABLED=True

echo "LMCache Environment Variables Set:"
echo "  LMCACHE_CHUNK_SIZE=$LMCACHE_CHUNK_SIZE"
echo "  LMCACHE_LOCAL_CPU=$LMCACHE_LOCAL_CPU"
echo "  LMCACHE_MAX_LOCAL_CPU_SIZE=$LMCACHE_MAX_LOCAL_CPU_SIZE"
echo "  LMCACHE_REMOTE_SERDE=$LMCACHE_REMOTE_SERDE"
echo "========================================"

# Launch vLLM with proper KV transfer config
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both", "enable_cache":true}'