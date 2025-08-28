#!/bin/bash

# Launch script for MLA decoder with DeepSeek-V2-Lite model
# This script launches the decoder service that handles decode requests

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=8200
HOST="localhost"
TENSOR_PARALLEL=2  # DeepSeek-V2-Lite benefits from tensor parallelism

# LMCache configuration
export LMCACHE_CONFIG_FILE=configs/lmcache-decoder-config.yaml

# Optional: Set environment variables for performance
export CUDA_VISIBLE_DEVICES=2,3  # Use GPUs 2 and 3 for TP=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For high concurrency scenarios, disable cloning to save memory
# export LMCACHE_NIXL_ENABLE_CLONE=0

echo "Starting MLA Decoder with DeepSeek-V2-Lite model"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "LMCache Config: $LMCACHE_CONFIG_FILE"
echo "----------------------------------------"

# Launch vLLM with LMCache and MLA support
vllm serve $MODEL \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 256 \
    --enforce-eager \
    --disable-log-requests \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.75 \
    --disable-log-requests \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}' \
    2>&1 | tee decoder_mla.log

    # --kv-connector lmcache \
    # --kv-role reuse \
    # --kv-transfer-mode decode \