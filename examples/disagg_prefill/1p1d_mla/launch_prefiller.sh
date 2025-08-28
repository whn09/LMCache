#!/bin/bash

# Launch script for MLA prefiller with DeepSeek-V2-Lite model
# This script launches the prefiller service that handles prefill requests

MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=8100
HOST="localhost"
TENSOR_PARALLEL=2  # DeepSeek-V2-Lite benefits from tensor parallelism

# LMCache configuration
export LMCACHE_CONFIG_FILE=configs/lmcache-prefiller-config.yaml

# Optional: Set environment variables for performance
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1 for TP=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting MLA Prefiller with DeepSeek-V2-Lite model"
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
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}' \
    2>&1 | tee prefiller_mla.log

    # --kv-connector lmcache \
    # --kv-role fill \
    # --kv-transfer-mode prefill \