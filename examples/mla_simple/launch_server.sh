#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Launch script for vLLM server with LMCache for MLA models

# Default values
MODEL="deepseek-ai/DeepSeek-V2-Lite"
PORT=8000
TP_SIZE=1
LMCACHE_CONFIG="lmcache_config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --config)
            LMCACHE_CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL         Model to use (default: deepseek-ai/DeepSeek-V2-Lite)"
            echo "  --port PORT          Server port (default: 8000)"
            echo "  --tp-size SIZE       Tensor parallel size (default: 1)"
            echo "  --config FILE        LMCache config file (default: lmcache_config.yaml)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$LMCACHE_CONFIG" ]; then
    echo "Error: Config file $LMCACHE_CONFIG not found!"
    exit 1
fi

echo "========================================"
echo "Starting vLLM Server with LMCache"
echo "========================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "LMCache Config: $LMCACHE_CONFIG"
echo "========================================"

# Set environment variable for LMCache
export LMCACHE_CONFIG_FILE=$LMCACHE_CONFIG

# Launch vLLM server with LMCache integration
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
