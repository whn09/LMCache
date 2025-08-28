# MLA Disaggregated Prefill/Decode Example

This example demonstrates how to use LMCache with MLA (Multi-head Latent Attention) models in a disaggregated prefill/decode architecture using DeepSeek-V2-Lite.

## Overview

MLA is an optimization technique used in models like DeepSeek that compresses the KV cache representation, reducing memory usage and improving efficiency. This example shows:

- How to configure LMCache for MLA models
- Running disaggregated prefill and decode services
- Handling MLA-specific constraints

## Key Differences from Standard Models

1. **KV Cache Shape**: MLA models use a different KV cache format
   - Standard: `(num_layers, 2, chunk_size, num_kv_heads, head_size)` - separate K and V
   - MLA: `(num_layers, 1, chunk_size, num_kv_heads, head_size)` - compressed representation

2. **Serialization**: MLA models only support `naive` serde mode

3. **Layerwise Transfer**: Not yet supported for MLA models

## Prerequisites

1. Install LMCache and dependencies:
```bash
pip install -e /path/to/LMCache
```

2. Download the DeepSeek-V2-Lite model (will auto-download on first run):
```bash
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite
```

3. Ensure you have at least 4 GPUs available for tensor parallelism (TP=2 for both services)

## Configuration Files

- `configs/lmcache-prefiller-config.yaml`: Configuration for the prefiller service
- `configs/lmcache-decoder-config.yaml`: Configuration for the decoder service

Key MLA-specific settings:
```yaml
remote_serde: "naive"  # MLA only works with naive serde
use_layerwise: False   # Layerwise not supported for MLA
```

## Running the Example

### Step 1: Start the Decoder Service

```bash
cd examples/disagg_prefill/1p1d_mla
bash launch_decoder.sh
```

This will:
- Start the decoder on port 8200
- Use GPUs 2,3 with tensor parallelism
- Configure NIXL receiver for KV cache transfer

### Step 2: Start the Prefiller Service

In a new terminal:
```bash
cd examples/disagg_prefill/1p1d_mla
bash launch_prefiller.sh
```

This will:
- Start the prefiller on port 8100
- Use GPUs 0,1 with tensor parallelism
- Configure NIXL sender for KV cache transfer

### Step 3: Start the Proxy Server

In a new terminal:
```bash
cd examples/disagg_prefill/1p1d_mla
python proxy_server.py
```

The proxy server:
- Listens on port 8000
- Routes requests between prefiller and decoder
- Manages the disaggregated workflow

### Step 4: Test the System

```bash
# Single request test
python test_mla_inference.py

# Concurrent requests test
python test_mla_inference.py --test-concurrent --num-concurrent 10
```

## Performance Tuning

### Memory Management

For high concurrency scenarios with limited GPU memory:

1. **Disable tensor cloning** (saves memory but may cause issues):
```bash
export LMCACHE_NIXL_ENABLE_CLONE=0
```

2. **Enable PyTorch memory optimization**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

3. **Adjust buffer sizes** in config files:
```yaml
nixl_buffer_size: 4294967296  # Reduce if OOM
```

### Garbage Collection

The decoder configuration enables aggressive GC:
```yaml
nixl_enable_gc: True  # Enable garbage collection
```

## Monitoring

Check the logs for:
- `prefiller_mla.log`: Prefiller service logs
- `decoder_mla.log`: Decoder service logs
- Proxy console output: TTFT statistics and request metrics

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors with high concurrency:
1. Reduce `nixl_buffer_size` in decoder config
2. Set `LMCACHE_NIXL_ENABLE_CLONE=0`
3. Reduce `max-num-seqs` in launch scripts
4. Use fewer concurrent requests

### Model Loading Issues

If the model fails to load:
1. Ensure you have enough GPU memory (DeepSeek-V2-Lite requires ~30GB with TP=2)
2. Check CUDA_VISIBLE_DEVICES settings
3. Verify model is downloaded: `huggingface-cli download deepseek-ai/DeepSeek-V2-Lite`

### Transfer Failures

If KV cache transfer fails:
1. Check network connectivity between services
2. Verify NIXL ports are not blocked
3. Ensure buffer sizes are sufficient for your workload

## Supported MLA Models

- `deepseek-ai/DeepSeek-V2-Lite` (used in this example)
- `deepseek-ai/DeepSeek-V3`
- `deepseek-ai/DeepSeek-R1`

## Limitations

1. **No Layerwise Support**: MLA models don't yet support layerwise KV transfer
2. **Naive Serde Only**: Cannot use optimized serialization formats
3. **Memory Intensive**: Requires careful memory management for high concurrency

## References

- [DeepSeek MLA Paper](https://arxiv.org/abs/2405.04434)
- [LMCache Documentation](https://github.com/LMCache/LMCache)
- [vLLM Documentation](https://docs.vllm.ai/)