# Simple MLA (Multi-head Latent Attention) Example with LMCache

This example demonstrates how to use MLA models (such as DeepSeek-V2) with LMCache for efficient KV cache management, without the complexity of prefill-decode separation.

## Overview

Multi-head Latent Attention (MLA) is an optimization technique that compresses the KV cache by using latent representations, significantly reducing memory usage while maintaining model performance. This example shows:

- Basic MLA model inference with LMCache
- KV cache reuse for improved performance
- Benchmarking and performance metrics
- Simple configuration for MLA models

## Prerequisites

1. **Install LMCache and dependencies:**
   ```bash
   pip install lmcache
   pip install vllm
   pip install openai transformers
   ```

2. **Download an MLA model (optional):**
   The example uses `deepseek-ai/DeepSeek-V2-Lite` by default, which will be downloaded automatically on first use.

## Files

- `mla_example.py` - Main example script with MLA inference client
- `lmcache_config.yaml` - LMCache configuration optimized for MLA
- `launch_server.sh` - Helper script to launch vLLM server with LMCache
- `README.md` - This documentation

## Quick Start

### Step 1: Start the vLLM Server with LMCache

```bash
# Using the launch script (recommended)
./launch_server.sh --model deepseek-ai/DeepSeek-V2-Lite --port 8000

# Or manually
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --port 8000 \
    --enable-lmcache \
    --lmcache-config-file lmcache_config.yaml \
    --trust-remote-code
```

### Step 2: Run the Example

```bash
# Run default demo
python mla_example.py

# Run with custom prompt
python mla_example.py --prompt "Explain quantum computing in simple terms"

# Run full benchmark suite
python mla_example.py --benchmark

# Use different model or server
python mla_example.py --api-base http://localhost:8080/v1 --model your-mla-model
```

## Configuration

The `lmcache_config.yaml` file contains important settings for MLA models:

```yaml
# Key MLA-specific settings
remote_serde: "naive"      # Required for MLA models
use_layerwise: false        # Not supported for MLA yet
chunk_size: 256            # Optimal chunk size for MLA
local_cpu: true            # Use CPU memory for caching
max_local_cpu_size: 10.0   # GB of CPU memory for cache
```

### Important MLA Configuration Notes:

1. **Serialization**: MLA models require `naive` serde for proper operation
2. **Layerwise Operations**: Currently not supported for MLA models
3. **Memory Management**: Can use both CPU and GPU memory for caching
4. **Cache Persistence**: Enable `save_decode_cache` for cross-session reuse

## Usage Examples

### Basic Completion
```python
from mla_example import MLAInferenceClient

client = MLAInferenceClient()
result = client.generate_completion(
    "What is MLA?", 
    max_tokens=100
)
print(f"TTFT: {result['ttft_ms']}ms")
```

### Chat Completion
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain MLA benefits"}
]
result = client.chat_completion(messages, max_tokens=150)
```

### Testing KV Cache Reuse
```python
# Send same prompt multiple times to test cache
results = client.test_kv_cache_reuse(
    "Your prompt here",
    num_iterations=3
)
# Second and third iterations should be faster
```

## Performance Optimization

### 1. Enable GPU Caching
Modify `lmcache_config.yaml`:
```yaml
local_device: "cuda"
max_local_gpu_size: 4.0  # GB
```

### 2. Adjust Chunk Size
For longer contexts, increase chunk size:
```yaml
chunk_size: 512  # or 1024 for very long contexts
```

### 3. Use Remote Storage for Persistence
Add Redis backend for persistent caching:
```yaml
remote_backend: "redis"
remote_host: "localhost"
remote_port: 6379
```

## Benchmarking

The example includes built-in benchmarks:

```bash
# Run full benchmark suite
python mla_example.py --benchmark
```

This will test:
1. Simple completion performance
2. KV cache reuse effectiveness
3. Chat completion performance

Expected improvements with cache:
- 2-5x faster TTFT on cache hits
- Reduced memory usage compared to standard attention
- Better throughput for repeated contexts

## Troubleshooting

### Issue: "MLA not supported" error
- Ensure you're using an MLA-compatible model (e.g., DeepSeek-V2 series)
- Verify `remote_serde: "naive"` in config

### Issue: Slow performance
- Check available memory (CPU/GPU)
- Adjust `chunk_size` based on your context length
- Enable GPU caching if available

### Issue: Cache not working
- Verify LMCache is enabled in vLLM server
- Check config file path is correct
- Ensure sufficient memory allocated

## Advanced Usage

### Custom Cache Policies
```yaml
eviction_policy: "lru"  # Options: lru, fifo, lfu
```

### Monitoring
```yaml
enable_metrics: true
log_level: "debug"
```

### Multi-GPU Setup
```bash
./launch_server.sh --tp-size 2  # For 2 GPUs
```

## Model Compatibility

MLA examples work with:
- DeepSeek-V2 series (Lite, Chat, Base)
- Other models implementing MLA architecture
- Custom MLA models with proper configuration

## References

- [LMCache Documentation](https://github.com/LMCache/LMCache)
- [DeepSeek MLA Paper](https://arxiv.org/abs/2405.04434)
- [vLLM Integration Guide](https://docs.vllm.ai/)