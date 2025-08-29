# MLA Cache Performance Troubleshooting

Based on your test results showing limited cache speedup (1.1-1.2x), here are potential issues and solutions:

## Common Issues & Solutions

### 1. Cache Not Actually Enabled
**Symptoms:** Very small speedup (< 1.2x)

**Solutions:**
- Use environment variables instead of config file:
  ```bash
  ./launch_with_env.sh
  ```
- Check server logs for "LMCache enabled" messages
- Verify with: `curl http://localhost:8000/v1/models`

### 2. Model-Specific Limitations
**Issue:** Some MLA models may not fully support KV caching in vLLM

**Solutions:**
- Try different MLA models if available
- Check vLLM version compatibility
- Consider using the disaggregated prefill-decode setup for better performance

### 3. Context Length Too Short
**Issue:** Cache benefits are minimal with short contexts

**Solutions:**
- Use longer prompts (1000+ tokens)
- Test with document-based queries
- Try the context sharing test: `python test_simple.py --test context`

### 4. Memory Allocation Issues
**Issue:** Insufficient memory allocated for caching

**Solutions:**
- Increase CPU cache size:
  ```bash
  export LMCACHE_MAX_LOCAL_CPU_SIZE=30
  ```
- Enable GPU caching:
  ```bash
  export LMCACHE_LOCAL_DEVICE=cuda
  export LMCACHE_MAX_LOCAL_GPU_SIZE=4
  ```

### 5. vLLM Integration Issues
**Issue:** vLLM may not be properly using LMCache

**Solutions:**
- Ensure vLLM is built with LMCache support
- Try different vLLM versions
- Use explicit KV transfer config:
  ```python
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
  ```

## Diagnostic Steps

1. **Check if LMCache is loaded:**
   ```bash
   # Look for LMCache initialization in server logs
   grep -i "lmcache" server.log
   ```

2. **Test with simple prompt:**
   ```bash
   python test_simple.py --test simple
   ```

3. **Monitor memory usage:**
   ```bash
   # While running tests
   watch -n 1 'free -h'
   ```

4. **Try different configurations:**
   - CPU-only: `simple_lmcache_config.yaml`
   - With environment vars: `launch_with_env.sh`
   - Original config: `lmcache_config.yaml`

## Expected Performance

With proper setup, you should see:
- **Exact prompt reuse:** 2-5x speedup
- **Prefix sharing:** 1.5-3x speedup
- **Long contexts:** Better speedup with longer texts

## Alternative Approaches

If cache benefits remain limited:

1. **Use Disaggregated Setup:**
   The 1p1d_mla example in `../disagg_prefill/1p1d_mla/` may provide better performance

2. **Try Different Storage Backends:**
   - Redis for persistent caching
   - Local disk for larger cache capacity

3. **Batch Processing:**
   Process multiple requests with shared contexts together

## Getting Help

- Check vLLM logs: `journalctl -u vllm -f`
- LMCache debug logs: Set `LMCACHE_LOG_LEVEL=debug`
- GitHub issues: https://github.com/LMCache/LMCache/issues