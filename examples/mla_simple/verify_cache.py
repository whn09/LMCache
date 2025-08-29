#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic script to verify LMCache is working properly with MLA models.
"""

import time
import json
import argparse
import requests
from typing import Dict, Any


def check_server_status(api_base: str) -> bool:
    """Check if vLLM server is running and accessible."""
    try:
        response = requests.get(f"{api_base}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✓ Server is running with models: {models}")
            return True
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
    return False


def test_cache_hit(api_base: str, model: str) -> Dict[str, Any]:
    """
    Test if cache hits are working by sending identical requests.
    """
    url = f"{api_base}/completions"
    
    # Use a distinctive prompt that should be cacheable
    test_prompt = """System: You are a helpful AI assistant.
Context: The following is technical documentation about LMCache and MLA optimization.
LMCache is a KV cache management system that enables efficient sharing and reuse of 
key-value caches across multiple LLM inference requests. MLA (Multi-head Latent Attention) 
is an optimization that compresses KV pairs into a latent space.

Question: What is LMCache?
Answer:"""
    
    payload = {
        "model": model,
        "prompt": test_prompt,
        "max_tokens": 30,
        "temperature": 0.0,  # Deterministic for testing
        "top_p": 1.0,
        "stream": False,
        "n": 1,
        "seed": 42  # Fixed seed for reproducibility
    }
    
    results = {}
    
    print("\n=== Cache Hit Test ===")
    print(f"Sending identical requests to test cache...")
    
    # First request (cold cache)
    print("\n1. First request (cold cache)...")
    start = time.time()
    response1 = requests.post(url, json=payload)
    time1 = time.time() - start
    
    if response1.status_code == 200:
        text1 = response1.json()["choices"][0]["text"]
        print(f"   ✓ Completed in {time1:.3f}s")
        print(f"   Response: {text1[:50]}...")
        results["cold_time"] = time1
        results["cold_text"] = text1
    else:
        print(f"   ✗ Failed: {response1.status_code}")
        return results
    
    # Wait briefly
    time.sleep(0.5)
    
    # Second request (should hit cache)
    print("\n2. Second request (should hit cache)...")
    start = time.time()
    response2 = requests.post(url, json=payload)
    time2 = time.time() - start
    
    if response2.status_code == 200:
        text2 = response2.json()["choices"][0]["text"]
        print(f"   ✓ Completed in {time2:.3f}s")
        print(f"   Response: {text2[:50]}...")
        results["warm_time"] = time2
        results["warm_text"] = text2
        
        # Check if responses are identical (they should be with temp=0)
        if text1 == text2:
            print("   ✓ Responses are identical (good for deterministic generation)")
        else:
            print("   ⚠ Responses differ (may indicate cache miss or non-deterministic generation)")
        
        # Calculate speedup
        speedup = time1 / time2 if time2 > 0 else 0
        results["speedup"] = speedup
        print(f"\n   📊 Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("   ✓ Cache is working effectively!")
        elif speedup > 1.1:
            print("   ⚠ Cache provides modest improvement")
        else:
            print("   ✗ Cache doesn't seem to be working")
            
    else:
        print(f"   ✗ Failed: {response2.status_code}")
    
    return results


def test_prefix_sharing(api_base: str, model: str) -> Dict[str, Any]:
    """
    Test prefix sharing by sending requests with shared prefix.
    """
    url = f"{api_base}/completions"
    
    # Shared prefix
    prefix = """Technical Documentation:
LMCache is an advanced caching system for Large Language Models that enables:
1. KV cache sharing across requests
2. Efficient memory management
3. Reduced inference latency
4. Support for various storage backends

"""
    
    # Different suffixes
    suffixes = [
        "Question: What is the main purpose of LMCache?\nAnswer:",
        "Question: How does LMCache reduce latency?\nAnswer:",
        "Question: What storage backends are supported?\nAnswer:"
    ]
    
    print("\n=== Prefix Sharing Test ===")
    print("Testing cache reuse with shared prefix...")
    
    results = []
    
    for i, suffix in enumerate(suffixes):
        prompt = prefix + suffix
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 30,
            "temperature": 0.0,
            "stream": False
        }
        
        print(f"\n{i+1}. Request with suffix: '{suffix[:30]}...'")
        start = time.time()
        response = requests.post(url, json=payload)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            text = response.json()["choices"][0]["text"]
            print(f"   ✓ Completed in {elapsed:.3f}s")
            print(f"   Response: {text[:50]}...")
            results.append({"time": elapsed, "success": True})
        else:
            print(f"   ✗ Failed: {response.status_code}")
            results.append({"time": elapsed, "success": False})
    
    if len(results) >= 2 and results[0]["success"] and results[1]["success"]:
        speedup = results[0]["time"] / results[1]["time"] if results[1]["time"] > 0 else 0
        print(f"\n   📊 Prefix sharing speedup (1st vs 2nd): {speedup:.2f}x")
        
        if speedup > 1.3:
            print("   ✓ Prefix sharing is working!")
        else:
            print("   ⚠ Prefix sharing benefit is limited")
    
    return {"results": results}


def main():
    parser = argparse.ArgumentParser(
        description="Verify LMCache is working with MLA models"
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model to test"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("LMCache Verification for MLA Models")
    print("="*60)
    
    # Check server
    if not check_server_status(args.api_base):
        print("\n❌ Server is not running. Please start it with:")
        print("   ./launch_server.sh")
        return
    
    # Run tests
    cache_results = test_cache_hit(args.api_base, args.model)
    
    if args.full:
        prefix_results = test_prefix_sharing(args.api_base, args.model)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if "speedup" in cache_results:
        speedup = cache_results["speedup"]
        if speedup > 1.5:
            print("✅ LMCache is working well with your MLA model!")
            print(f"   Cache provides {speedup:.1f}x speedup")
        elif speedup > 1.1:
            print("⚠️  LMCache provides modest improvements")
            print(f"   Cache provides {speedup:.1f}x speedup")
            print("\nConsider:")
            print("  - Using longer contexts for better cache benefit")
            print("  - Ensuring GPU memory is available for caching")
            print("  - Checking if the model supports KV caching properly")
        else:
            print("❌ LMCache doesn't seem to be providing benefits")
            print("\nTroubleshooting:")
            print("  1. Check server logs for LMCache initialization")
            print("  2. Verify configuration file is loaded")
            print("  3. Ensure model supports KV caching")
            print("  4. Try with longer contexts")
    else:
        print("❌ Could not complete cache tests")


if __name__ == "__main__":
    main()