#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script for MLA disaggregated inference with DeepSeek models.
"""

import json
import time
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_single_request(proxy_url: str, prompt: str, max_tokens: int = 100):
    """Send a single test request to the proxy server."""
    
    url = f"{proxy_url}/v1/completions"
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    
    print(f"Sending request with prompt: '{prompt[:50]}...'")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        # Process streaming response
        full_text = ""
        first_token_time = None
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            text = data["choices"][0].get("text", "")
                            if text and first_token_time is None:
                                first_token_time = time.time()
                            full_text += text
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else total_time
        
        print(f"✓ Response received:")
        print(f"  - Generated text: {full_text[:100]}...")
        print(f"  - Total tokens: ~{len(full_text.split())}")
        print(f"  - TTFT: {ttft*1000:.2f}ms")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Throughput: {len(full_text.split())/total_time:.2f} tokens/s")
        
        return {"success": True, "ttft": ttft, "total_time": total_time}
        
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return {"success": False, "error": str(e)}


def test_concurrent_requests(proxy_url: str, num_requests: int = 5):
    """Test concurrent requests to measure system under load."""
    
    print(f"\n=== Testing {num_requests} concurrent requests ===")
    
    # Different prompts for variety
    prompts = [
        "Explain the concept of multi-head latent attention in deep learning models.",
        "What are the advantages of disaggregated prefill and decode in LLM serving?",
        "Describe how KV cache transfer works in distributed inference systems.",
        "Compare transformer models with and without MLA optimization.",
        "Explain the role of tensor parallelism in large language model inference.",
    ]
    
    results = []
    
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            future = executor.submit(test_single_request, proxy_url, prompt, 50)
            futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    if successful:
        avg_ttft = sum(r["ttft"] for r in successful) / len(successful)
        avg_total = sum(r["total_time"] for r in successful) / len(successful)
        
        print(f"\n=== Concurrent Test Results ===")
        print(f"Successful requests: {len(successful)}/{num_requests}")
        print(f"Average TTFT: {avg_ttft*1000:.2f}ms")
        print(f"Average total time: {avg_total:.2f}s")
    else:
        print("All requests failed!")


def main():
    parser = argparse.ArgumentParser(description="Test MLA disaggregated inference")
    parser.add_argument("--proxy-url", type=str, default="http://localhost:8000",
                        help="URL of the proxy server")
    parser.add_argument("--test-concurrent", action="store_true",
                        help="Test concurrent requests")
    parser.add_argument("--num-concurrent", type=int, default=5,
                        help="Number of concurrent requests")
    
    args = parser.parse_args()
    
    print("""
    ======================================
    MLA Disaggregated Inference Test
    ======================================
    Testing DeepSeek-V2-Lite with MLA
    ======================================
    """)
    
    # Test single request
    print("\n=== Testing single request ===")
    test_prompt = "Explain how multi-head latent attention (MLA) improves efficiency in large language models"
    test_single_request(args.proxy_url, test_prompt, max_tokens=100)
    
    # Test concurrent requests if requested
    if args.test_concurrent:
        time.sleep(2)  # Brief pause
        test_concurrent_requests(args.proxy_url, args.num_concurrent)


if __name__ == "__main__":
    main()