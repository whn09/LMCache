#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simplified test to verify MLA cache behavior.
"""

import time
import argparse
from openai import OpenAI


def simple_cache_test(api_base="http://localhost:8000/v1"):
    """
    Very simple test with exact same prompt multiple times.
    """
    client = OpenAI(api_key="EMPTY", base_url=api_base)
    model = "deepseek-ai/DeepSeek-V2-Lite"
    
    # Fixed prompt for testing
    test_prompt = "The capital of France is"
    
    print("="*50)
    print("Simple MLA Cache Test")
    print("="*50)
    print(f"Prompt: '{test_prompt}'")
    print("="*50)
    
    times = []
    
    for i in range(5):
        print(f"\nRequest {i+1}:")
        start = time.time()
        
        response = client.completions.create(
            model=model,
            prompt=test_prompt,
            max_tokens=10,
            temperature=0,
            seed=42
        )
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        text = response.choices[0].text.strip()
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Response: {text}")
    
    print("\n" + "="*50)
    print("Timing Analysis:")
    print("="*50)
    for i, t in enumerate(times):
        if i == 0:
            print(f"Request 1 (cold): {t:.3f}s (baseline)")
        else:
            speedup = times[0] / t
            print(f"Request {i+1} (warm): {t:.3f}s ({speedup:.2f}x speedup)")
    
    avg_warm = sum(times[1:]) / len(times[1:]) if len(times) > 1 else 0
    if avg_warm > 0:
        avg_speedup = times[0] / avg_warm
        print(f"\nAverage warm speedup: {avg_speedup:.2f}x")


def context_sharing_test(api_base="http://localhost:8000/v1"):
    """
    Test with shared context prefix.
    """
    client = OpenAI(api_key="EMPTY", base_url=api_base)
    model = "deepseek-ai/DeepSeek-V2-Lite"
    
    # Shared context
    context = "Paris is the capital and largest city of France. It is located on the Seine River in northern France. "
    
    questions = [
        "What river is Paris located on?",
        "Is Paris the largest city?",
        "Where is Paris located?"
    ]
    
    print("\n" + "="*50)
    print("Context Sharing Test")
    print("="*50)
    print(f"Context: {context[:50]}...")
    print("="*50)
    
    times = []
    
    for i, q in enumerate(questions):
        prompt = context + f"Question: {q} Answer:"
        print(f"\nQuestion {i+1}: {q}")
        
        start = time.time()
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=20,
            temperature=0
        )
        elapsed = time.time() - start
        times.append(elapsed)
        
        text = response.choices[0].text.strip()
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Answer: {text[:50]}")
    
    if len(times) >= 2:
        speedup = times[0] / times[1]
        print(f"\nContext reuse speedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--test", choices=["simple", "context", "both"], default="both")
    args = parser.parse_args()
    
    if args.test in ["simple", "both"]:
        simple_cache_test(args.api_base)
    
    if args.test in ["context", "both"]:
        context_sharing_test(args.api_base)


if __name__ == "__main__":
    main()