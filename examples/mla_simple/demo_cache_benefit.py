#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simple demonstration of MLA KV cache benefits with LMCache.
This script shows how cache reuse dramatically improves performance.
"""

import time
import argparse
from openai import OpenAI


def demonstrate_cache_benefit(api_base="http://localhost:8000/v1", 
                            model="deepseek-ai/DeepSeek-V2-Lite"):
    """
    Demonstrate the performance benefit of KV cache reuse.
    """
    client = OpenAI(api_key="EMPTY", base_url=api_base)
    
    # Use a much longer context to make cache benefits more apparent
    long_context = """
    The transformer architecture has revolutionized natural language processing
    through its self-attention mechanism. Multi-head attention allows models to
    attend to different positions and capture various relationships. However,
    the quadratic complexity of attention with respect to sequence length poses
    significant computational challenges for long contexts.
    
    Multi-head Latent Attention (MLA) addresses these challenges by compressing
    the key-value pairs into a latent space, reducing memory requirements while
    maintaining model performance. This compression is particularly beneficial
    for inference scenarios where the KV cache can dominate memory usage.
    
    In traditional transformers, each attention head maintains separate key and
    value projections, leading to substantial memory overhead. MLA instead uses
    a shared latent representation that is then projected to head-specific
    keys and values, achieving significant compression ratios.
    
    The benefits of MLA become even more pronounced in production environments
    where multiple inference requests may share common prefixes or contexts.
    By enabling efficient KV cache sharing and reuse, MLA can dramatically
    reduce both memory consumption and computation time. This is especially
    valuable for applications like chatbots, code assistants, and document
    analysis systems where context reuse is common.
    
    Furthermore, MLA's compression doesn't significantly impact model quality.
    Empirical studies have shown that MLA models can achieve comparable or
    even superior performance to traditional multi-head attention while using
    a fraction of the memory. This makes MLA particularly attractive for
    deploying large language models in resource-constrained environments.
    
    The implementation of MLA requires careful consideration of the latent
    dimension size and the projection mechanisms. Too small a latent dimension
    may lose important information, while too large defeats the purpose of
    compression. Modern MLA implementations like those in DeepSeek models
    have found effective balance points that maximize both efficiency and
    model capability.
    """ * 3  # Repeat to make it even longer
    
    prompt = long_context + "\n\nQuestion: What is the main benefit of MLA?"
    
    print("="*70)
    print("MLA KV Cache Benefit Demonstration")
    print("="*70)
    print(f"Model: {model}")
    print(f"Context length: ~{len(long_context.split())} words")
    print("="*70)
    
    # First request (cold cache)
    print("\n1️⃣  First request (cold cache - needs to process full context)...")
    start = time.time()
    
    response1 = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=50,
        temperature=0,
        stream=False
    )
    
    time1 = time.time() - start
    print(f"   Time: {time1:.2f}s")
    print(f"   Response: {response1.choices[0].text.strip()[:100]}...")
    
    # Small delay
    time.sleep(1)
    
    # Second request (warm cache)
    print("\n2️⃣  Second request (warm cache - reuses KV cache)...")
    start = time.time()
    
    response2 = client.completions.create(
        model=model,
        prompt=prompt,  # Same prompt
        max_tokens=50,
        temperature=0,
        stream=False
    )
    
    time2 = time.time() - start
    print(f"   Time: {time2:.2f}s")
    print(f"   Response: {response2.choices[0].text.strip()[:100]}...")
    
    # Third request with slightly different question (partial cache reuse)
    prompt_variant = long_context + "\n\nQuestion: How does MLA reduce memory usage?"
    
    print("\n3️⃣  Third request (partial cache - reuses context, new question)...")
    start = time.time()
    
    response3 = client.completions.create(
        model=model,
        prompt=prompt_variant,
        max_tokens=50,
        temperature=0,
        stream=False
    )
    
    time3 = time.time() - start
    print(f"   Time: {time3:.2f}s")
    print(f"   Response: {response3.choices[0].text.strip()[:100]}...")
    
    # Calculate speedups
    print("\n" + "="*70)
    print("📊 Performance Summary:")
    print("="*70)
    print(f"First request (cold):     {time1:.2f}s (baseline)")
    print(f"Second request (cached):   {time2:.2f}s ({time1/time2:.1f}x speedup)")
    print(f"Third request (partial):   {time3:.2f}s ({time1/time3:.1f}x speedup)")
    
    if time2 < time1 * 0.5:
        print("\n✅ Excellent! Cache is providing >2x speedup!")
    elif time2 < time1 * 0.8:
        print("\n✅ Good! Cache is providing measurable speedup.")
    else:
        print("\n⚠️  Cache benefit is limited. Check configuration.")
    
    return time1, time2, time3


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate MLA KV cache benefits"
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model to use"
    )
    
    args = parser.parse_args()
    
    try:
        demonstrate_cache_benefit(args.api_base, args.model)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the vLLM server is running with LMCache enabled:")
        print("  ./launch_server.sh")


if __name__ == "__main__":
    main()