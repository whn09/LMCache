#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simple MLA (Multi-head Latent Attention) example with LMCache.
This example demonstrates using MLA models (like DeepSeek-V2) with LMCache
without prefill-decode separation.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from openai import OpenAI


class MLAInferenceClient:
    """Client for MLA model inference with LMCache."""
    
    def __init__(self, api_base: str = "http://localhost:8000/v1", 
                 api_key: str = "EMPTY",
                 model: str = "deepseek-ai/DeepSeek-V2-Lite"):
        """
        Initialize the MLA inference client.
        
        Args:
            api_base: Base URL for the API endpoint
            api_key: API key (default "EMPTY" for local servers)
            model: Model identifier for MLA model
        """
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.api_base = api_base
        
    def generate_completion(self, 
                           prompt: str, 
                           max_tokens: int = 100,
                           temperature: float = 0.7,
                           stream: bool = True) -> Dict[str, Any]:
        """
        Generate completion using MLA model.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing response text and metrics
        """
        start_time = time.time()
        
        if stream:
            # Streaming response
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            generated_text = ""
            first_token_time = None
            tokens_count = 0
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].text:
                    text = chunk.choices[0].text
                    if first_token_time is None:
                        first_token_time = time.time()
                    generated_text += text
                    tokens_count += 1
                    print(text, end="", flush=True)
            
            print()  # New line after streaming
            
            end_time = time.time()
            ttft = (first_token_time - start_time) if first_token_time else 0
            
        else:
            # Non-streaming response
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            generated_text = response.choices[0].text
            end_time = time.time()
            ttft = end_time - start_time  # For non-streaming, TTFT = total time
            tokens_count = len(generated_text.split())
        
        total_time = end_time - start_time
        
        return {
            "text": generated_text,
            "ttft_ms": ttft * 1000,
            "total_time_s": total_time,
            "tokens": tokens_count,
            "throughput": tokens_count / total_time if total_time > 0 else 0
        }
    
    def chat_completion(self, 
                        messages: list,
                        max_tokens: int = 100,
                        temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate chat completion using MLA model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing response and metrics
        """
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        generated_text = ""
        first_token_time = None
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                if first_token_time is None:
                    first_token_time = time.time()
                generated_text += text
                print(text, end="", flush=True)
        
        print()  # New line after streaming
        
        end_time = time.time()
        ttft = (first_token_time - start_time) if first_token_time else 0
        total_time = end_time - start_time
        tokens_count = len(generated_text.split())
        
        return {
            "text": generated_text,
            "ttft_ms": ttft * 1000,
            "total_time_s": total_time,
            "tokens": tokens_count,
            "throughput": tokens_count / total_time if total_time > 0 else 0
        }
    
    def test_kv_cache_reuse(self, 
                           prompt: str,
                           num_iterations: int = 3) -> list:
        """
        Test KV cache reuse by sending the same prompt multiple times.
        
        Args:
            prompt: The prompt to test
            num_iterations: Number of times to send the prompt
            
        Returns:
            List of results for each iteration
        """
        results = []
        
        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            result = self.generate_completion(prompt, max_tokens=50, temperature=0)
            results.append({
                "iteration": i + 1,
                **result
            })
            
            # Print metrics
            print(f"\nMetrics:")
            print(f"  TTFT: {result['ttft_ms']:.2f} ms")
            print(f"  Total time: {result['total_time_s']:.2f} s")
            print(f"  Throughput: {result['throughput']:.2f} tokens/s")
            
            # Small delay between iterations
            if i < num_iterations - 1:
                time.sleep(1)
        
        return results


def run_benchmarks(client: MLAInferenceClient):
    """Run various benchmarks to demonstrate MLA with LMCache."""
    
    print("\n" + "="*60)
    print("MLA Model Inference Benchmarks with LMCache")
    print("="*60)
    
    # Benchmark 1: Simple completion
    print("\n### Benchmark 1: Simple Completion ###")
    prompt = "Explain the concept of multi-head latent attention in one paragraph:"
    result = client.generate_completion(prompt, max_tokens=100)
    print(f"\nMetrics: TTFT={result['ttft_ms']:.2f}ms, "
          f"Throughput={result['throughput']:.2f} tokens/s")
    
    # Benchmark 2: KV Cache Reuse
    print("\n### Benchmark 2: KV Cache Reuse Test ###")
    cache_prompt = "What are the key advantages of using MLA over traditional multi-head attention?"
    cache_results = client.test_kv_cache_reuse(cache_prompt, num_iterations=3)
    
    # Analyze cache performance
    print("\n### Cache Performance Analysis ###")
    for res in cache_results:
        print(f"Iteration {res['iteration']}: TTFT={res['ttft_ms']:.2f}ms")
    
    if len(cache_results) >= 2:
        speedup = cache_results[0]['ttft_ms'] / cache_results[1]['ttft_ms']
        print(f"\nCache speedup (1st vs 2nd): {speedup:.2f}x")
    
    # Benchmark 3: Chat completion
    print("\n### Benchmark 3: Chat Completion ###")
    messages = [
        {"role": "system", "content": "You are an AI assistant expert in deep learning."},
        {"role": "user", "content": "What is MLA and how does it improve transformer efficiency?"}
    ]
    chat_result = client.chat_completion(messages, max_tokens=100)
    print(f"\nMetrics: TTFT={chat_result['ttft_ms']:.2f}ms, "
          f"Throughput={chat_result['throughput']:.2f} tokens/s")


def main():
    parser = argparse.ArgumentParser(
        description="MLA inference example with LMCache"
    )
    parser.add_argument(
        "--api-base", 
        type=str, 
        default="http://localhost:8000/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="MLA model to use"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt to test"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark suite"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = MLAInferenceClient(
        api_base=args.api_base,
        model=args.model
    )
    
    if args.benchmark:
        # Run benchmark suite
        run_benchmarks(client)
    elif args.prompt:
        # Run custom prompt
        print(f"\nGenerating response for custom prompt...")
        print(f"Prompt: {args.prompt}\n")
        result = client.generate_completion(
            args.prompt, 
            max_tokens=args.max_tokens
        )
        print(f"\nMetrics:")
        print(f"  TTFT: {result['ttft_ms']:.2f} ms")
        print(f"  Total time: {result['total_time_s']:.2f} s")
        print(f"  Tokens generated: {result['tokens']}")
        print(f"  Throughput: {result['throughput']:.2f} tokens/s")
    else:
        # Default demo
        print("\n=== MLA Inference Demo ===")
        prompt = "Explain how MLA improves efficiency compared to standard attention:"
        print(f"Prompt: {prompt}\n")
        result = client.generate_completion(prompt, max_tokens=75)
        print(f"\nTTFT: {result['ttft_ms']:.2f}ms, "
              f"Throughput: {result['throughput']:.2f} tokens/s")


if __name__ == "__main__":
    main()