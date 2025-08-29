#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive test script to demonstrate LMCache and MLA benefits.
This script tests various scenarios where cache is most effective.
"""

import time
import json
import argparse
import statistics
from typing import List, Dict, Any
from openai import OpenAI
import requests


class ColoredOutput:
    """Helper class for colored terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @classmethod
    def header(cls, text):
        return f"{cls.HEADER}{cls.BOLD}{text}{cls.ENDC}"
    
    @classmethod
    def success(cls, text):
        return f"{cls.OKGREEN}{text}{cls.ENDC}"
    
    @classmethod
    def warning(cls, text):
        return f"{cls.WARNING}{text}{cls.ENDC}"
    
    @classmethod
    def fail(cls, text):
        return f"{cls.FAIL}{text}{cls.ENDC}"
    
    @classmethod
    def info(cls, text):
        return f"{cls.OKCYAN}{text}{cls.ENDC}"


def check_model_info(api_base: str, model: str) -> Dict[str, Any]:
    """Check model information and MLA support."""
    try:
        response = requests.get(f"{api_base}/models")
        models = response.json()
        
        # Check if model is loaded
        model_info = {}
        for m in models.get("data", []):
            if model in m.get("id", ""):
                model_info = m
                break
        
        # Detect MLA based on model name
        is_mla = False
        mla_indicators = ["deepseek", "DeepSeek", "mla", "MLA"]
        for indicator in mla_indicators:
            if indicator in model:
                is_mla = True
                break
        
        return {
            "model_loaded": bool(model_info),
            "model_name": model,
            "is_mla": is_mla,
            "model_info": model_info
        }
    except Exception as e:
        return {
            "model_loaded": False,
            "error": str(e)
        }


def test_exact_prompt_reuse(client: OpenAI, model: str, iterations: int = 5) -> Dict[str, Any]:
    """Test exact prompt reuse - best case for cache."""
    print(ColoredOutput.header("\n🔄 Test 1: Exact Prompt Reuse"))
    print("="*60)
    print("Testing cache performance with identical prompts...")
    
    prompt = """Explain in detail: What is the transformer architecture in deep learning? 
    Include information about self-attention, multi-head attention, and positional encoding."""
    
    times = []
    for i in range(iterations):
        start = time.time()
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=100,
            temperature=0,
            seed=42
        )
        elapsed = time.time() - start
        times.append(elapsed)
        
        if i == 0:
            print(f"\n  Request 1 (cold): {elapsed:.3f}s ❄️")
            first_response = response.choices[0].text[:50]
            print(f"  Response preview: {first_response}...")
        else:
            speedup = times[0] / elapsed
            emoji = "🚀" if speedup > 2 else "✅" if speedup > 1.5 else "⚠️"
            print(f"  Request {i+1} (warm): {elapsed:.3f}s ({speedup:.2f}x) {emoji}")
    
    avg_warm = statistics.mean(times[1:]) if len(times) > 1 else times[0]
    speedup = times[0] / avg_warm if avg_warm > 0 else 1
    
    print(f"\n  {ColoredOutput.success(f'Average speedup: {speedup:.2f}x')}")
    
    return {
        "test": "exact_prompt_reuse",
        "cold_time": times[0],
        "warm_times": times[1:],
        "average_speedup": speedup
    }


def test_conversation_history(client: OpenAI, model: str) -> Dict[str, Any]:
    """Test conversation with accumulating history - common chat scenario."""
    print(ColoredOutput.header("\n💬 Test 2: Conversation History (Accumulating Context)"))
    print("="*60)
    print("Testing cache with growing conversation history...")
    
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant expert in technology."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    responses = []
    times = []
    
    # First exchange
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        max_tokens=50,
        temperature=0
    )
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"\n  Turn 1 (initial): {elapsed:.3f}s")
    
    # Add assistant response
    conversation.append({"role": "assistant", "content": response.choices[0].message.content})
    
    # Continue conversation
    follow_ups = [
        "What about deep learning?",
        "How does it relate to neural networks?",
        "Can you give a simple example?"
    ]
    
    for i, question in enumerate(follow_ups, 2):
        conversation.append({"role": "user", "content": question})
        
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            max_tokens=50,
            temperature=0
        )
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Cache should help as conversation history is reused
        cache_benefit = "✅ Cache hit" if elapsed < times[0] * 0.8 else "⚠️ Possible miss"
        print(f"  Turn {i} (history reuse): {elapsed:.3f}s - {cache_benefit}")
        
        conversation.append({"role": "assistant", "content": response.choices[0].message.content})
    
    avg_time = statistics.mean(times[1:]) if len(times) > 1 else times[0]
    improvement = ((times[0] - avg_time) / times[0] * 100) if times[0] > 0 else 0
    
    print(f"\n  {ColoredOutput.success(f'Performance improvement: {improvement:.1f}%')}")
    
    return {
        "test": "conversation_history",
        "turn_times": times,
        "improvement_percent": improvement
    }


def test_document_qa(client: OpenAI, model: str) -> Dict[str, Any]:
    """Test document Q&A - common RAG scenario."""
    print(ColoredOutput.header("\n📄 Test 3: Document Q&A (Shared Context)"))
    print("="*60)
    print("Testing cache with document-based questions...")
    
    document = """
    LMCache is an advanced caching system designed specifically for Large Language Model inference.
    It provides several key benefits:
    1. KV Cache Sharing: Enables reuse of key-value pairs across multiple inference requests
    2. Memory Efficiency: Reduces memory consumption by up to 70% in typical scenarios
    3. Latency Reduction: Decreases time-to-first-token by 2-5x for cached content
    4. Storage Flexibility: Supports multiple backends including CPU, GPU, disk, and Redis
    5. Seamless Integration: Works with popular frameworks like vLLM and SGLang
    
    The system is particularly effective for applications with:
    - Repeated queries on the same documents
    - Chat applications with conversation history
    - Code completion with common prefixes
    - Batch processing of related requests
    """
    
    questions = [
        "What is LMCache?",
        "How much memory can it save?",
        "What backends are supported?",
        "What applications benefit most?",
        "How much latency reduction is possible?"
    ]
    
    times = []
    for i, question in enumerate(questions, 1):
        prompt = f"{document}\n\nQuestion: {question}\nAnswer:"
        
        start = time.time()
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=30,
            temperature=0
        )
        elapsed = time.time() - start
        times.append(elapsed)
        
        answer = response.choices[0].text.strip()[:50]
        
        if i == 1:
            print(f"\n  Q1 (cold): {elapsed:.3f}s")
        else:
            speedup = times[0] / elapsed
            status = "🚀" if speedup > 1.5 else "✅" if speedup > 1.2 else "⚠️"
            print(f"  Q{i} (warm): {elapsed:.3f}s ({speedup:.2f}x) {status}")
        print(f"     → {answer}...")
    
    avg_warm = statistics.mean(times[1:]) if len(times) > 1 else times[0]
    speedup = times[0] / avg_warm if avg_warm > 0 else 1
    
    print(f"\n  {ColoredOutput.success(f'Document reuse speedup: {speedup:.2f}x')}")
    
    return {
        "test": "document_qa",
        "first_query_time": times[0],
        "subsequent_times": times[1:],
        "speedup": speedup
    }


def test_template_completion(client: OpenAI, model: str) -> Dict[str, Any]:
    """Test template/code completion - common coding scenario."""
    print(ColoredOutput.header("\n🔧 Test 4: Template Completion (Prefix Caching)"))
    print("="*60)
    print("Testing cache with template-based completions...")
    
    template = """def process_data(data):
    '''Process input data and return results.
    
    Args:
        data: Input data to process
        
    Returns:
        Processed results
    '''
    # Implementation:"""
    
    completions = [
        "return sorted(data)",
        "return [x * 2 for x in data]",
        "return sum(data) / len(data)",
        "return max(data) - min(data)"
    ]
    
    times = []
    for i, expected in enumerate(completions, 1):
        prompt = f"{template}\n    # TODO: {expected[:20]}..."
        
        start = time.time()
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=20,
            temperature=0
        )
        elapsed = time.time() - start
        times.append(elapsed)
        
        if i == 1:
            print(f"\n  Template 1 (cold): {elapsed:.3f}s")
        else:
            speedup = times[0] / elapsed
            benefit = "🚀 Great!" if speedup > 1.5 else "✅ Good" if speedup > 1.2 else "⚠️ Limited"
            print(f"  Template {i} (warm): {elapsed:.3f}s ({speedup:.2f}x) {benefit}")
    
    avg_speedup = times[0] / statistics.mean(times[1:]) if len(times) > 1 else 1
    
    print(f"\n  {ColoredOutput.success(f'Template caching speedup: {avg_speedup:.2f}x')}")
    
    return {
        "test": "template_completion",
        "times": times,
        "average_speedup": avg_speedup
    }


def analyze_mla_benefits(model_info: Dict[str, Any], results: List[Dict[str, Any]]):
    """Analyze and explain MLA benefits based on test results."""
    print(ColoredOutput.header("\n🔍 MLA (Multi-head Latent Attention) Analysis"))
    print("="*60)
    
    if model_info.get("is_mla"):
        print(ColoredOutput.success("✅ MLA Model Detected!"))
        print(f"Model: {model_info.get('model_name')}")
        
        print("\n📊 MLA Benefits in Your Tests:")
        
        # Calculate average speedup across all tests
        total_speedup = 0
        test_count = 0
        for result in results:
            if "speedup" in result or "average_speedup" in result:
                speedup = result.get("speedup", result.get("average_speedup", 1))
                total_speedup += speedup
                test_count += 1
        
        avg_speedup = total_speedup / test_count if test_count > 0 else 1
        
        print(f"\n1. **Cache Performance**: {avg_speedup:.2f}x average speedup")
        print("   MLA compresses KV pairs, making cache more efficient")
        
        print("\n2. **Memory Efficiency**:")
        print("   - Standard Attention: O(n × d × h) memory for KV cache")
        print("   - MLA: O(n × c) where c << d × h")
        print("   - Typical reduction: 60-80% less memory")
        
        print("\n3. **Throughput Benefits**:")
        print("   - More requests fit in memory due to compression")
        print("   - Faster cache lookups with smaller data")
        print("   - Better GPU utilization")
        
        print("\n4. **Your Specific Benefits**:")
        for result in results:
            test_name = result.get("test", "unknown")
            if test_name == "exact_prompt_reuse":
                speedup = result.get("average_speedup", 1)
                if speedup > 2:
                    print(f"   ✅ Excellent cache reuse ({speedup:.2f}x) for repeated queries")
                elif speedup > 1.5:
                    print(f"   ✅ Good cache reuse ({speedup:.2f}x) for repeated queries")
                else:
                    print(f"   ⚠️ Limited cache benefit ({speedup:.2f}x) - check configuration")
            
            elif test_name == "conversation_history":
                improvement = result.get("improvement_percent", 0)
                if improvement > 20:
                    print(f"   ✅ Great for chat ({improvement:.1f}% faster with history)")
                elif improvement > 10:
                    print(f"   ✅ Good for chat ({improvement:.1f}% faster with history)")
                else:
                    print(f"   ⚠️ Limited chat benefit ({improvement:.1f}%)")
            
            elif test_name == "document_qa":
                speedup = result.get("speedup", 1)
                if speedup > 1.5:
                    print(f"   ✅ Excellent for RAG ({speedup:.2f}x on shared docs)")
                elif speedup > 1.2:
                    print(f"   ✅ Good for RAG ({speedup:.2f}x on shared docs)")
                else:
                    print(f"   ⚠️ Limited RAG benefit ({speedup:.2f}x)")
    else:
        print(ColoredOutput.warning("⚠️ Non-MLA Model"))
        print(f"Model: {model_info.get('model_name')}")
        print("\nThis model doesn't use MLA, so you're missing out on:")
        print("- 60-80% memory reduction")
        print("- Better cache compression")
        print("- Higher throughput potential")
        print("\nConsider using MLA models like:")
        print("- deepseek-ai/DeepSeek-V2-Lite")
        print("- deepseek-ai/DeepSeek-V2-Chat")


def print_summary(results: List[Dict[str, Any]]):
    """Print final summary and recommendations."""
    print(ColoredOutput.header("\n📈 Final Summary"))
    print("="*60)
    
    # Calculate overall statistics
    best_speedup = 0
    best_test = ""
    
    for result in results:
        speedup = result.get("speedup", result.get("average_speedup", 1))
        if speedup > best_speedup:
            best_speedup = speedup
            best_test = result.get("test", "unknown")
    
    print(f"\n🏆 Best Performance: {best_test.replace('_', ' ').title()}")
    print(f"   Achieved {best_speedup:.2f}x speedup")
    
    print("\n💡 Optimization Tips:")
    if best_speedup > 2:
        print("   ✅ Cache is working excellently!")
        print("   - Continue using for production")
        print("   - Consider increasing cache size for even better performance")
    elif best_speedup > 1.5:
        print("   ✅ Cache is providing good benefits")
        print("   - Try longer contexts for better results")
        print("   - Ensure GPU memory is available for caching")
    else:
        print("   ⚠️ Cache benefits are limited")
        print("   - Check server logs for LMCache initialization")
        print("   - Verify configuration is loaded correctly")
        print("   - Consider using disaggregated setup for better performance")
    
    print("\n🎯 Best Use Cases for Your Setup:")
    for result in results:
        speedup = result.get("speedup", result.get("average_speedup", 1))
        test = result.get("test", "unknown")
        if speedup > 1.5:
            use_cases = {
                "exact_prompt_reuse": "Batch processing, API with repeated queries",
                "conversation_history": "Chatbots, interactive assistants",
                "document_qa": "RAG systems, document analysis",
                "template_completion": "Code completion, template filling"
            }
            print(f"   • {use_cases.get(test, test)}: {speedup:.2f}x speedup")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive LMCache and MLA testing"
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
        "--quick",
        action="store_true",
        help="Run quick tests only"
    )
    
    args = parser.parse_args()
    
    print(ColoredOutput.header("🚀 LMCache & MLA Comprehensive Test Suite"))
    print("="*60)
    
    # Check model and MLA status
    print("\n📋 Checking model configuration...")
    model_info = check_model_info(args.api_base, args.model)
    
    if not model_info.get("model_loaded"):
        print(ColoredOutput.fail("❌ Cannot connect to model server!"))
        print("Please start the server with: ./launch_server.sh")
        return
    
    print(ColoredOutput.success(f"✅ Model loaded: {args.model}"))
    if model_info.get("is_mla"):
        print(ColoredOutput.success("✅ MLA support detected"))
    else:
        print(ColoredOutput.warning("⚠️ Non-MLA model (limited compression benefits)"))
    
    # Initialize client
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)
    
    # Run tests
    results = []
    
    # Test 1: Exact prompt reuse
    results.append(test_exact_prompt_reuse(client, args.model, 3 if args.quick else 5))
    
    # Test 2: Conversation history
    results.append(test_conversation_history(client, args.model))
    
    if not args.quick:
        # Test 3: Document Q&A
        results.append(test_document_qa(client, args.model))
        
        # Test 4: Template completion
        results.append(test_template_completion(client, args.model))
    
    # Analyze MLA benefits
    analyze_mla_benefits(model_info, results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*60)
    print(ColoredOutput.success("✅ Testing complete!"))


if __name__ == "__main__":
    main()