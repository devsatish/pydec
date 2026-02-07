#!/usr/bin/env python3
"""
Inference Engine Benchmark: vLLM vs Friendli Engine

This script provides a fair, reproducible comparison between vLLM and Friendli Engine
for LLM inference performance evaluation.

Metrics measured:
- Time to First Token (TTFT): Latency until the first token is generated
- Throughput: Tokens generated per second
- End-to-End Latency: Total time for complete response
- Tokens Per Second (TPS): Output generation rate

Requirements:
- Python 3.11+
- requests, numpy, matplotlib, aiohttp

Usage:
    python benchmark.py --vllm-url http://localhost:8000 --friendli-url http://localhost:8001
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single request."""
    ttft: float  # Time to First Token (seconds)
    total_latency: float  # End-to-end latency (seconds)
    output_tokens: int  # Number of tokens generated
    tokens_per_second: float  # Output token generation rate


@dataclass
class AggregatedResults:
    """Aggregated benchmark results across multiple requests."""
    engine_name: str
    ttft_mean: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    latency_mean: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput_mean: float = 0.0
    throughput_total: float = 0.0
    tps_mean: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    raw_results: list[BenchmarkResult] = field(default_factory=list)


# Standard benchmark prompts of varying complexity
BENCHMARK_PROMPTS = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "user", "content": "Explain the concept of machine learning in one paragraph."},
    {"role": "user", "content": "Write a Python function to calculate the factorial of a number."},
    {"role": "user", "content": "Summarize the key differences between REST and GraphQL APIs."},
    {"role": "user", "content": "What are the main benefits of using containerization with Docker?"},
    {"role": "user", "content": "Explain how transformer architecture works in neural networks."},
    {"role": "user", "content": "Write a short story about a robot learning to paint."},
    {"role": "user", "content": "What are the SOLID principles in software engineering?"},
]


async def stream_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict],
    model: str = "default",
    max_tokens: int = 256,
) -> AsyncIterator[tuple[float, str, bool]]:
    """
    Stream completions from an OpenAI-compatible API endpoint.

    Yields tuples of (timestamp, token_content, is_first_token).
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    headers = {"Content-Type": "application/json"}
    start_time = time.perf_counter()
    first_token = True

    async with session.post(url, json=payload, headers=headers) as response:
        response.raise_for_status()
        async for line in response.content:
            line = line.decode("utf-8").strip()
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                        content = chunk["choices"][0]["delta"]["content"]
                        yield (time.perf_counter() - start_time, content, first_token)
                        first_token = False
                except json.JSONDecodeError:
                    continue


async def benchmark_single_request(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict],
    model: str = "default",
    max_tokens: int = 256,
) -> BenchmarkResult | None:
    """Run a single benchmark request and return timing metrics."""
    try:
        start_time = time.perf_counter()
        ttft = None
        tokens = []

        async for elapsed, content, is_first in stream_completion(
            session, base_url, messages, model, max_tokens
        ):
            if is_first:
                ttft = elapsed
            tokens.append(content)

        total_latency = time.perf_counter() - start_time
        output_tokens = len(tokens)

        if ttft is None or output_tokens == 0:
            return None

        # Calculate tokens per second (excluding TTFT)
        generation_time = total_latency - ttft
        tps = output_tokens / generation_time if generation_time > 0 else 0

        return BenchmarkResult(
            ttft=ttft,
            total_latency=total_latency,
            output_tokens=output_tokens,
            tokens_per_second=tps,
        )
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


async def warmup_engine(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str = "default",
    max_tokens: int = 50,
    num_warmup_requests: int = 2,
) -> None:
    """
    Warm up the inference engine with a few requests to ensure consistent performance.
    
    Args:
        session: HTTP session to use
        base_url: Base URL of the inference API
        model: Model identifier
        max_tokens: Maximum tokens for warmup requests
        num_warmup_requests: Number of warmup requests to send
    """
    print(f"  Warming up engine...")
    warmup_prompt = {"role": "user", "content": "Hello"}
    
    for i in range(num_warmup_requests):
        try:
            async for _ in stream_completion(session, base_url, [warmup_prompt], model, max_tokens):
                pass
        except Exception:
            pass  # Ignore warmup errors


async def run_benchmark(
    base_url: str,
    engine_name: str,
    model: str = "default",
    num_iterations: int = 3,
    max_tokens: int = 256,
    concurrency: int = 1,
    warmup: bool = True,
) -> AggregatedResults:
    """
    Run the full benchmark suite against an inference engine.

    When concurrency=1, requests run sequentially (measuring per-request latency).
    When concurrency>1, requests are dispatched in concurrent batches using an
    asyncio.Semaphore, measuring how the engine performs under parallel load.

    Args:
        base_url: Base URL of the inference API
        engine_name: Name of the engine for reporting
        model: Model identifier
        num_iterations: Number of times to run each prompt
        max_tokens: Maximum tokens to generate per request
        concurrency: Number of concurrent requests
        warmup: Whether to warm up the engine before benchmarking

    Returns:
        Aggregated benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine_name}")
    print(f"URL: {base_url}")
    print(f"Iterations per prompt: {num_iterations}")
    print(f"Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    results: list[BenchmarkResult] = []
    failed_count = 0

    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warm up the engine before benchmarking
        if warmup:
            await warmup_engine(session, base_url, model, max_tokens=50, num_warmup_requests=2)
            print()  # Empty line after warmup

        if concurrency == 1:
            # Sequential mode: one request at a time for clean per-request metrics
            for iteration in range(num_iterations):
                print(f"Iteration {iteration + 1}/{num_iterations}")

                for i, prompt in enumerate(BENCHMARK_PROMPTS):
                    result = await benchmark_single_request(
                        session, base_url, [prompt], model, max_tokens
                    )
                    if result:
                        results.append(result)
                        print(f"  Prompt {i+1}: TTFT={result.ttft:.3f}s, "
                              f"Latency={result.total_latency:.3f}s, "
                              f"TPS={result.tokens_per_second:.1f}")
                    else:
                        failed_count += 1
                        print(f"  Prompt {i+1}: FAILED")
        else:
            # Concurrent mode: dispatch requests in parallel, bounded by semaphore
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded_request(prompt_idx: int, prompt: dict) -> tuple[int, BenchmarkResult | None]:
                async with semaphore:
                    result = await benchmark_single_request(
                        session, base_url, [prompt], model, max_tokens
                    )
                    return prompt_idx, result

            for iteration in range(num_iterations):
                print(f"Iteration {iteration + 1}/{num_iterations} (concurrency={concurrency})")

                tasks = [
                    bounded_request(i, prompt)
                    for i, prompt in enumerate(BENCHMARK_PROMPTS)
                ]
                batch_results = await asyncio.gather(*tasks)

                for prompt_idx, result in sorted(batch_results, key=lambda x: x[0]):
                    if result:
                        results.append(result)
                        print(f"  Prompt {prompt_idx+1}: TTFT={result.ttft:.3f}s, "
                              f"Latency={result.total_latency:.3f}s, "
                              f"TPS={result.tokens_per_second:.1f}")
                    else:
                        failed_count += 1
                        print(f"  Prompt {prompt_idx+1}: FAILED")

    # Aggregate results
    aggregated = AggregatedResults(
        engine_name=engine_name,
        total_requests=len(BENCHMARK_PROMPTS) * num_iterations,
        successful_requests=len(results),
        failed_requests=failed_count,
        raw_results=results,
    )

    if results:
        ttfts = [r.ttft for r in results]
        latencies = [r.total_latency for r in results]
        tps_values = [r.tokens_per_second for r in results]

        aggregated.ttft_mean = statistics.mean(ttfts)
        aggregated.ttft_p50 = statistics.median(ttfts)
        aggregated.ttft_p95 = np.percentile(ttfts, 95)
        aggregated.ttft_p99 = np.percentile(ttfts, 99)

        aggregated.latency_mean = statistics.mean(latencies)
        aggregated.latency_p50 = statistics.median(latencies)
        aggregated.latency_p95 = np.percentile(latencies, 95)
        aggregated.latency_p99 = np.percentile(latencies, 99)

        aggregated.tps_mean = statistics.mean(tps_values)

        total_tokens = sum(r.output_tokens for r in results)
        total_time = sum(r.total_latency for r in results)
        aggregated.throughput_total = total_tokens / total_time if total_time > 0 else 0

    return aggregated


def print_results(vllm_results: AggregatedResults, friendli_results: AggregatedResults) -> None:
    """Print a formatted comparison of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    def print_metric(name: str, vllm_val: float, friendli_val: float, unit: str = "s", lower_better: bool = True):
        if lower_better:
            improvement = ((vllm_val - friendli_val) / vllm_val * 100) if vllm_val > 0 else 0
            better = "Friendli" if friendli_val < vllm_val else "vLLM"
        else:
            improvement = ((friendli_val - vllm_val) / vllm_val * 100) if vllm_val > 0 else 0
            better = "Friendli" if friendli_val > vllm_val else "vLLM"

        print(f"\n{name}:")
        print(f"  vLLM:    {vllm_val:.4f} {unit}")
        print(f"  Friendli: {friendli_val:.4f} {unit}")
        print(f"  Winner:  {better} ({abs(improvement):.1f}% {'better' if improvement > 0 else 'difference'})")

    print_metric("Time to First Token (Mean)", vllm_results.ttft_mean, friendli_results.ttft_mean)
    print_metric("Time to First Token (P95)", vllm_results.ttft_p95, friendli_results.ttft_p95)
    print_metric("End-to-End Latency (Mean)", vllm_results.latency_mean, friendli_results.latency_mean)
    print_metric("End-to-End Latency (P95)", vllm_results.latency_p95, friendli_results.latency_p95)
    print_metric("Tokens Per Second (Mean)", vllm_results.tps_mean, friendli_results.tps_mean, "tok/s", lower_better=False)
    print_metric("Overall Throughput", vllm_results.throughput_total, friendli_results.throughput_total, "tok/s", lower_better=False)

    print(f"\n{'='*80}")
    print("Request Statistics:")
    print(f"  vLLM:    {vllm_results.successful_requests}/{vllm_results.total_requests} successful")
    print(f"  Friendli: {friendli_results.successful_requests}/{friendli_results.total_requests} successful")
    print("=" * 80)


def generate_comparison_chart(
    vllm_results: AggregatedResults,
    friendli_results: AggregatedResults,
    output_path: str = "benchmark_comparison.png"
) -> None:
    """
    Generate a single comprehensive comparison chart showing performance differences.

    The chart uses normalized values (vLLM as baseline = 1.0) to show relative
    performance across multiple metrics, making it easy to identify which engine
    performs better in each category. All metrics are normalized so higher = better.
    """
    # Metrics to display (label, vllm_value, friendli_value, lower_is_better)
    metrics = [
        ("TTFT\n(Mean)", vllm_results.ttft_mean, friendli_results.ttft_mean, True),
        ("TTFT\n(P95)", vllm_results.ttft_p95, friendli_results.ttft_p95, True),
        ("Latency\n(Mean)", vllm_results.latency_mean, friendli_results.latency_mean, True),
        ("Latency\n(P95)", vllm_results.latency_p95, friendli_results.latency_p95, True),
        ("TPS\n(Mean)", vllm_results.tps_mean, friendli_results.tps_mean, False),
        ("Throughput\n(tok/s)", vllm_results.throughput_total, friendli_results.throughput_total, False),
    ]

    labels = [m[0] for m in metrics]
    vllm_values = [m[1] for m in metrics]
    friendli_values = [m[2] for m in metrics]
    lower_is_better = [m[3] for m in metrics]

    # Normalize values for comparison (vLLM as baseline = 1.0)
    vllm_normalized = []
    friendli_normalized = []
    improvements = []

    for i, (vllm_val, friendli_val, lower_better) in enumerate(zip(vllm_values, friendli_values, lower_is_better)):
        if vllm_val == 0:
            vllm_normalized.append(1.0)
            friendli_normalized.append(1.0)
            improvements.append(0.0)
        else:
            vllm_normalized.append(1.0)
            ratio = friendli_val / vllm_val
            # For lower-is-better metrics, invert so higher bar = better
            if lower_better:
                normalized = 1.0 / ratio if ratio > 0 else 1.0
                friendli_normalized.append(normalized)
                improvements.append((normalized - 1.0) * 100)
            else:
                friendli_normalized.append(ratio)
                improvements.append((ratio - 1.0) * 100)

    # Create single figure
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Inference Engine Performance Comparison: vLLM vs Friendli Engine",
                 fontsize=16, fontweight="bold", y=0.98)

    x = np.arange(len(labels))
    width = 0.35

    colors_vllm = "#4A90D9"
    colors_friendli = "#50C878"

    # Create grouped bars
    bars1 = ax.bar(x - width/2, vllm_normalized, width, label="vLLM (baseline = 1.0)", 
                   color=colors_vllm, edgecolor="black", linewidth=0.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, friendli_normalized, width, label="Friendli Engine", 
                   color=colors_friendli, edgecolor="black", linewidth=0.5, alpha=0.8)

    # Add baseline reference line
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="vLLM Baseline")

    # Add percentage improvement labels on Friendli bars
    for i, (bar, norm_val, improvement) in enumerate(zip(bars2, friendli_normalized, improvements)):
        color = "green" if improvement > 0 else "red"
        sign = "+" if improvement > 0 else ""
        # Position label above bar
        height = bar.get_height()
        ax.annotate(f"{sign}{improvement:.1f}%",
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                   fontsize=10, fontweight="bold", color=color)
        # Add raw value annotation below
        ax.annotate(f"{friendli_values[i]:.2f}",
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, -15), textcoords="offset points", ha="center", va="top",
                   fontsize=8, color=colors_friendli, style="italic")

    # Add raw value labels on vLLM bars
    for bar, val in zip(bars1, vllm_values):
        ax.annotate(f"{val:.2f}",
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, -15), textcoords="offset points", ha="center", va="top",
                   fontsize=8, color=colors_vllm, style="italic")

    ax.set_xlabel("Performance Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Performance (vLLM = 1.0 baseline, higher = better)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    # Add summary text box
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    summary_text = f"Average Improvement: {avg_improvement:+.1f}%"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")
    plt.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference performance: vLLM vs Friendli Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with default settings
  python benchmark.py --vllm-url http://localhost:8000 --friendli-url http://localhost:8001

  # Extended benchmark with more iterations
  python benchmark.py --vllm-url http://localhost:8000 --friendli-url http://localhost:8001 \\
                      --iterations 10 --max-tokens 512

  # Specify model name and seed for reproducibility
  python benchmark.py --vllm-url http://localhost:8000 --friendli-url http://localhost:8001 \\
                      --model meta-llama/Llama-2-7b-chat-hf --seed 42

  # Disable warmup (if engines are already warmed up)
  python benchmark.py --vllm-url http://localhost:8000 --friendli-url http://localhost:8001 \\
                      --no-warmup
        """
    )
    parser.add_argument("--vllm-url", required=True, help="Base URL of the vLLM server (e.g., http://localhost:8000)")
    parser.add_argument("--friendli-url", required=True, help="Base URL of the Friendli Engine server")
    parser.add_argument("--model", default="default", help="Model identifier to use for both engines")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per prompt (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate (default: 256)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests (default: 1)")
    parser.add_argument("--output", default="benchmark_comparison.png", help="Output path for the comparison chart")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip engine warmup phase")

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    print("=" * 80)
    print("INFERENCE ENGINE BENCHMARK")
    print("vLLM vs Friendli Engine Performance Comparison")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  vLLM URL:      {args.vllm_url}")
    print(f"  Friendli URL:  {args.friendli_url}")
    print(f"  Model:         {args.model}")
    print(f"  Iterations:    {args.iterations}")
    print(f"  Max Tokens:    {args.max_tokens}")
    print(f"  Concurrency:   {args.concurrency}")
    print(f"  Random Seed:   {args.seed if args.seed is not None else 'None (not set)'}")
    print(f"  Warmup:        {'Disabled' if args.no_warmup else 'Enabled'}")
    print(f"  Output Chart:  {args.output}")

    # Run benchmarks
    vllm_results = await run_benchmark(
        base_url=args.vllm_url,
        engine_name="vLLM",
        model=args.model,
        num_iterations=args.iterations,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        warmup=not args.no_warmup,
    )

    friendli_results = await run_benchmark(
        base_url=args.friendli_url,
        engine_name="Friendli Engine",
        model=args.model,
        num_iterations=args.iterations,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        warmup=not args.no_warmup,
    )

    # Print results comparison
    print_results(vllm_results, friendli_results)

    # Generate visualization
    generate_comparison_chart(vllm_results, friendli_results, args.output)

    # Save raw results to JSON
    results_json = {
        "configuration": {
            "vllm_url": args.vllm_url,
            "friendli_url": args.friendli_url,
            "model": args.model,
            "iterations": args.iterations,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "warmup": not args.no_warmup,
        },
        "vllm": {
            "ttft_mean": vllm_results.ttft_mean,
            "ttft_p50": vllm_results.ttft_p50,
            "ttft_p95": vllm_results.ttft_p95,
            "ttft_p99": vllm_results.ttft_p99,
            "latency_mean": vllm_results.latency_mean,
            "latency_p50": vllm_results.latency_p50,
            "latency_p95": vllm_results.latency_p95,
            "latency_p99": vllm_results.latency_p99,
            "tps_mean": vllm_results.tps_mean,
            "throughput_total": vllm_results.throughput_total,
            "successful_requests": vllm_results.successful_requests,
            "failed_requests": vllm_results.failed_requests,
        },
        "friendli": {
            "ttft_mean": friendli_results.ttft_mean,
            "ttft_p50": friendli_results.ttft_p50,
            "ttft_p95": friendli_results.ttft_p95,
            "ttft_p99": friendli_results.ttft_p99,
            "latency_mean": friendli_results.latency_mean,
            "latency_p50": friendli_results.latency_p50,
            "latency_p95": friendli_results.latency_p95,
            "latency_p99": friendli_results.latency_p99,
            "tps_mean": friendli_results.tps_mean,
            "throughput_total": friendli_results.throughput_total,
            "successful_requests": friendli_results.successful_requests,
            "failed_requests": friendli_results.failed_requests,
        },
    }

    json_output = args.output.rsplit(".", 1)[0] + "_results.json"
    with open(json_output, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Raw results saved to: {json_output}")


if __name__ == "__main__":
    asyncio.run(main())
