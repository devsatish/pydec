# Inference Engine Benchmark: vLLM vs Friendli Engine

A reproducible benchmarking tool to compare inference performance between vLLM and Friendli Engine.

## Overview

This benchmark measures key performance metrics that are critical for production LLM inference:

- **Time to First Token (TTFT)**: The latency from request submission to receiving the first token
- **End-to-End Latency**: Total time to complete a request
- **Tokens Per Second (TPS)**: The rate of token generation
- **Throughput**: Overall tokens generated per unit time

## Why These Metrics?

### Time to First Token (TTFT)
TTFT is crucial for user experience in interactive applications. A lower TTFT means users see responses faster, reducing perceived latency. This metric captures the engine's efficiency in prompt processing and initial inference.

### End-to-End Latency
Total latency matters for batch processing and API SLA compliance. It includes both TTFT and token generation time.

### Tokens Per Second (TPS)
TPS measures the generation speed after the first token. Higher TPS means faster complete responses and better resource utilization.

### Throughput
Overall throughput indicates how efficiently the engine handles workloads, important for cost optimization.

## Why the Visualization Works

The benchmark generates a single comprehensive chart that shows normalized performance:

- **Normalized Comparison**: All metrics normalized with vLLM as baseline (1.0)
  - Values >1.0 indicate Friendli Engine outperforms vLLM
  - Percentage labels show exact improvement for each metric
  - All metrics normalized so "higher = better" for easy interpretation

- **Visual Features**:
  - Grouped bars for side-by-side comparison
  - Percentage improvement labels (green for better, red for worse)
  - Raw values shown below bars for reference
  - Baseline reference line at 1.0
  - Summary box showing average improvement across all metrics

This visualization:
- Enables quick comparison across different metric scales
- Highlights performance gaps with clear percentage labels
- Uses intuitive color coding (green for better, red for worse)
- Provides both normalized relative performance and raw values
- Shows a single, clear conclusion about overall performance

## Requirements

- Python 3.11+
- Both vLLM and Friendli Engine servers deployed and accessible

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python benchmark.py \
    --vllm-url http://localhost:8000 \
    --friendli-url http://localhost:8001
```

### Advanced Options

```bash
python benchmark.py \
    --vllm-url http://localhost:8000 \
    --friendli-url http://localhost:8001 \
    --model meta-llama/Llama-2-7b-chat-hf \
    --iterations 10 \
    --max-tokens 512 \
    --concurrency 4 \
    --seed 42 \
    --output results/benchmark.png
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--vllm-url` | Base URL of vLLM server | Required |
| `--friendli-url` | Base URL of Friendli Engine server | Required |
| `--model` | Model identifier for both engines | `default` |
| `--iterations` | Number of iterations per prompt | `3` |
| `--max-tokens` | Maximum tokens to generate | `256` |
| `--concurrency` | Number of concurrent requests | `1` |
| `--output` | Output path for comparison chart | `benchmark_comparison.png` |
| `--seed` | Random seed for reproducibility | `None` |
| `--no-warmup` | Skip engine warmup phase | `False` |

## Output

The benchmark produces:

1. **Console Output**: Detailed per-request metrics and summary statistics
2. **PNG Chart**: Visual comparison (`benchmark_comparison.png`)
3. **JSON Results**: Raw data for further analysis (`benchmark_comparison_results.json`)

## Example Output

```
================================================================================
BENCHMARK RESULTS SUMMARY
================================================================================

Time to First Token (Mean):
  vLLM:    0.1523 s
  Friendli: 0.0891 s
  Winner:  Friendli (41.5% better)

Tokens Per Second (Mean):
  vLLM:    45.2300 tok/s
  Friendli: 62.8400 tok/s
  Winner:  Friendli (38.9% better)
```

## Reproducibility

The benchmark includes several features to ensure reproducible results:

1. **Random Seed**: Use `--seed` to set a random seed for consistent prompt ordering
2. **Engine Warmup**: Automatic warmup phase (can be disabled with `--no-warmup`) ensures engines are in a consistent state
3. **Same Prompts**: Both engines receive identical prompts in the same order
4. **Identical Parameters**: Same model, max_tokens, temperature, and concurrency settings

For best reproducibility:

1. Use the same hardware for both engines
2. Run benchmarks during consistent system load
3. Use identical model configurations
4. Set a random seed: `--seed 42`
5. Use multiple iterations (recommended: 10+)
6. Ensure engines are warmed up (default behavior)

## License

MIT License
