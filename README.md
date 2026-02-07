# LLM Infrastructure Tasks

A collection of solutions for LLM inference infrastructure challenges, covering gateway server design, inference engine benchmarking, and model debugging.

## Project Structure

```
tasks/
├── model/                    # Task 1: Reasoning LLM Gateway Server
│   ├── server.py             # FastAPI gateway with phased streaming
│   ├── client.py             # CLI client for the gateway
│   ├── mock_server.py        # Mock upstream LLM for local testing
│   ├── validate_dict.py      # Utility decorator for dict validation
│   ├── Dockerfile            # Container support
│   ├── Q2_ANALYSIS.md        # Analysis of broken model & reasoning_effort
│   └── requirements.txt
├── inference/                # Task 2: Inference Engine Benchmark
│   ├── benchmark.py          # vLLM vs Friendli Engine comparison tool
│   ├── mock_server.py        # Mock inference servers for testing
│   ├── validate_dict.py      # Utility decorator for dict validation
│   └── requirements.txt
├── fixed-model-repo/         # Fixed model config with corrections
├── test_broken_model.py      # Validates the fixed model configuration
├── test_reasoning_effort.py  # Tests for reasoning_effort parameter
```

## Task 1: Reasoning LLM Gateway Server

A Python 3.11+ API server that acts as a gateway in front of `/chat/completions` APIs, providing enhanced streaming with reasoning summarization.

### Highlights

- **Phased streaming** -- responses are delivered in three stages: Prompt Summary, Reasoning Summary, and Final Output
- **Reduced perceived TTFT** by immediately streaming a prompt summary while waiting for upstream inference
- **Circuit breaker pattern** for failure resilience with automatic recovery
- **Multi-model support** with configurable reasoning detection patterns (e.g. `<think>...</think>`)
- **Docker-ready** with health checks

### Quick Start

```bash
cd model
pip install -r requirements.txt

# Terminal 1: mock upstream LLM
python mock_server.py

# Terminal 2: gateway server
python server.py

# Terminal 3: send a request
python client.py --prompt "What is 2+2?" -v
```

See [`model/README.md`](model/README.md) for full API reference and configuration options.

---

## Task 2: Inference Engine Benchmark (vLLM vs Friendli Engine)

A reproducible benchmarking tool that compares inference performance between vLLM and Friendli Engine across key production metrics.

### Metrics

| Metric | Why It Matters |
|--------|---------------|
| **Time to First Token (TTFT)** | User-perceived latency in interactive apps |
| **End-to-End Latency** | Batch processing and SLA compliance |
| **Tokens Per Second (TPS)** | Generation speed and resource utilization |
| **Throughput** | Cost efficiency at scale |

### Quick Start

```bash
cd inference
pip install -r requirements.txt

python benchmark.py \
    --vllm-url http://localhost:8000 \
    --friendli-url http://localhost:8001 \
    --iterations 10 \
    --seed 42
```

Outputs a normalized comparison chart (`benchmark_comparison.png`) and raw JSON results.

See [`inference/README.md`](inference/README.md) for advanced options and reproducibility guidance.

---

## Task 3: Debugging a Broken LLM

Investigation and fix of a broken HuggingFace model configuration that prevented inference via a `/chat/completions` API, plus analysis of why the `reasoning_effort` parameter has no effect.

### Problem a: Broken Model Config

The original model in `broken-model-original/` contained configuration errors (mismatched vocab size, incorrect token IDs, missing chat template, etc.). The corrected configuration lives in `fixed-model-repo/`.

**Validate the fix:**

```bash
python test_broken_model.py
```

### Problem b: `reasoning_effort` Parameter

Analysis of why the parameter has no observable effect and what changes are needed (model, engine, and API layer) to make it functional. Full write-up in [`model/Q2_ANALYSIS.md`](model/Q2_ANALYSIS.md).

**Run parameter tests:**

```bash
python test_reasoning_effort.py
```

---

## Requirements

- Python 3.11+
- Task-specific dependencies listed in each subdirectory's `requirements.txt`

### Install All Dependencies

```bash
pip install -r model/requirements.txt -r inference/requirements.txt
```

