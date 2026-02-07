# Reasoning LLM Gateway Server

A Python 3.11+ API server that acts as a gateway in front of `/chat/completions` APIs, providing enhanced streaming with reasoning summarization.

## Design Overview

### Architecture

```
┌─────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Client │────▶│  Gateway Server  │────▶│  Upstream LLM    │
│         │◀────│  (this server)   │◀────│  /chat/completions│
└─────────┘     └──────────────────┘     └──────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ Enhanced SSE Stream │
              │ 1. Prompt Summary   │
              │ 2. Reasoning Summary│
              │ 3. Final Output     │
              └─────────────────────┘
```

### Key Features

1. **Phased Streaming Response**
   - **Phase 1 - Prompt Summary**: Immediately streams a summary of the user's prompt (minimizes TTFT)
   - **Phase 2 - Reasoning Summary**: Summarizes the model's reasoning process
   - **Phase 3 - Final Output**: Streams the model's final response

2. **Failure Resilience**
   - Circuit breaker pattern prevents cascade failures
   - Automatic recovery after timeout
   - Graceful error handling with informative messages

3. **Multi-Model Support**
   - Configurable reasoning detection patterns per model via `ModelConfig`
   - Works with any reasoning-capable LLM that uses marker-based thinking
   - Easy to extend by adding entries to `DEFAULT_MODELS`

4. **Optimized Latency**
   - Immediate prompt summary reduces perceived TTFT
   - Streaming throughout the entire response
   - No buffering of final output

## Assumptions

### Reasoning Detection

Since the upstream API doesn't distinguish between reasoning tokens and output tokens, we use the following heuristics:

1. **Marker-based detection**: Many reasoning models use explicit markers:
   - `<think>...</think>`
   - `<reasoning>...</reasoning>`
   - Natural language markers: "Let me think", "Let's analyze"

2. **End markers**: Reasoning typically ends with:
   - `</think>`, `</reasoning>`
   - "Therefore,", "In conclusion,", "The answer is"

3. **Fallback behavior**: If no markers detected, entire output is treated as final output

These patterns are configurable per model in `ModelConfig`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
# Basic startup
python server.py

# With uvicorn (production)
uvicorn server:app --host 0.0.0.0 --port 8080 --workers 4

# With environment variables
UPSTREAM_URL=http://your-llm-host:8000 PORT=8080 python server.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UPSTREAM_URL` | Upstream LLM API URL | `http://localhost:9000` |
| `UPSTREAM_API_KEY` | API key for upstream service | None |
| `PORT` | Gateway server port | `8080` |
| `MOCK_PORT` | Mock server port (for testing) | `9000` |

### Using the Client

```bash
# Single query
python client.py --url http://localhost:8080 --prompt "Explain quantum computing"

# Interactive mode
python client.py --url http://localhost:8080 --interactive

# Health check
python client.py --url http://localhost:8080 --health

# Verbose mode (shows phase transitions)
python client.py --url http://localhost:8080 --prompt "Hello" -v
```

### API Usage

#### Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": true
  }'
```

#### List Models

```bash
curl http://localhost:8080/v1/models
```

#### Health Check

```bash
curl http://localhost:8080/health
```

## Expected Output

When making a request, you'll receive SSE events in this order:

```
event: phase
data: {"phase": "prompt_summary"}

data: {"id": "...", "choices": [{"delta": {"content": "[Prompt Summary] Processing: \"What is 2+2?\"..."}}], "phase": "prompt_summary"}

event: phase
data: {"phase": "reasoning_summary"}

data: {"id": "...", "choices": [{"delta": {"content": "[Reasoning Summary] Adding 2 and 2 together..."}}], "phase": "reasoning_summary"}

event: phase
data: {"phase": "final_output"}

data: {"id": "...", "choices": [{"delta": {"content": "The"}}], "phase": "final_output"}
data: {"id": "...", "choices": [{"delta": {"content": " answer"}}], "phase": "final_output"}
data: {"id": "...", "choices": [{"delta": {"content": " is"}}], "phase": "final_output"}
data: {"id": "...", "choices": [{"delta": {"content": " 4"}}], "phase": "final_output"}

data: {"id": "...", "choices": [{"delta": {}, "finish_reason": "stop"}]}
data: [DONE]
```

## Docker Support

Build and run:

```bash
docker build -t reasoning-gateway .

# Run gateway pointing at a host-side upstream
docker run -p 8080:8080 -e UPSTREAM_URL=http://host.docker.internal:9000 reasoning-gateway

# Or run the mock server inside the container first
docker run -p 9000:9000 reasoning-gateway python mock_server.py
```

## Testing

### Quick Start (three terminals)

A bundled mock server simulates an upstream LLM that produces `<think>…</think>` reasoning markers so you can test the full pipeline without a real model.

```bash
# Terminal 1 — mock upstream LLM (port 9000)
python mock_server.py

# Terminal 2 — gateway server (port 8080, points at mock by default)
python server.py

# Terminal 3 — client
python client.py --prompt "What is 2+2?" -v
```

The mock server streams canned responses that include reasoning markers. The gateway detects them, generates a reasoning summary, and streams the final output — all visible in the client's colour-coded output.

### Manual Testing Against a Real LLM

```bash
UPSTREAM_URL=http://your-llm-host:8000 python server.py
python client.py --prompt "Explain quantum computing" --interactive
```

## API Reference

### POST /v1/chat/completions

**Request Body:**
```json
{
  "model": "string",
  "messages": [{"role": "user", "content": "string"}],
  "max_tokens": 1024,
  "temperature": 0.7,
  "stream": true,
  "reasoning_effort": "medium"
}
```

**Response:** Server-Sent Events stream with phased content.

### GET /v1/models

Returns list of available models and their capabilities.

### GET /health

Returns server health status.

## License

MIT License
