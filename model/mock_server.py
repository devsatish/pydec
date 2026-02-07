#!/usr/bin/env python3
"""
Mock upstream LLM server for testing the Reasoning Gateway.

Simulates an OpenAI-compatible /v1/chat/completions endpoint that streams
responses containing <think>...</think> reasoning markers, mimicking the
behaviour of reasoning-capable LLMs (e.g. DeepSeek-R1, Qwen-QwQ).

The upstream API "does not distinguish between reasoning tokens and output
tokens in its stream" — this mock faithfully reproduces that: <think> markers
are simply part of the streamed text.

Usage:
    # Terminal 1 — start the mock upstream (port 9000)
    python mock_server.py

    # Terminal 2 — start the gateway (port 8080, points at mock by default)
    python server.py

    # Terminal 3 — run the client
    python client.py --prompt "What is 2+2?"
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Mock Upstream LLM Server")


# ---------------------------------------------------------------------------
# Canned responses — each has a thinking block and a final answer.
# The gateway's ReasoningDetector should split these apart.
# ---------------------------------------------------------------------------

CANNED_RESPONSES = [
    {
        "thinking": (
            "Let me work through this step by step. "
            "First, I need to understand what is being asked. "
            "The question requires analysing the key components carefully. "
            "I can identify the main factors involved here. "
            "Considering each factor leads to a clear pattern. "
            "The logical conclusion follows from these observations."
        ),
        "output": (
            "Based on careful analysis, here is the answer to your question. "
            "The key insight is that the problem breaks down into fundamental "
            "components, each of which leads to a clear conclusion when "
            "examined systematically. This approach provides a robust and "
            "well-reasoned answer."
        ),
    },
    {
        "thinking": (
            "I should consider multiple perspectives here. "
            "The first approach involves looking at this mathematically. "
            "However, there are also practical considerations to weigh. "
            "Comparing both approaches helps identify the strongest path forward. "
            "After careful evaluation, the mathematical approach yields a more "
            "rigorous answer because it accounts for edge cases."
        ),
        "output": (
            "The answer requires balancing theoretical and practical perspectives. "
            "The most rigorous analysis shows that the solution follows directly "
            "from the underlying principles. This gives us a confident and "
            "well-supported result that holds under scrutiny."
        ),
    },
    {
        "thinking": (
            "This is an interesting problem that requires careful thought. "
            "Let me break it down into smaller, manageable parts. "
            "The first part involves understanding the core concept. "
            "The second part requires applying that concept to the specific case. "
            "Combining these insights reveals the complete picture and the "
            "relationship between the parts becomes clear."
        ),
        "output": (
            "The answer emerges from breaking the problem into its core "
            "components. By understanding each piece individually and then "
            "combining them, we arrive at a comprehensive solution that is "
            "both elegant and practical."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_response_text(prompt: str) -> str:
    """Build a full response with <think>…</think> markers."""
    idx = abs(hash(prompt)) % len(CANNED_RESPONSES)
    canned = CANNED_RESPONSES[idx]
    return f"<think>{canned['thinking']}</think>\n\n{canned['output']}"


def tokenize(text: str) -> list[str]:
    """Split text into word-level tokens, preserving whitespace."""
    tokens: list[str] = []
    current = ""
    for char in text:
        if char in (" ", "\n"):
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with SSE streaming."""
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    prompt = messages[-1]["content"] if messages else "Hello"
    full_text = build_response_text(prompt)
    tokens = tokenize(full_text)

    request_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # ---- Non-streaming (for completeness) --------------------------------
    if not stream:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(tokens),
                "total_tokens": 10 + len(tokens),
            },
        }

    # ---- Streaming (SSE) -------------------------------------------------
    async def generate() -> AsyncIterator[str]:
        # Simulate time-to-first-token latency
        await asyncio.sleep(0.08)

        for token in tokens:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.015)  # ~66 tokens/sec

        # Finish chunk
        finish_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(finish_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "server": "mock-upstream-llm"}


@app.get("/v1/models")
async def list_models():
    """List mock models."""
    return {
        "object": "list",
        "data": [{"id": "mock-reasoning-model", "object": "model", "owned_by": "mock"}],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock upstream LLM server for gateway testing")
    parser.add_argument("--port", type=int, default=int(os.getenv("MOCK_PORT", "9000")),
                        help="Port to listen on (default: 9000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"""{CYAN}
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ┌┐┌┐  ┌──┐  ┌──┐ ┐┌                                    ║
    ║   │└┘│  │  │  │    ├┴┐                                    ║
    ║   │  │  └──┘  └──┘ ┘ └   upstream                        ║
    ║                                                           ║
    ║   {YELLOW}Fake LLM with <think> reasoning markers{CYAN}               ║
    ║                                                           ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║   {BOLD}host{RESET}{CYAN}   {DIM}....{RESET}{CYAN}  {args.host:<40s} ║
    ║   {BOLD}port{RESET}{CYAN}   {DIM}....{RESET}{CYAN}  {str(args.port):<40s} ║
    ║   {BOLD}stream{RESET}{CYAN} {DIM}....{RESET}{CYAN}  {"SSE (text/event-stream)":<40s} ║
    ║   {BOLD}delay{RESET}{CYAN}  {DIM}....{RESET}{CYAN}  {"~15 ms/token  (~66 tok/s)":<40s} ║
    ║                                                           ║
    ║   {DIM}POST /v1/chat/completions   GET /health{RESET}{CYAN}            ║
    ║   {DIM}GET  /v1/models{RESET}{CYAN}                                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝{RESET}
""")
    uvicorn.run(app, host=args.host, port=args.port)
