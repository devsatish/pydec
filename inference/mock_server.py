#!/usr/bin/env python3
"""
Mock OpenAI-compatible streaming server for testing benchmark.py.

Simulates the /v1/chat/completions endpoint with configurable latency
to mimic different engine performance characteristics.

Usage:
    # Terminal 1 — "vLLM" (slower)
    python mock_server.py --port 8000 --ttft 0.15 --token-delay 0.022 --name vLLM

    # Terminal 2 — "Friendli" (faster)
    python mock_server.py --port 8001 --ttft 0.06 --token-delay 0.012 --name Friendli

    # Terminal 3 — run the benchmark
    python benchmark.py --vllm-url http://localhost:8000 --friendli-url http://localhost:8001
"""

import argparse
import asyncio
import json
import random
import time

from aiohttp import web


def make_app(
    ttft_seconds: float,
    token_delay_seconds: float,
    name: str,
    tokens_per_response: int,
) -> web.Application:
    """Create the mock server app with the given latency profile."""

    async def handle_completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        max_tokens = min(body.get("max_tokens", tokens_per_response), tokens_per_response)

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)

        # Simulate time-to-first-token latency (with small jitter)
        jitter = random.uniform(0.8, 1.2)
        await asyncio.sleep(ttft_seconds * jitter)

        words = [
            "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog.",
            "This", "is", "a", "mock", "response", "for", "benchmarking", "purposes.",
            "Each", "token", "is", "streamed", "with", "configurable", "delay.",
        ]

        for i in range(max_tokens):
            word = words[i % len(words)]
            chunk = {
                "id": f"chatcmpl-mock-{i}",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None,
                    }
                ],
            }
            line = f"data: {json.dumps(chunk)}\n\n"
            await response.write(line.encode("utf-8"))

            # Simulate inter-token latency (with small jitter)
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(token_delay_seconds * jitter)

        # Send done signal
        await response.write(b"data: [DONE]\n\n")
        return response

    async def handle_health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "engine": name})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_completions)
    app.router.add_get("/health", handle_health)
    return app


def main():
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible streaming server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--ttft", type=float, default=0.10, help="Time-to-first-token in seconds (default: 0.10)")
    parser.add_argument("--token-delay", type=float, default=0.02, help="Delay between tokens in seconds (default: 0.02)")
    parser.add_argument("--name", default="MockEngine", help="Engine name for logging (default: MockEngine)")
    parser.add_argument("--tokens", type=int, default=60, help="Max tokens per response (default: 60)")

    args = parser.parse_args()

    print(f"Starting mock server: {args.name}")
    print(f"  Port:        {args.port}")
    print(f"  TTFT:        {args.ttft:.3f}s")
    print(f"  Token delay: {args.token_delay:.3f}s  (~{1/args.token_delay:.0f} tok/s)")
    print(f"  Max tokens:  {args.tokens}")
    print()

    app = make_app(args.ttft, args.token_delay, args.name, args.tokens)
    web.run_app(app, port=args.port, print=lambda msg: print(f"  {msg}"))


if __name__ == "__main__":
    main()
