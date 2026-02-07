#!/usr/bin/env python3
"""
Reasoning LLM Gateway Server

A Python 3.11+ API server that acts as a gateway in front of /chat/completions APIs,
providing enhanced streaming with reasoning summarization.

Features:
- Streams content in order: Prompt Summary -> Reasoning Summary -> Final Output
- Failure-resilient with automatic retries and circuit breakers
- Minimized TTFT through immediate prompt summary streaming
- Support for multiple reasoning-capable models

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8080
    # or
    python server.py
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a reasoning-capable model."""
    name: str
    upstream_url: str
    reasoning_start_markers: list[str] = field(default_factory=lambda: ["<think>", "<reasoning>", "Let me think", "Let's think"])
    reasoning_end_markers: list[str] = field(default_factory=lambda: ["</think>", "</reasoning>", "Therefore,", "In conclusion,"])
    supports_reasoning: bool = True


# Default model configurations — extend by adding entries to this dict.
# Models not found here fall back to "default".
# To add a model at runtime, append to DEFAULT_MODELS before the server starts,
# or set UPSTREAM_URL to point at the appropriate backend.
DEFAULT_MODELS: dict[str, ModelConfig] = {
    "default": ModelConfig(
        name="default",
        upstream_url=os.getenv("UPSTREAM_URL", "http://localhost:9000"),
    ),
}


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for upstream service protection."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        # HALF_OPEN
        return self.half_open_calls < self.half_open_max_calls

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failures = 0
        else:
            self.failures = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN


# Global circuit breakers per upstream
circuit_breakers: dict[str, CircuitBreaker] = {}


# Reasoning effort to token budget mapping
REASONING_EFFORT_CONFIG = {
    "low": {
        "max_reasoning_tokens": 256,
        "summarize_max_length": 100,
    },
    "medium": {
        "max_reasoning_tokens": 1024,
        "summarize_max_length": 200,
    },
    "high": {
        "max_reasoning_tokens": 4096,
        "summarize_max_length": 500,
    },
}


# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[Message]
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = True
    reasoning_effort: str | None = Field(default=None, description="Reasoning effort level: low, medium, high")

    # Additional OpenAI-compatible fields
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: list[str] | str | None = None


class StreamPhase(Enum):
    """Phases of the enhanced streaming response."""
    PROMPT_SUMMARY = "prompt_summary"
    REASONING_SUMMARY = "reasoning_summary"
    FINAL_OUTPUT = "final_output"


# ============================================================================
# Reasoning Detection and Summarization
# ============================================================================

class ReasoningDetector:
    """
    Detects and extracts reasoning content from model output.

    Assumptions documented:
    1. Reasoning content is typically enclosed in markers like <think>...</think>
    2. Or indicated by phrases like "Let me think", "Let's analyze"
    3. Final output follows reasoning, often after "Therefore" or "In conclusion"
    4. If no explicit markers, we use heuristics based on content structure
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.reasoning_buffer = ""
        self.output_buffer = ""
        self.in_reasoning = False
        self.reasoning_complete = False

    def process_token(self, token: str) -> tuple[str | None, str | None]:
        """
        Process a token and determine if it's reasoning or final output.

        Returns: (reasoning_token, output_token) - one will be None
        """
        combined = self.reasoning_buffer + self.output_buffer + token

        # Check for reasoning start
        if not self.in_reasoning and not self.reasoning_complete:
            for marker in self.config.reasoning_start_markers:
                if marker and marker in combined:
                    self.in_reasoning = True
                    # Return content before marker as output, rest as reasoning
                    idx = combined.find(marker)
                    before = combined[:idx]
                    after = combined[idx + len(marker):]
                    self.reasoning_buffer = after
                    self.output_buffer = ""
                    return (None, before) if before else (None, None)

        # Check for reasoning end
        if self.in_reasoning:
            self.reasoning_buffer += token
            for marker in self.config.reasoning_end_markers:
                if marker and marker in self.reasoning_buffer:
                    self.in_reasoning = False
                    self.reasoning_complete = True
                    idx = self.reasoning_buffer.find(marker)
                    reasoning_content = self.reasoning_buffer[:idx]
                    remaining = self.reasoning_buffer[idx + len(marker):]
                    self.reasoning_buffer = ""
                    self.output_buffer = remaining
                    # Return remaining text after end-marker as output
                    if remaining:
                        return (reasoning_content, remaining)
                    return (reasoning_content, None)
            # Buffer reasoning tokens without emitting — avoids double-counting
            return (None, None)

        # After reasoning or no reasoning detected
        self.output_buffer += token
        return (None, token)

    def get_accumulated_reasoning(self) -> str:
        """Get all accumulated reasoning content."""
        return self.reasoning_buffer

    def get_accumulated_output(self) -> str:
        """Get all accumulated output content."""
        return self.output_buffer


def summarize_prompt(messages: list[Message], max_length: int = 100) -> str:
    """
    Generate a concise summary of the user's prompt.

    This provides immediate feedback to the user while waiting for the model.
    """
    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        return "Processing your request..."

    last_message = user_messages[-1].content
    if len(last_message) <= max_length:
        return f"Processing: \"{last_message}\""

    # Truncate and add ellipsis
    truncated = last_message[:max_length].rsplit(" ", 1)[0]
    return f"Processing: \"{truncated}...\""


def summarize_reasoning(reasoning_content: str, max_length: int = 200) -> str:
    """
    Summarize the model's reasoning into a digestible format.

    Extracts key points and presents them concisely.
    """
    if not reasoning_content:
        return ""

    # Extract sentences/points
    sentences = re.split(r'[.!?]\s+', reasoning_content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ""

    # Take key sentences (first and last, plus any with keywords)
    key_phrases = ["therefore", "because", "conclude", "result", "answer", "solution"]
    summary_parts = []

    # Always include first sentence (sets up the reasoning)
    if sentences:
        summary_parts.append(sentences[0])

    # Include sentences with key phrases
    for sentence in sentences[1:-1]:
        if any(phrase in sentence.lower() for phrase in key_phrases):
            summary_parts.append(sentence)
            if len(". ".join(summary_parts)) > max_length:
                break

    # Include last sentence if different (often contains conclusion)
    if len(sentences) > 1 and sentences[-1] not in summary_parts:
        summary_parts.append(sentences[-1])

    summary = ". ".join(summary_parts)
    if len(summary) > max_length:
        summary = summary[:max_length].rsplit(" ", 1)[0] + "..."

    return summary


# ============================================================================
# Upstream Communication
# ============================================================================

async def stream_from_upstream(
    client: httpx.AsyncClient,
    url: str,
    request: ChatCompletionRequest,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> AsyncIterator[str]:
    """
    Stream tokens from the upstream /chat/completions API.

    Handles SSE parsing and yields individual tokens.
    Retries on connection failures and 5xx errors with exponential backoff.
    Once streaming begins, errors propagate immediately (no mid-stream retry).
    """
    payload = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": True,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
    }
    if request.stop:
        payload["stop"] = request.stop

    # Add reasoning_effort to upstream payload if specified
    if request.reasoning_effort:
        effort_config = REASONING_EFFORT_CONFIG.get(request.reasoning_effort.lower())
        if effort_config:
            payload["reasoning_effort"] = request.reasoning_effort
            payload["max_reasoning_tokens"] = effort_config["max_reasoning_tokens"]

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    # Add API key if configured
    api_key = os.getenv("UPSTREAM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with client.stream(
                "POST",
                f"{url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                                yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue
                return  # Stream completed successfully
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(1.0 * (2 ** attempt), 8.0)
                logger.warning(
                    "Upstream connection failed (attempt %d/%d): %s. Retrying in %.1fs…",
                    attempt + 1, max_retries, e, delay,
                )
                await asyncio.sleep(delay)
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500 and attempt < max_retries - 1:
                delay = min(1.0 * (2 ** attempt), 8.0)
                logger.warning(
                    "Upstream server error %d (attempt %d/%d). Retrying in %.1fs…",
                    e.response.status_code, attempt + 1, max_retries, delay,
                )
                await asyncio.sleep(delay)
                last_error = e
            else:
                raise  # 4xx or final attempt — propagate immediately
    if last_error:
        raise last_error


# ============================================================================
# Gateway Logic
# ============================================================================

def format_sse_event(event_type: str, data: dict) -> str:
    """Format data as an SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def format_sse_data(data: dict) -> str:
    """Format data as SSE data-only event."""
    return f"data: {json.dumps(data)}\n\n"


async def enhanced_stream_response(
    request: ChatCompletionRequest,
    model_config: ModelConfig,
) -> AsyncIterator[str]:
    """
    Generate the enhanced streaming response.

    Order of content:
    1. Prompt summary (immediate, to minimize TTFT)
    2. Reasoning summary (after reasoning is detected and collected)
    3. Final output (streamed as received)
    """
    request_id = str(uuid.uuid4())
    created = int(time.time())

    # Get or create circuit breaker for this upstream
    if model_config.upstream_url not in circuit_breakers:
        circuit_breakers[model_config.upstream_url] = CircuitBreaker()
    breaker = circuit_breakers[model_config.upstream_url]

    if not breaker.can_execute():
        yield format_sse_event("error", {
            "error": "Service temporarily unavailable",
            "retry_after": breaker.recovery_timeout,
        })
        return

    # Phase 1: Stream prompt summary immediately (minimizes TTFT)
    prompt_summary = summarize_prompt(request.messages)
    yield format_sse_event("phase", {"phase": "prompt_summary"})
    yield format_sse_data({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": f"[Prompt Summary] {prompt_summary}\n\n"},
            "finish_reason": None,
        }],
        "phase": "prompt_summary",
    })

    # Initialize reasoning detector
    detector = ReasoningDetector(model_config)
    reasoning_chunks: list[str] = []
    output_chunks: list[str] = []
    reasoning_summary_sent = False

    # Get reasoning effort config for summarization
    effort_config = REASONING_EFFORT_CONFIG.get(
        (request.reasoning_effort or "medium").lower(),
        REASONING_EFFORT_CONFIG["medium"]
    )
    reasoning_summary_max_length = effort_config["summarize_max_length"]

    # Phase 2 & 3: Stream from upstream and process
    async with httpx.AsyncClient() as client:
        try:
            async for token in stream_from_upstream(
                client,
                model_config.upstream_url,
                request,
            ):
                reasoning_token, output_token = detector.process_token(token)

                if reasoning_token:
                    reasoning_chunks.append(reasoning_token)

                if output_token:
                    # Before streaming output, send reasoning summary if we have reasoning
                    if not reasoning_summary_sent and reasoning_chunks:
                        full_reasoning = "".join(reasoning_chunks)
                        reasoning_summary = summarize_reasoning(full_reasoning, reasoning_summary_max_length)
                        if reasoning_summary:
                            yield format_sse_event("phase", {"phase": "reasoning_summary"})
                            yield format_sse_data({
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": f"[Reasoning Summary] {reasoning_summary}\n\n"},
                                    "finish_reason": None,
                                }],
                                "phase": "reasoning_summary",
                            })
                        reasoning_summary_sent = True
                        yield format_sse_event("phase", {"phase": "final_output"})

                    # Stream the output token
                    output_chunks.append(output_token)
                    yield format_sse_data({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": output_token},
                            "finish_reason": None,
                        }],
                        "phase": "final_output",
                    })

            # Handle incomplete or unsummarized reasoning at end of stream
            pending = detector.reasoning_buffer if detector.in_reasoning else ""
            if pending or (reasoning_chunks and not reasoning_summary_sent):
                # Reasoning started but never closed, or closed with no output following
                full_reasoning = pending or "".join(reasoning_chunks)
                reasoning_summary = summarize_reasoning(full_reasoning, reasoning_summary_max_length)
                if reasoning_summary:
                    yield format_sse_event("phase", {"phase": "reasoning_summary"})
                    yield format_sse_data({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"[Reasoning Summary] {reasoning_summary}\n\n"},
                            "finish_reason": None,
                        }],
                        "phase": "reasoning_summary",
                    })

                # Stream reasoning as final output when no real output followed
                if not output_chunks:
                    yield format_sse_event("phase", {"phase": "final_output"})
                    yield format_sse_data({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": full_reasoning},
                            "finish_reason": None,
                        }],
                        "phase": "final_output",
                    })

            # Final message
            yield format_sse_data({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            })
            yield "data: [DONE]\n\n"

            breaker.record_success()

        except httpx.HTTPStatusError as e:
            breaker.record_failure()
            logger.error(f"Upstream HTTP error: {e.response.status_code}")
            yield format_sse_event("error", {
                "error": f"Upstream error: {e.response.status_code}",
                "message": str(e),
            })
        except httpx.RequestError as e:
            breaker.record_failure()
            logger.error(f"Upstream request error: {e}")
            yield format_sse_event("error", {
                "error": "Upstream connection failed",
                "message": str(e),
            })
        except Exception as e:
            breaker.record_failure()
            logger.exception("Unexpected error during streaming")
            yield format_sse_event("error", {
                "error": "Internal server error",
                "message": str(e),
            })


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Reasoning LLM Gateway Server")
    logger.info(f"Available models: {list(DEFAULT_MODELS.keys())}")
    yield
    logger.info("Shutting down Reasoning LLM Gateway Server")


app = FastAPI(
    title="Reasoning LLM Gateway",
    description="Enhanced streaming gateway for reasoning-capable LLMs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models": list(DEFAULT_MODELS.keys())}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "owned_by": "gateway",
                "supports_reasoning": config.supports_reasoning,
            }
            for name, config in DEFAULT_MODELS.items()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Enhanced chat completions endpoint.

    Returns streaming response with:
    1. Prompt summary
    2. Reasoning summary (if detected)
    3. Final model output
    """
    # Get model configuration
    model_name = request.model if request.model in DEFAULT_MODELS else "default"
    model_config = DEFAULT_MODELS[model_name]

    if not request.stream:
        raise HTTPException(
            status_code=400,
            detail="This gateway only supports streaming mode. Set stream=true."
        )

    return StreamingResponse(
        enhanced_stream_response(request, model_config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Reasoning LLM Gateway",
        "version": "1.0.0",
        "endpoints": {
            "/v1/chat/completions": "POST - Enhanced chat completions with reasoning summarization",
            "/v1/models": "GET - List available models",
            "/health": "GET - Health check",
        },
        "documentation": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
