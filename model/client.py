#!/usr/bin/env python3
"""
Reasoning LLM Gateway Client

A demonstration client that shows how to interact with the gateway server.
Handles SSE streaming and displays the phased output.

Usage:
    python client.py --url http://localhost:8080 --prompt "What is 2+2?"
    python client.py --url http://localhost:8080 --interactive
"""

import argparse
import json
import sys
from typing import Iterator

import httpx


class GatewayClient:
    """Client for the Reasoning LLM Gateway."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        prompt: str,
        model: str = "default",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> Iterator[dict]:
        """
        Send a chat request and yield streaming responses.

        Yields dictionaries with:
        - 'type': 'phase' | 'content' | 'error' | 'done'
        - 'data': The content or phase information
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                current_event = None
                for line in response.iter_lines():
                    line = line.strip()

                    if line.startswith("event: "):
                        current_event = line[7:]
                        continue

                    if line.startswith("data: "):
                        data = line[6:]

                        if data == "[DONE]":
                            yield {"type": "done", "data": None}
                            return

                        try:
                            parsed = json.loads(data)

                            if current_event == "phase":
                                yield {"type": "phase", "data": parsed.get("phase")}
                            elif current_event == "error":
                                yield {"type": "error", "data": parsed}
                            else:
                                # Content chunk
                                if parsed.get("choices"):
                                    delta = parsed["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield {
                                            "type": "content",
                                            "data": content,
                                            "phase": parsed.get("phase"),
                                        }

                                    finish_reason = parsed["choices"][0].get("finish_reason")
                                    if finish_reason:
                                        yield {"type": "finish", "data": finish_reason}

                            current_event = None
                        except json.JSONDecodeError:
                            continue

    def get_models(self) -> list[dict]:
        """Get list of available models."""
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json().get("data", [])

    def health_check(self) -> dict:
        """Check server health."""
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


def print_colored(text: str, color: str, end: str = "\n"):
    """Print colored text to terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}", end=end, flush=True)


def run_single_query(client: GatewayClient, prompt: str, model: str, verbose: bool = False):
    """Run a single query and display the results."""
    print_colored(f"\nQuery: {prompt}", "cyan")
    print("-" * 60)

    current_phase = None

    try:
        for event in client.chat(prompt, model=model):
            if event["type"] == "phase":
                phase = event["data"]
                if phase != current_phase:
                    current_phase = phase
                    if verbose:
                        print_colored(f"\n[Phase: {phase}]", "yellow")

            elif event["type"] == "content":
                content = event["data"]
                phase = event.get("phase")

                # Color code by phase
                if phase == "prompt_summary":
                    print_colored(content, "blue", end="")
                elif phase == "reasoning_summary":
                    print_colored(content, "magenta", end="")
                else:
                    print(content, end="", flush=True)

            elif event["type"] == "error":
                print_colored(f"\nError: {event['data']}", "red")

            elif event["type"] == "done":
                print("\n")

    except httpx.HTTPStatusError as e:
        print_colored(f"HTTP Error: {e.response.status_code}", "red")
    except httpx.RequestError as e:
        print_colored(f"Connection Error: {e}", "red")


def run_interactive(client: GatewayClient, model: str, verbose: bool = False):
    """Run in interactive mode."""
    print_colored("Reasoning LLM Gateway - Interactive Mode", "green")
    print_colored("Type 'quit' or 'exit' to stop, 'models' to list models\n", "yellow")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit"):
                print_colored("Goodbye!", "green")
                break
            if prompt.lower() == "models":
                models = client.get_models()
                print_colored("Available models:", "cyan")
                for m in models:
                    print(f"  - {m['id']} (reasoning: {m.get('supports_reasoning', 'unknown')})")
                continue

            run_single_query(client, prompt, model, verbose)

        except KeyboardInterrupt:
            print_colored("\nInterrupted. Goodbye!", "yellow")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Client for Reasoning LLM Gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python client.py --url http://localhost:8080 --prompt "Explain quantum computing"

  # Interactive mode
  python client.py --url http://localhost:8080 --interactive

  # With specific model
  python client.py --url http://localhost:8080 --prompt "Hello" --model deepseek-r1
        """
    )
    parser.add_argument("--url", default="http://localhost:8080", help="Gateway server URL")
    parser.add_argument("--prompt", help="Single prompt to send")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--model", default="default", help="Model to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show phase transitions")
    parser.add_argument("--health", action="store_true", help="Check server health")

    args = parser.parse_args()

    client = GatewayClient(args.url)

    if args.health:
        try:
            health = client.health_check()
            print_colored(f"Server Status: {health['status']}", "green")
            print(f"Available Models: {', '.join(health.get('models', []))}")
        except Exception as e:
            print_colored(f"Health check failed: {e}", "red")
            sys.exit(1)
        return

    if args.interactive:
        run_interactive(client, args.model, args.verbose)
    elif args.prompt:
        run_single_query(client, args.prompt, args.model, args.verbose)
    else:
        parser.print_help()
        print_colored("\nError: Provide --prompt or --interactive", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()
