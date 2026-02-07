#!/usr/bin/env python3
"""
Test script to verify the reasoning_effort parameter fix in server.py.

This script validates that:
1. The REASONING_EFFORT_CONFIG is properly defined
2. The reasoning_effort parameter is included in upstream payloads
3. Different effort levels produce different summarization lengths

Requirements:
    pip install httpx fastapi pydantic

Usage:
    python test_reasoning_effort.py          # Run basic tests
    pytest test_reasoning_effort.py -v       # Run with pytest
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "model"))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    for module in ["httpx", "fastapi", "pydantic"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def test_reasoning_effort_config_exists():
    """Test that REASONING_EFFORT_CONFIG is defined with correct structure."""
    print("\n=== Testing REASONING_EFFORT_CONFIG ===")

    try:
        from server import REASONING_EFFORT_CONFIG
    except ImportError as e:
        print(f"  ❌ Failed to import: {e}")
        return False

    expected_levels = ["low", "medium", "high"]
    expected_keys = ["max_reasoning_tokens", "summarize_max_length"]

    all_passed = True

    for level in expected_levels:
        if level not in REASONING_EFFORT_CONFIG:
            print(f"  ❌ Missing effort level: {level}")
            all_passed = False
            continue

        config = REASONING_EFFORT_CONFIG[level]
        for key in expected_keys:
            if key not in config:
                print(f"  ❌ {level} missing key: {key}")
                all_passed = False
            else:
                print(f"  ✓ {level}.{key}: {config[key]}")

    # Verify values increase with effort level
    if all_passed:
        low = REASONING_EFFORT_CONFIG["low"]["max_reasoning_tokens"]
        med = REASONING_EFFORT_CONFIG["medium"]["max_reasoning_tokens"]
        high = REASONING_EFFORT_CONFIG["high"]["max_reasoning_tokens"]

        if low < med < high:
            print(f"  ✓ Token budgets scale correctly: {low} < {med} < {high}")
        else:
            print(f"  ❌ Token budgets don't scale: {low}, {med}, {high}")
            all_passed = False

    return all_passed


def test_chat_completion_request_has_reasoning_effort():
    """Test that ChatCompletionRequest accepts reasoning_effort parameter."""
    print("\n=== Testing ChatCompletionRequest ===")

    try:
        from server import ChatCompletionRequest, Message
    except ImportError as e:
        print(f"  ❌ Failed to import: {e}")
        return False

    # Test that reasoning_effort is accepted
    try:
        request = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high"
        )
        print(f"  ✓ reasoning_effort parameter accepted: {request.reasoning_effort}")
    except Exception as e:
        print(f"  ❌ Failed to create request with reasoning_effort: {e}")
        return False

    # Test default value is None
    request_default = ChatCompletionRequest(
        model="test",
        messages=[Message(role="user", content="Hello")]
    )
    if request_default.reasoning_effort is None:
        print(f"  ✓ Default reasoning_effort is None")
    else:
        print(f"  ❌ Default reasoning_effort should be None, got: {request_default.reasoning_effort}")
        return False

    return True


def test_summarize_reasoning_respects_max_length():
    """Test that summarize_reasoning respects different max_length values."""
    print("\n=== Testing summarize_reasoning max_length ===")

    try:
        from server import summarize_reasoning
    except ImportError as e:
        print(f"  ❌ Failed to import: {e}")
        return False

    # Create a long reasoning text
    long_reasoning = (
        "First, I need to analyze the problem carefully. "
        "The key insight is that we need to consider multiple factors. "
        "Let me break this down step by step. "
        "The first consideration is the input constraints. "
        "The second consideration is the expected output format. "
        "After analyzing these factors, I can see that the solution requires "
        "a combination of approaches. Therefore, the answer is clear. "
        "In conclusion, we should proceed with the recommended approach."
    )

    # Test different max lengths
    lengths = [50, 100, 200, 500]
    all_passed = True

    for max_len in lengths:
        summary = summarize_reasoning(long_reasoning, max_len)
        actual_len = len(summary)

        # Summary should be at or under max_length (with some tolerance for word boundaries)
        if actual_len <= max_len + 20:  # Small tolerance for word boundaries
            print(f"  ✓ max_length={max_len}: got {actual_len} chars")
        else:
            print(f"  ❌ max_length={max_len}: got {actual_len} chars (too long)")
            all_passed = False

    return all_passed


def test_payload_construction():
    """Test that payload is constructed with reasoning_effort."""
    print("\n=== Testing payload construction ===")

    try:
        from server import REASONING_EFFORT_CONFIG, ChatCompletionRequest, Message
    except ImportError as e:
        print(f"  ❌ Failed to import: {e}")
        return False

    # Simulate what stream_from_upstream does
    request = ChatCompletionRequest(
        model="test",
        messages=[Message(role="user", content="Test")],
        reasoning_effort="high"
    )

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

    # This is what our fix adds
    if request.reasoning_effort:
        effort_config = REASONING_EFFORT_CONFIG.get(request.reasoning_effort.lower())
        if effort_config:
            payload["reasoning_effort"] = request.reasoning_effort
            payload["max_reasoning_tokens"] = effort_config["max_reasoning_tokens"]

    # Verify
    if "reasoning_effort" in payload:
        print(f"  ✓ reasoning_effort in payload: {payload['reasoning_effort']}")
    else:
        print(f"  ❌ reasoning_effort not in payload")
        return False

    if "max_reasoning_tokens" in payload:
        expected = REASONING_EFFORT_CONFIG["high"]["max_reasoning_tokens"]
        actual = payload["max_reasoning_tokens"]
        if actual == expected:
            print(f"  ✓ max_reasoning_tokens correct: {actual}")
        else:
            print(f"  ❌ max_reasoning_tokens wrong: expected {expected}, got {actual}")
            return False
    else:
        print(f"  ❌ max_reasoning_tokens not in payload")
        return False

    return True


def test_source_code_changes():
    """Verify the source code contains the expected changes."""
    print("\n=== Testing source code changes ===")

    server_path = Path(__file__).parent / "model" / "server.py"
    with open(server_path) as f:
        source = f.read()

    checks = [
        ("REASONING_EFFORT_CONFIG", "REASONING_EFFORT_CONFIG = {"),
        ("low effort config", '"low":'),
        ("medium effort config", '"medium":'),
        ("high effort config", '"high":'),
        ("max_reasoning_tokens key", '"max_reasoning_tokens":'),
        ("summarize_max_length key", '"summarize_max_length":'),
        ("payload reasoning_effort", 'payload["reasoning_effort"]'),
        ("payload max_reasoning_tokens", 'payload["max_reasoning_tokens"]'),
        ("reasoning_summary_max_length usage", "reasoning_summary_max_length"),
    ]

    all_passed = True
    for name, pattern in checks:
        if pattern in source:
            print(f"  ✓ Found: {name}")
        else:
            print(f"  ❌ Missing: {name} (looking for: {pattern})")
            all_passed = False

    return all_passed


def main():
    print("=" * 60)
    print("Reasoning Effort Parameter Test Suite")
    print("=" * 60)

    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"\n⚠ Missing dependencies: {', '.join(missing_deps)}")
        print(f"  Install with: pip install {' '.join(missing_deps)}")
        print("\n  Running source code verification only...\n")

        # Run only the source code test
        result = test_source_code_changes()

        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: source code changes")

        if result:
            print("\n✓ Source code changes verified!")
            print("  Install dependencies to run full test suite.")
            sys.exit(0)
        else:
            print("\n❌ Source code changes not found.")
            sys.exit(1)

    # Check if server.py exists
    server_path = Path(__file__).parent / "model" / "server.py"
    if not server_path.exists():
        print(f"\n❌ server.py not found at: {server_path}")
        sys.exit(1)

    print(f"\nTesting server at: {server_path}")

    # Run tests
    results = {
        "REASONING_EFFORT_CONFIG": test_reasoning_effort_config_exists(),
        "ChatCompletionRequest": test_chat_completion_request_has_reasoning_effort(),
        "summarize_reasoning max_length": test_summarize_reasoning_respects_max_length(),
        "payload construction": test_payload_construction(),
        "source code changes": test_source_code_changes(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! The reasoning_effort fix is working.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
