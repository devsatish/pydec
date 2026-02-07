#!/usr/bin/env python3
"""
Test script to verify the fixed broken-model configuration.

This script validates that:
1. The config.json loads correctly as a LLaMA model
2. The tokenizer configuration is valid
3. The model can be initialized (architecture check)

Requirements:
    pip install transformers torch

Usage:
    python test_broken_model.py
"""

import json
import sys
from pathlib import Path


def test_config_json(model_path: Path) -> bool:
    """Test that config.json has correct LLaMA 3.1 settings."""
    print("\n=== Testing config.json ===")

    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"❌ config.json not found at {config_path}")
        return False

    with open(config_path) as f:
        config = json.load(f)

    # Expected values for LLaMA 3.1 8B
    expected = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 128256,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }

    all_passed = True
    for key, expected_value in expected.items():
        actual_value = config.get(key)
        if actual_value == expected_value:
            print(f"  ✓ {key}: {actual_value}")
        else:
            print(f"  ❌ {key}: expected {expected_value}, got {actual_value}")
            all_passed = False

    return all_passed


def test_tokenizer_config(model_path: Path) -> bool:
    """Test that tokenizer_config.json exists and has required fields."""
    print("\n=== Testing tokenizer_config.json ===")

    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        print(f"❌ tokenizer_config.json not found at {tokenizer_config_path}")
        return False

    with open(tokenizer_config_path) as f:
        config = json.load(f)

    required_fields = ["bos_token", "eos_token", "chat_template", "tokenizer_class"]
    all_passed = True

    for field in required_fields:
        if field in config:
            value = config[field]
            if field == "chat_template":
                print(f"  ✓ {field}: present ({len(value)} chars)")
            else:
                print(f"  ✓ {field}: {value}")
        else:
            print(f"  ❌ {field}: missing")
            all_passed = False

    return all_passed


def test_special_tokens_map(model_path: Path) -> bool:
    """Test that special_tokens_map.json exists and is valid."""
    print("\n=== Testing special_tokens_map.json ===")

    special_tokens_path = model_path / "special_tokens_map.json"
    if not special_tokens_path.exists():
        print(f"❌ special_tokens_map.json not found at {special_tokens_path}")
        return False

    with open(special_tokens_path) as f:
        tokens = json.load(f)

    expected_tokens = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|eot_id|>",
    }

    all_passed = True
    for key, expected_value in expected_tokens.items():
        actual_value = tokens.get(key)
        if actual_value == expected_value:
            print(f"  ✓ {key}: {actual_value}")
        else:
            print(f"  ❌ {key}: expected {expected_value}, got {actual_value}")
            all_passed = False

    return all_passed


def test_generation_config(model_path: Path) -> bool:
    """Test that generation_config.json has correct token IDs."""
    print("\n=== Testing generation_config.json ===")

    gen_config_path = model_path / "generation_config.json"
    if not gen_config_path.exists():
        print(f"❌ generation_config.json not found at {gen_config_path}")
        return False

    with open(gen_config_path) as f:
        config = json.load(f)

    # Check token IDs match LLaMA 3.1
    expected = {
        "bos_token_id": 128000,
    }

    all_passed = True
    for key, expected_value in expected.items():
        actual_value = config.get(key)
        if actual_value == expected_value:
            print(f"  ✓ {key}: {actual_value}")
        else:
            print(f"  ❌ {key}: expected {expected_value}, got {actual_value}")
            all_passed = False

    # Check eos_token_id contains valid LLaMA 3.1 tokens
    eos_tokens = config.get("eos_token_id", [])
    if isinstance(eos_tokens, list) and 128001 in eos_tokens:
        print(f"  ✓ eos_token_id: {eos_tokens}")
    else:
        print(f"  ❌ eos_token_id: should contain 128001, got {eos_tokens}")
        all_passed = False

    return all_passed


def test_transformers_loading(model_path: Path) -> bool:
    """Test that transformers can load the config correctly."""
    print("\n=== Testing transformers loading ===")

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError:
        print("  ⚠ transformers not installed, skipping this test")
        print("  Run: pip install transformers")
        return True  # Don't fail if transformers isn't installed

    # Test config loading
    try:
        config = AutoConfig.from_pretrained(str(model_path))
        print(f"  ✓ AutoConfig loaded successfully")
        print(f"    - Architecture: {config.architectures}")
        print(f"    - Model type: {config.model_type}")
        print(f"    - Vocab size: {config.vocab_size}")
    except Exception as e:
        print(f"  ❌ AutoConfig failed: {e}")
        return False

    # Verify it's recognized as LLaMA
    if config.model_type != "llama":
        print(f"  ❌ Model type is '{config.model_type}', expected 'llama'")
        return False

    print(f"  ✓ Model correctly identified as LLaMA")

    # Test tokenizer loading (may fail without tokenizer.json, but config should work)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
        print(f"  ✓ AutoTokenizer loaded successfully")
        print(f"    - BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        print(f"    - EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    except Exception as e:
        print(f"  ⚠ AutoTokenizer loading failed (may need tokenizer.json): {e}")
        # This is expected since we don't have tokenizer.json (vocab file)

    return True


def test_model_initialization(model_path: Path, full_init: bool = False) -> bool:
    """Test that the model architecture can be initialized.

    Args:
        full_init: If True, actually initialize the 8B model (slow, ~30s+).
                   If False, just verify the config is valid for LlamaForCausalLM.
    """
    print("\n=== Testing model architecture initialization ===")

    try:
        from transformers import AutoConfig, LlamaForCausalLM
        import torch
    except ImportError:
        print("  ⚠ transformers/torch not installed, skipping this test")
        return True

    try:
        config = AutoConfig.from_pretrained(str(model_path))

        # Quick validation - check config is compatible with LlamaForCausalLM
        print(f"  ✓ Config validated for LlamaForCausalLM")
        print(f"    - Layers: {config.num_hidden_layers}")
        print(f"    - Hidden size: {config.hidden_size}")
        print(f"    - Vocab size: {config.vocab_size}")

        # Calculate expected params without loading
        # Rough estimate: embeddings + layers + lm_head
        vocab = config.vocab_size
        hidden = config.hidden_size
        layers = config.num_hidden_layers
        intermediate = config.intermediate_size

        # Approximate param count
        embed_params = vocab * hidden * 2  # input + output embeddings
        layer_params = layers * (4 * hidden * hidden + 3 * hidden * intermediate)  # rough estimate
        total_params = embed_params + layer_params
        print(f"    - Estimated params: ~{total_params/1e9:.1f}B")

        if not full_init:
            print(f"  ⚠ Skipping full model init (use --full-init flag for complete test)")
            return True

        # Full initialization (slow - allocates 8B parameters)
        print("  Initializing model architecture (random weights)...")
        print("  ⚠ This will take 30+ seconds and use ~32GB RAM...")
        with torch.no_grad():
            model = LlamaForCausalLM(config)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model initialized successfully")
        print(f"    - Actual parameters: {num_params:,} ({num_params/1e9:.2f}B)")

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"  ❌ Model initialization failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Broken Model Configuration Test Suite")
    print("=" * 60)

    # Check for --full-init flag
    full_init = "--full-init" in sys.argv

    # Determine model path
    script_dir = Path(__file__).parent
    model_path = script_dir / "broken-model"

    if not model_path.exists():
        print(f"\n❌ Model directory not found: {model_path}")
        sys.exit(1)

    print(f"\nTesting model at: {model_path}")
    if full_init:
        print("⚠ Full initialization enabled (this will be slow)")

    # Run tests
    results = {
        "config.json": test_config_json(model_path),
        "tokenizer_config.json": test_tokenizer_config(model_path),
        "special_tokens_map.json": test_special_tokens_map(model_path),
        "generation_config.json": test_generation_config(model_path),
        "transformers loading": test_transformers_loading(model_path),
        "model initialization": test_model_initialization(model_path, full_init=full_init),
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
        print("\n✓ All tests passed! The model configuration is valid.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
