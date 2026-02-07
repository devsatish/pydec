---
library_name: transformers
pipeline_tag: text-generation
base_model:
- meta-llama/Meta-Llama-3.1-8B
---

# Fixed LLaMA 3.1 8B Configuration

This repository contains **corrected configuration files** for Meta-Llama-3.1-8B, fixing issues from the original [yunmorning/broken-model](https://huggingface.co/yunmorning/broken-model).

URL for the fixed model: [devsatish/fixed_model](https://huggingface.co/devsatish/fixed_model)


## Problem Identified

The original model's `config.json` contained **Qwen3 model settings** while the README claimed it was based on **Meta-Llama-3.1-8B**. This mismatch causes inference failures because:

1. The architecture class (`Qwen3ForCausalLM`) doesn't match the actual model weights
2. Token IDs (BOS, EOS) are wrong for LLaMA tokenizer
3. Vocabulary size mismatches the LLaMA tokenizer
4. Layer count and intermediate size are incorrect for LLaMA 3.1 8B

## Changes Made

### config.json

| Field | Original (Broken) | Fixed | Reason |
|-------|-------------------|-------|--------|
| `architectures` | `["Qwen3ForCausalLM"]` | `["LlamaForCausalLM"]` | Must match LLaMA model class |
| `model_type` | `"qwen3"` | `"llama"` | Required for correct model loading |
| `vocab_size` | `151936` | `128256` | LLaMA 3.1 vocabulary size |
| `bos_token_id` | `151643` | `128000` | LLaMA 3.1 BOS token |
| `eos_token_id` | `151645` | `128001` | LLaMA 3.1 EOS token |
| `num_hidden_layers` | `36` | `32` | LLaMA 3.1 8B has 32 layers |
| `intermediate_size` | `12288` | `14336` | Correct FFN dimension for LLaMA 3.1 8B |
| `rms_norm_eps` | `1e-06` | `1e-05` | LLaMA standard value |
| `rope_scaling` | `null` | `{...llama3 config}` | LLaMA 3.1 uses RoPE scaling |
| `rope_theta` | `1000000` | `500000.0` | LLaMA 3.1 8B value |

### generation_config.json

| Field | Original | Fixed | Reason |
|-------|----------|-------|--------|
| `bos_token_id` | `151643` | `128000` | Match LLaMA tokenizer |
| `eos_token_id` | `[151645, 151643]` | `[128001, 128008, 128009]` | LLaMA 3.1 EOS tokens |
| `pad_token_id` | `151643` | `128004` | LLaMA finetune pad token |

### New Files Added

- **`tokenizer_config.json`** - Required for chat template and proper tokenization
- **`special_tokens_map.json`** - Defines BOS, EOS, PAD token strings
- **`test_broken_model.py`** - Validation script to verify fixes

## Validation

Run the test script to verify the configuration:

```bash
pip install transformers torch
python test_broken_model.py
```

### Test Output

```
============================================================
Broken Model Configuration Test Suite
============================================================

Testing model at: /Users/satish/Desktop/tasks/broken-model

=== Testing config.json ===
  ✓ architectures: ['LlamaForCausalLM']
  ✓ model_type: llama
  ✓ vocab_size: 128256
  ✓ bos_token_id: 128000
  ✓ eos_token_id: 128001
  ✓ num_hidden_layers: 32
  ✓ hidden_size: 4096
  ✓ num_attention_heads: 32
  ✓ num_key_value_heads: 8

=== Testing tokenizer_config.json ===
  ✓ bos_token: <|begin_of_text|>
  ✓ eos_token: <|eot_id|>
  ✓ chat_template: present (4648 chars)
  ✓ tokenizer_class: PreTrainedTokenizerFast

=== Testing special_tokens_map.json ===
  ✓ bos_token: <|begin_of_text|>
  ✓ eos_token: <|eot_id|>

=== Testing generation_config.json ===
  ✓ bos_token_id: 128000
  ✓ eos_token_id: [128001, 128008, 128009]

=== Testing transformers loading ===
  ✓ AutoConfig loaded successfully
    - Architecture: ['LlamaForCausalLM']
    - Model type: llama
    - Vocab size: 128256
  ✓ Model correctly identified as LLaMA

=== Testing model architecture initialization ===
  ✓ Config validated for LlamaForCausalLM
    - Layers: 32
    - Hidden size: 4096
    - Vocab size: 128256
    - Estimated params: ~8.8B

============================================================
Test Summary
============================================================
  ✓ PASS: config.json
  ✓ PASS: tokenizer_config.json
  ✓ PASS: special_tokens_map.json
  ✓ PASS: generation_config.json
  ✓ PASS: transformers loading
  ✓ PASS: model initialization

Result: 6/6 tests passed

✓ All tests passed! The model configuration is valid.
```

## Usage

These config files can be used with actual LLaMA 3.1 8B weights:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load with official weights
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
```

Or use these corrected config files with your own fine-tuned weights.

## Files

| File | Description |
|------|-------------|
| `config.json` | Model architecture configuration |
| `generation_config.json` | Generation parameters |
| `tokenizer_config.json` | Tokenizer settings with chat template |
| `special_tokens_map.json` | Special token definitions |
| `merges.txt` | BPE merge rules |
| `test_broken_model.py` | Validation test script |
