# Q2: Debugging LLMs - Analysis

## Problem a) Analysis: Broken Model Investigation

### Hugging Face Repository
Link: https://huggingface.co/yunmorning/broken-model

### Investigation Approach

To identify why inference doesn't work with this model for a `/chat/completions` API server, we need to examine:

1. **config.json** - Model configuration
2. **tokenizer_config.json** - Tokenizer settings
3. **special_tokens_map.json** - Special token definitions
4. **chat_template** - How messages are formatted

### Common Issues That Break /chat/completions

1. **Missing or incorrect `chat_template`** - Required for proper message formatting
2. **Mismatched vocabulary sizes** - `vocab_size` in config vs actual tokenizer
3. **Incorrect `model_type`** - Must match the architecture
4. **Missing special tokens** - BOS, EOS, PAD tokens not defined
5. **Architecture mismatches** - Layers/heads don't match the model type

### Minimal Fix Approach

When fixing model configurations:
- Only change values that are **objectively incorrect**
- Don't "improve" values that are just non-standard
- Document each change with reasoning

### Typical Fixes Required

| Field | Issue | Fix |
|-------|-------|-----|
| `vocab_size` | Doesn't match tokenizer | Update to actual tokenizer vocabulary size |
| `chat_template` | Missing or malformed | Add proper Jinja2 template for the model family |
| `eos_token_id` | Wrong or missing | Set to correct token ID |
| `bos_token_id` | Wrong or missing | Set to correct token ID |
| `model_type` | Doesn't match architecture | Correct to actual model type |

---

## Problem b) Analysis: reasoning_effort Parameter

### Why reasoning_effort Has No Effect

The `reasoning_effort` parameter is currently ineffective because:

1. **The model is not a reasoning model**: Standard LLMs don't have a built-in reasoning mode. The `reasoning_effort` parameter is specific to models like:
   - OpenAI's o1/o1-mini
   - DeepSeek-R1
   - Other explicitly reasoning-trained models

2. **No inference engine support**: Even if the model supported reasoning, the inference engine (vLLM, TGI, etc.) must implement special handling for this parameter.

3. **No architectural support**: Reasoning models typically have:
   - Special training for chain-of-thought
   - Reasoning tokens that are generated but optionally hidden
   - Budget allocation mechanisms for thinking time

### Requirements to Make reasoning_effort Functional

#### 1. Model Requirements

- **Use a reasoning-capable model**: Replace with a model trained for reasoning:
  - DeepSeek-R1 series
  - Qwen-QwQ
  - Models fine-tuned with reasoning traces

- **Model must support reasoning tokens**: The model architecture should distinguish between:
  - Reasoning/thinking tokens (internal deliberation)
  - Output tokens (final response)

#### 2. Inference Engine Changes

- **Parameter parsing**: Engine must recognize and handle `reasoning_effort`
- **Budget allocation**: Map effort levels to token budgets:
  ```
  low    -> ~256 reasoning tokens
  medium -> ~1024 reasoning tokens
  high   -> ~4096+ reasoning tokens
  ```
- **Reasoning token handling**:
  - Generate reasoning tokens
  - Optionally expose or hide them based on configuration

#### 3. API Layer Changes

- **Request validation**: Accept `reasoning_effort` as valid parameter
- **Response format**: Include reasoning content if requested:
  ```json
  {
    "choices": [{
      "message": {
        "content": "Final answer",
        "reasoning_content": "Step-by-step thinking..."
      }
    }],
    "usage": {
      "reasoning_tokens": 512,
      "completion_tokens": 50
    }
  }
  ```

#### 4. Implementation Steps

1. **Deploy a reasoning model** (DeepSeek-R1, Qwen-QwQ)
2. **Configure inference engine** to support reasoning parameters
3. **Update API gateway** to:
   - Parse `reasoning_effort` parameter
   - Forward to inference engine
   - Format response with reasoning content
4. **Test with varying effort levels** to verify behavior changes

### Example Configuration for DeepSeek-R1

```python
# Inference engine configuration
reasoning_config = {
    "model": "deepseek-ai/DeepSeek-R1",
    "reasoning_effort_mapping": {
        "low": {"max_reasoning_tokens": 256},
        "medium": {"max_reasoning_tokens": 1024},
        "high": {"max_reasoning_tokens": 4096},
    },
    "expose_reasoning": True,  # Include in response
}
```

### Summary

To make `reasoning_effort` meaningful:

1. **Model**: Switch to a reasoning-capable model
2. **Engine**: Use inference engine with reasoning support
3. **API**: Implement parameter handling and response formatting
4. **Testing**: Validate that different effort levels produce observable differences in reasoning depth
