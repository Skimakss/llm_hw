import os
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from config import MODEL_DTYPE, USE_FLASH_ATTN


def setup_torch_backend():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def create_model(tokenizer, is_smoke_test=False):
    model_config = {
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "head_dim": 128,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **model_config,
    )

    if USE_FLASH_ATTN:
        model = Qwen3ForCausalLM._from_config(
            config,
            attn_implementation="flash_attention_2",
        )
    else:
        model = Qwen3ForCausalLM._from_config(config)

    if os.getenv("RANK", "0") == "0":
        print(f"Model pad token id: {model.config.pad_token_id}")

    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        if os.getenv("RANK", "0") == "0":
            print(f"Total params: {total_params:,}")

    return model