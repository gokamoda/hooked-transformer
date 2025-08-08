GPT2_IO_KEYS = {
    "embed_tokens": {"output": ["hidden_states"]},
    "embed_positions": {"output": ["hidden_states"]},
    "layers": {
        "self": {"output": ["hidden_states"]},
        "pre_attn_norm": {"output": ["hidden_states"]},
        "self_attn": {"output": ["hidden_states", "attn_weights"]},
        "pre_mlp_norm": {"output": ["hidden_states"]},
        "mlp": {"output": ["hidden_states"]},
        "attn_c_attn": {"output": ["hidden_states"]},
    },
    "norm": {"output": ["hidden_states"]},
    "lm_head": {"output": ["logits"]},
}
