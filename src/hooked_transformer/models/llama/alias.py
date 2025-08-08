def pos_type(self):
    return "rope"


LLAMAFORCAUDALLM_ALIAS = [
    ("model.layers[*].pre_attn_norm", "model.layers[*].input_layernorm"),
    (
        "model.layers[*].pre_attn_norm.eps",
        "model.layers[*].input_layernorm.variance_epsilon",
    ),
    ("model.layers[*].pre_mlp_norm", "model.layers[*].post_attention_layernorm"),
    ("config.pos_type", pos_type),
]
