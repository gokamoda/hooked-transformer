def get_head_dim(self):
    return self.hidden_size // self.num_attention_heads


def pos_type(self):
    return "ape"


def get_li_weight(self):
    return self.weight.T


GPT2LMHEADMODEL_ALIAS = [
    ("model", "transformer"),
    ("model.embed_tokens", "transformer.wte"),
    ("model.embed_positions", "transformer.wpe"),
    ("model.norm", "transformer.ln_f"),
    ("model.layers", "transformer.h"),
    ("model.layers[*].self_attn", "transformer.h[*].attn"),
    ("model.layers[*].self_attn.o_proj", "transformer.h[*].attn.c_proj"),
    ("model.layers[*].self_attn.o_proj.li_weight", get_li_weight),
    ("model.layers[*].pre_attn_norm", "transformer.h[*].ln_1"),
    ("model.layers[*].pre_mlp_norm", "transformer.h[*].ln_2"),
    ("config.num_hidden_layers", "config.n_layer"),
    ("config.hidden_size", "config.n_embd"),
    ("config.num_attention_heads", "config.n_head"),
    ("config.head_dim", get_head_dim),
    ("config.pos_type", pos_type),
]
