import torch


def get_head_dim(self):
    return self.hidden_size // self.num_attention_heads


def pos_type(self):
    return "ape"


def get_li_weight(self):
    return self.weight.T


def get_q_proj(self) -> torch.nn.Linear:
    hidden_size = self.c_attn.nx
    linear = torch.nn.Linear(
        in_features=hidden_size, out_features=hidden_size, bias=True
    )
    linear.weight = torch.nn.Parameter(self.c_attn.weight[:, :hidden_size].T)
    linear.bias = torch.nn.Parameter(self.c_attn.bias[:hidden_size])
    return linear


def get_k_proj(self) -> torch.nn.Linear:
    hidden_size = self.c_attn.nx
    linear = torch.nn.Linear(
        in_features=hidden_size, out_features=hidden_size, bias=True
    )
    linear.weight = torch.nn.Parameter(
        self.c_attn.weight[:, hidden_size : 2 * hidden_size].T
    )
    linear.bias = torch.nn.Parameter(self.c_attn.bias[hidden_size : 2 * hidden_size])
    return linear


def get_v_proj(self) -> torch.nn.Linear:
    hidden_size = self.c_attn.nx
    linear = torch.nn.Linear(
        in_features=hidden_size, out_features=hidden_size, bias=True
    )
    linear.weight = torch.nn.Parameter(self.c_attn.weight[:, 2 * hidden_size :].T)
    linear.bias = torch.nn.Parameter(self.c_attn.bias[2 * hidden_size :])
    return linear


GPT2LMHEADMODEL_ALIAS = [
    ("model", "transformer"),
    ("model.embed_tokens", "transformer.wte"),
    ("model.embed_positions", "transformer.wpe"),
    ("model.norm", "transformer.ln_f"),
    ("model.layers", "transformer.h"),
    ("model.layers[*].self_attn", "transformer.h[*].attn"),
    ("model.layers[*].self_attn.q_proj", get_q_proj),
    ("model.layers[*].self_attn.k_proj", get_k_proj),
    ("model.layers[*].self_attn.v_proj", get_v_proj),
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
