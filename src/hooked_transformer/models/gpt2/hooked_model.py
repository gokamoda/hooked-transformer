from hooked_transformer.hook import Hook
from hooked_transformer.models.base.hooked_model import HookedModelForCausalLM
from hooked_transformer.utils.typing import (
    BATCH,
    HIDDEN_DIM,
    HIDDEN_DIM_TRIPLE,
    SEQUENCE,
    Tensor,
)

from .hook import AttnQKVHookResult, Hooks, _LayerHookResult


def split_qkv(
    hook_result: dict[str, Tensor[BATCH, SEQUENCE, HIDDEN_DIM_TRIPLE]],
) -> dict[str, Tensor[BATCH, SEQUENCE, HIDDEN_DIM]]:
    """Split the concatenated Q, K, V tensors into separate tensors."""
    hidden_dim_triple = hook_result["out_hidden_states"].shape[-1]
    hidden_dim = hidden_dim_triple // 3
    q, k, v = hook_result["out_hidden_states"].split(hidden_dim, dim=2)

    return {"out_q_proj": q, "out_k_proj": k, "out_v_proj": v}


class HookedGPT2LMHeadModel(HookedModelForCausalLM):
    """Hooked GPT-2 model for causal language modeling."""

    def __init__(self, model):
        super().__init__(model)

    def set_qkv_proj_hooks(self, layer_index: int, qkv_config: dict[str, list[str]]):
        self.hooks.layers[layer_index].self_attn_qkv_proj = Hook(
            module=self.model.model.layers[layer_index].self_attn.c_attn,
            result_class=AttnQKVHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["layers"]["attn_c_attn"]["output"],
            keep=["out_hidden_states"],
            post_process=split_qkv,
        )

    def set_layers_hook(self, layers_hook_config: list[dict[str, list[str]]]):
        print(layers_hook_config)
        for layer_index, layer_config in enumerate(layers_hook_config):
            self.hooks.layers.append(_LayerHookResult())
            self.set_layer_hook(
                layer_config=layer_config,
                layer_index=layer_index,
            )

    def initialize_hooks(self):
        self.hooks = Hooks()
