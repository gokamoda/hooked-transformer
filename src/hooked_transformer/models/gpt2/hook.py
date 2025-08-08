from dataclasses import dataclass

from hooked_transformer.hook import Hook
from hooked_transformer.models.base.abstract import AbstractBatchResult
from hooked_transformer.models.base.hook import Hooks as BaseHooks
from hooked_transformer.utils.typing import BATCH, HIDDEN_DIM, SEQUENCE, Tensor


class Hooks(BaseHooks):
    """Hooks for the GPT-2 model."""

    embed_tokens: Hook
    embed_positions: Hook
    layers: list[Hook]
    norm: Hook
    lm_head: Hook

    def __init__(self):
        super().__init__()

    def get_layer_qkv_results(
        self, layer_index: int
    ) -> tuple[
        Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
        Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
        Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    ]:
        layer_hook = self.layers[layer_index]
        try:
            q_proj_output = layer_hook.self_attn_qkv_proj.result.out_q_proj
        except AttributeError:
            q_proj_output = None

        try:
            k_proj_output = layer_hook.self_attn_qkv_proj.result.out_k_proj
        except AttributeError:
            k_proj_output = None

        try:
            v_proj_output = layer_hook.self_attn_qkv_proj.result.out_v_proj
        except AttributeError:
            v_proj_output = None

        return q_proj_output, k_proj_output, v_proj_output


@dataclass(repr=False, init=False)
class AttnQKVHookResult(AbstractBatchResult):
    in_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_q_proj: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_k_proj: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_v_proj: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class _LayerHookResult(AbstractBatchResult):
    self: Hook
    pre_attn_norm: Hook
    self_attn: Hook
    self_attn_qkv_proj: Hook
    pre_mlp_norm: Hook
    mlp: Hook
