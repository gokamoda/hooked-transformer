from dataclasses import dataclass

from hooked_transformer.hook import Hook
from hooked_transformer.utils.typing import BATCH, HIDDEN_DIM, SEQUENCE, VOCAB, Tensor

from .abstract import AbstractBatchResult
from .data import (
    CausalLMObservationBatchResult,
    EmbedPositionsObservationBatchResult,
    EmbedTokensObservationBatchResult,
    LayerObservationBatchResult,
    LMHeadObservationBatchResult,
    NormObservationBatchResult,
)


class Hooks:
    embed_tokens: Hook
    embed_positions: Hook
    layers: list[Hook]
    norm: Hook
    lm_head: Hook

    def __init__(self):
        self.embed_tokens = None
        self.embed_positions = None
        self.layers = []
        self.norm = None
        self.lm_head = None

    def __repr__(self, indent=1) -> str:
        tab = "\t" * indent
        return (
            f"Hooks\n"
            f"{tab}embed_tokens: {self.embed_tokens.__repr__(indent=indent + 1)}\n"
            f"{tab}embed_positions: {self.embed_positions.__repr__(indent=indent + 1)}\n"
            f"{tab}layers: {len(self.layers)} layers\n"
            f"{tab}norm: {self.norm.__repr__(indent=indent + 1)}\n"
            f"{tab}lm_head: {self.lm_head.__repr__(indent=indent + 1)}\n"
        )

    def get_layer_qkv_results(
        self, layer_index: int
    ) -> tuple[
        Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
        Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
        Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    ]:
        layer_hook = self.layers[layer_index]
        try:
            q_proj_output = layer_hook.self_attn_q_proj.result.out_hidden_states
        except AttributeError:
            q_proj_output = None

        try:
            k_proj_output = layer_hook.self_attn_k_proj.result.out_hidden_states
        except AttributeError:
            k_proj_output = None

        try:
            v_proj_output = layer_hook.self_attn_v_proj.result.out_hidden_states
        except AttributeError:
            v_proj_output = None

        return q_proj_output, k_proj_output, v_proj_output

    def get_layer_results(self, layer_index) -> LayerObservationBatchResult:
        layer_hook = self.layers[layer_index]

        try:
            pre_attn_norm_out = layer_hook.pre_attn_norm.result.out_hidden_states
        except AttributeError:
            pre_attn_norm_out = None

        try:
            attention_mask = layer_hook.self_attn.result.in_attention_mask
        except AttributeError:
            attention_mask = None

        try:
            position_embeddings = layer_hook.self_attn.result.in_position_embeddings
        except AttributeError:
            position_embeddings = None

        q_proj_output, k_proj_output, v_proj_output = self.get_layer_qkv_results(
            layer_index
        )

        try:
            attn_out_hidden_states = layer_hook.self_attn.result.out_hidden_states
        except AttributeError:
            attn_out_hidden_states = None

        try:
            pre_mlp_norm_out = layer_hook.pre_mlp_norm.result.out_hidden_states
        except AttributeError:
            pre_mlp_norm_out = None

        try:
            mlp_output = layer_hook.mlp.result.out_hidden_states
        except AttributeError:
            mlp_output = None

        try:
            out_hidden_states = layer_hook.self.result.out_hidden_states
        except AttributeError:
            out_hidden_states = None

        return LayerObservationBatchResult(
            pre_attn_norm_out=pre_attn_norm_out,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            attn_out_hidden_states=attn_out_hidden_states,
            q_proj_output=q_proj_output,
            k_proj_output=k_proj_output,
            v_proj_output=v_proj_output,
            pre_mlp_norm_out=pre_mlp_norm_out,
            mlp_out_hidden_states=mlp_output,
            out_hidden_states=out_hidden_states,
        )

    def get_results(self) -> CausalLMObservationBatchResult:
        if self.embed_tokens is None:
            embed_tokens_result = None
        else:
            try:
                input_tokens = self.embed_tokens.result.in_input
            except AttributeError:
                input_tokens = None

            try:
                out_hidden_states = self.embed_tokens.result.out_hidden_states
            except AttributeError:
                out_hidden_states = None
            embed_tokens_result = EmbedTokensObservationBatchResult(
                input_tokens=input_tokens,
                out_hidden_states=out_hidden_states,
            )

        if self.embed_positions is None:
            embed_positions_result = None
        else:
            embed_positions_result = EmbedPositionsObservationBatchResult(
                input_tokens=self.embed_positions.result.in_input,
                out_hidden_states=self.embed_positions.result.out_hidden_states,
            )

        if self.layers == []:
            layer_results = []
        else:
            layer_results = [
                self.get_layer_results(layer_index)
                for layer_index in range(len(self.layers))
            ]

        if self.norm is None:
            norm_result = None
        else:
            norm_result = NormObservationBatchResult(
                out_hidden_states=self.norm.result.out_hidden_states
            )

        if self.lm_head is None:
            lm_head_result = None
        else:
            lm_head_result = LMHeadObservationBatchResult(
                logits=self.lm_head.result.out_logits
            )

        return CausalLMObservationBatchResult(
            embed_tokens=embed_tokens_result,
            embed_positions=embed_positions_result,
            layers=layer_results,
            norm=norm_result,
            lm_head=lm_head_result,
        )


@dataclass(repr=False, init=False)
class EmbedTokensHookResult(AbstractBatchResult):
    in_input: Tensor[BATCH, SEQUENCE]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class EmbedPositionsHookResult(AbstractBatchResult):
    in_input: Tensor[BATCH, SEQUENCE]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class LMHeadHookResult(AbstractBatchResult):
    in_input: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_logits: Tensor[BATCH, SEQUENCE, VOCAB]


@dataclass(repr=False, init=False)
class NormHookResult(AbstractBatchResult):
    in_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class AttnHookResult(AbstractBatchResult):
    in_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    in_attention_mask: Tensor[BATCH, 1, SEQUENCE, SEQUENCE] | None = None
    in_position_embeddings: Tensor[BATCH, SEQUENCE, HIDDEN_DIM] | None = None


@dataclass(repr=False, init=False)
class AttnQProjHookResult(AbstractBatchResult):
    in_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class AttnKProjHookResult(AbstractBatchResult):
    in_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class AttnVProjHookResult(AbstractBatchResult):
    in_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class MLPHookResult(AbstractBatchResult):
    in_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class LayerHookResult(AbstractBatchResult):
    in_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class _LayerHookResult(AbstractBatchResult):
    self: Hook
    pre_attn_norm: Hook
    self_attn: Hook
    self_attn_q_proj: Hook
    self_attn_k_proj: Hook
    self_attn_v_proj: Hook
    pre_mlp_norm: Hook
    mlp: Hook
