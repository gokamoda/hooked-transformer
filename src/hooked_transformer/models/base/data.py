from dataclasses import dataclass

from hooked_transformer.utils.logger import init_logging
from hooked_transformer.utils.typing import (
    BATCH,
    HEAD_DIM,
    HIDDEN_DIM,
    SEQUENCE,
    VOCAB,
    Tensor,
)

from .abstract import AbstractBatchResult

logger = init_logging(__name__)


@dataclass(repr=False, init=False)
class EmbedTokensObservationBatchResult(AbstractBatchResult):
    input_tokens: Tensor[BATCH, SEQUENCE]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class EmbedPositionsObservationBatchResult(AbstractBatchResult):
    input_tokens: Tensor[BATCH, SEQUENCE]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class LayerObservationBatchResult(AbstractBatchResult):
    pre_attn_norm_out: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    attn_out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    attention_mask: Tensor[BATCH, 1, SEQUENCE, SEQUENCE] | None = None
    position_embeddings: (
        tuple[Tensor[1, SEQUENCE, HEAD_DIM], Tensor[1, SEQUENCE, HEAD_DIM]] | None
    ) = None
    q_proj_output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    k_proj_output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    v_proj_output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    pre_mlp_norm_out: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    mlp_out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class NormObservationBatchResult(AbstractBatchResult):
    out_hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass(repr=False, init=False)
class LMHeadObservationBatchResult(AbstractBatchResult):
    logits: Tensor[BATCH, SEQUENCE, VOCAB]


@dataclass(repr=False, init=False)
class CausalLMObservationBatchResult(AbstractBatchResult):
    embed_tokens: EmbedTokensObservationBatchResult
    embed_positions: EmbedPositionsObservationBatchResult
    layers: list[LayerObservationBatchResult]
    norm: NormObservationBatchResult
    lm_head: LMHeadObservationBatchResult


RESULT_CLASSES = {"layer": LayerObservationBatchResult}
