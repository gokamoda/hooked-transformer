from typing import Annotated

import torch
from typing_extensions import Generic, TypeVarTuple

Ts = TypeVarTuple("Ts")


class Tensor(Generic[*Ts], torch.Tensor):
    pass


BATCH = Annotated[int, "batch_size"]
SEQUENCE = Annotated[int, "length"]
VOCAB = Annotated[int, "vocab"]
HEAD = Annotated[int, "head"]
HEAD_KV = Annotated[int, "head_kv"]
HIDDEN_DIM = Annotated[int, "hidden_dim"]
HIDDEN_DIM_TRIPLE = Annotated[int, "hidden_dim_triple"]
HEAD_DIM = Annotated[int, "head_dim"]
HEAD_DIM_DIV_2 = Annotated[int, "head_dim_div_2"]
LAYER = Annotated[int, "layer"]
LAYER_PLUS_1 = Annotated[int, "layer_plus_1"]
N = Annotated[int, "n"]
