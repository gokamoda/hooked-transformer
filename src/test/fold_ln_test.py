import copy

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D

from hooked_transformer.auto_hooked_model import AutoHookedModelForCausalLM
from hooked_transformer.utils.typing import Tensor


class LinearRemovedLayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        return hidden_states / torch.sqrt(var + self.eps)


def fold_ln_weights(
    ln: nn.LayerNorm | nn.RMSNorm,
    linear: nn.Linear,
) -> tuple[LinearRemovedLayerNorm | nn.RMSNorm, nn.Linear]:
    """
    Fold linear computation inside layer norm into linear module.

    Step1: let x, y be the input and output of the linear module.
    Get W1.T and b1 such that y = x @ W1.T + b1.

    Note:
    Conv1D: y = x @ weight + bias
    Linear: y = x @ weight.T + bias

    Step2: let x,y be the input and output of the layer norm.
    Let f(x) be denominator of layer norm.
    Get W2 and b2 such that y = x / f(x) @ W2 + b2.

    Note:
    RMSNorm: f(x) = rms(x, eps) and W2 = diag(weight)
    LayerNorm: f(x) = sqrt(var + eps) and W2 = centering @ diag(weight)
    centering = I - 1/n

    Step3: Merge W2 and b2 into W1 and b1.
    ln(linear(x)) = (x/f(x) @ W2 + b2) @ W1.T + b1
    = x/f(x) @ (W2 @ W1.T) + (b2 @ W1.T + b1)
    """

    ln_has_bias = hasattr(ln, "bias") and ln.bias is not None
    linear_has_bias = hasattr(linear, "bias") and linear.bias is not None

    # get W1
    if isinstance(linear, Conv1D):
        linear_weight = linear.weight.T
    elif isinstance(linear, nn.Linear):
        linear_weight = linear.weight

    out_dim, in_dim = linear_weight.shape
    linear_new = nn.Linear(in_dim, out_dim, bias=(linear_has_bias or ln_has_bias))

    # merge W2 into W1
    new_linear_weight = linear_weight @ torch.diag(ln.weight)
    if isinstance(ln, nn.LayerNorm):
        centering = torch.diag(torch.ones(in_dim)) - 1 / in_dim
        new_linear_weight = new_linear_weight @ centering
    linear_new.weight = nn.Parameter(new_linear_weight)
    assert linear_new.weight.shape == (out_dim, in_dim)

    # merge b2 into b1
    if linear_has_bias or ln_has_bias:
        bias = torch.zeros(
            out_dim, dtype=linear_weight.dtype, device=linear.weight.device
        )
        if linear_has_bias:
            bias += linear.bias
        if ln_has_bias:
            bias += ln.bias @ linear_weight.T
        linear_new.bias = nn.Parameter(bias)

    # remove linear from ln
    if isinstance(ln, nn.LayerNorm):
        ln_new = LinearRemovedLayerNorm(ln.eps)
    else:
        ln_new = copy.deepcopy(ln)
        ln_new.weight = nn.Parameter(torch.ones_like(ln.weight))
        if ln_has_bias:
            ln_new.bias = nn.Parameter(torch.zeros_like(ln.bias))

    return (
        ln_new,
        linear_new,
    )


@pytest.mark.parametrize("linear_class", [nn.Linear, Conv1D])
@pytest.mark.parametrize("ln_class", [nn.LayerNorm, nn.RMSNorm])
@pytest.mark.parametrize("linear_bias", [True, False])
@pytest.mark.parametrize("ln_bias", [True, False])
def test_folded_ln_linear(
    linear_class: type[nn.Linear | Conv1D],
    ln_class: type[nn.LayerNorm | nn.RMSNorm],
    linear_bias: bool,
    ln_bias: bool,
):
    indim = 4
    outdim = 2

    if ln_class == nn.LayerNorm:
        ln = ln_class(indim, bias=ln_bias)
    elif ln_class == nn.RMSNorm:
        if not ln_bias:
            return
        ln = ln_class(indim)

    if linear_class == Conv1D:
        if not linear_bias:
            return
        linear = linear_class(outdim, indim)
    elif linear_class == nn.Linear:
        linear = linear_class(indim, outdim, bias=linear_bias)

    ln_new, linear_new = fold_ln_weights(ln, linear)

    x = torch.randn(3, indim) + 1
    assert torch.allclose(
        linear_new(ln_new(x)),
        linear(ln(x)),
        atol=1e-6,
        rtol=1e-6,
    ), (
        "Folded linear and layer norm do not match original linear and layer norm output."
    )


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "openai-community/gpt2",
        "meta-llama/Llama-3.2-1B",
    ],
)
def test_folded_ln_linear_model(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hooked_model = AutoHookedModelForCausalLM.from_pretrained(model_name_or_path)
    assert hooked_model is not None, "Model should be loaded successfully"

    hook_config = {
        "model.embed_tokens": ["out_hidden_states"],
        "model.model.layers[0].self_attn.q_proj": ["out_hidden_states"],
        "model.model.layers[0].self_attn.k_proj": ["out_hidden_states"],
        "model.model.layers[0].self_attn.v_proj": ["out_hidden_states"],
    }
    if hooked_model.model.config.pos_type == "ape":
        hook_config["model.embed_positions"] = ["out_hidden_states", "in_input"]
    hooked_model.register_hooks(hook_config)

    prompts = ["Tokyo is the capital of the", "Hello"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    generation_args = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
    }

    hooked_model(**inputs, **generation_args)

    result = hooked_model.hook_results()
    embed_out = result.embed_tokens.out_hidden_states
    if hooked_model.model.config.pos_type == "ape":
        embed_out += result.embed_positions.out_hidden_states
    original_q = result.layers[0].q_proj_output

    new_ln, new_q_proj = fold_ln_weights(
        hooked_model.model.model.layers[0].pre_attn_norm,
        hooked_model.model.model.layers[0].self_attn.q_proj,
    )
    reconstructed_q = new_q_proj(new_ln(embed_out))

    assert torch.allclose(
        reconstructed_q,
        original_q,
        atol=1e-5,
        rtol=1e-5,
    ), f"reconstructed:\n{reconstructed_q}\noriginal:\n{original_q}"
