import copy

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D

from hooked_transformer.auto_hooked_model import AutoHookedModelForCausalLM


def fold_ln_weights(
    ln: nn.LayerNorm | nn.RMSNorm,
    linear: nn.Linear,
) -> tuple[nn.LayerNorm | nn.RMSNorm, nn.Linear]:
    ln_has_bias = hasattr(ln, "bias") and ln.bias is not None
    linear_has_bias = hasattr(linear, "bias") and linear.bias is not None

    print(f"ln: {ln_has_bias}, linear: {linear_has_bias}")

    if isinstance(linear, Conv1D):
        linear_weight = linear.weight.T
    elif isinstance(linear, nn.Linear):
        linear_weight = linear.weight

    out_dim, in_dim = linear_weight.shape

    linear_new = nn.Linear(in_dim, out_dim, bias=linear_has_bias)
    linear_new.weight = nn.Parameter(linear_weight @ torch.diag(ln.weight))
    assert linear_new.weight.shape == (out_dim, in_dim)

    if linear_has_bias or ln_has_bias:
        bias = torch.zeros(
            out_dim, dtype=linear_weight.dtype, device=linear.weight.device
        )
        if linear_has_bias:
            bias += linear.bias
        if ln_has_bias:
            bias += ln.bias @ linear_weight.T
        linear_new.bias = nn.Parameter(bias)

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
            pass
        ln = ln_class(indim)

    if linear_class == Conv1D:
        if not linear_bias:
            pass
        linear = linear_class(outdim, indim)
    elif linear_class == nn.Linear:
        linear = linear_class(indim, outdim, bias=linear_bias)

    ln_new, linear_new = fold_ln_weights(ln, linear)

    x = torch.randn(3, indim)
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
