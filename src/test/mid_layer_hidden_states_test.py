import pytest
import torch
from transformers import AutoTokenizer

from hooked_transformer.auto_hooked_model import AutoHookedModelForCausalLM


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "openai-community/gpt2",
        "meta-llama/Llama-3.2-1B",
    ],
)
def test_mid_hidden_states(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hooked_model = AutoHookedModelForCausalLM.from_pretrained(model_name_or_path)
    assert hooked_model is not None, "Model should be loaded successfully"

    hook_config = {
        "model.embed_tokens": ["out_hidden_states", "in_input"],
        "model.model.layers[0]": ["out_hidden_states"],
        "model.model.layers[0].self_attn": ["out_hidden_states"],
        "model.model.layers[0].mlp": ["out_hidden_states"],
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
    hidden_state_reconstructed = (
        result.embed_tokens.out_hidden_states
        + result.layers[0].attn_out_hidden_states
        + result.layers[0].mlp_out_hidden_states
    )

    if hooked_model.model.config.pos_type == "ape":
        hidden_state_reconstructed += result.embed_positions.out_hidden_states

    assert torch.allclose(
        hidden_state_reconstructed,
        result.layers[0].out_hidden_states,
        atol=1e-5,
        rtol=1e-5,
    ), "Reconstructed hidden states should match the layer's output hidden states"
