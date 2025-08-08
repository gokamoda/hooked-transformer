import torch

from hooked_transformer.auto_hooked_model import AutoModelForCausalLMWithAliases


def test_gpt2_qproj():
    model_name_or_path = "openai-community/gpt2"
    model = AutoModelForCausalLMWithAliases.from_pretrained(model_name_or_path)

    hidden_size = model.model.config.hidden_size
    x = torch.randn(1, 1, hidden_size)

    q_proj = model.model.layers[0].self_attn.q_proj

    original = model.model.layers[0].self_attn.c_attn(x)[:, :, :hidden_size]
    reconstructed = q_proj(x)

    assert torch.allclose(
        reconstructed,
        original,
        atol=1e-5,
        rtol=1e-5,
    ), f"original:\n{original}\nreconstructed:\n{reconstructed}"


def test_gpt2_kproj():
    model_name_or_path = "openai-community/gpt2"
    model = AutoModelForCausalLMWithAliases.from_pretrained(model_name_or_path)

    hidden_size = model.model.config.hidden_size
    x = torch.randn(1, 1, hidden_size)

    k_proj = model.model.layers[0].self_attn.k_proj

    original = model.model.layers[0].self_attn.c_attn(x)[
        :, :, hidden_size : 2 * hidden_size
    ]
    reconstructed = k_proj(x)

    assert torch.allclose(
        reconstructed,
        original,
        atol=1e-5,
        rtol=1e-5,
    ), f"original:\n{original}\nreconstructed:\n{reconstructed}"


def test_gpt2_vproj():
    model_name_or_path = "openai-community/gpt2"
    model = AutoModelForCausalLMWithAliases.from_pretrained(model_name_or_path)

    hidden_size = model.model.config.hidden_size
    x = torch.randn(1, 1, hidden_size)

    v_proj = model.model.layers[0].self_attn.v_proj

    original = model.model.layers[0].self_attn.c_attn(x)[:, :, 2 * hidden_size :]
    reconstructed = v_proj(x)

    assert torch.allclose(
        reconstructed,
        original,
        atol=1e-5,
        rtol=1e-5,
    ), f"original:\n{original}\nreconstructed:\n{reconstructed}"
