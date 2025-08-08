import pytest

from hooked_transformer.auto_model import (
    AutoModelForCausalLMWithAliases,
    get_deep_attr,
)


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "openai-community/gpt2",
        "meta-llama/Llama-3.2-1B",
    ],
)
def test_auto_model(model_name_or_path: str):
    model = AutoModelForCausalLMWithAliases.from_pretrained(model_name_or_path)
    assert model is not None, "Model should be loaded successfully"

    check_attribute_paths = [
        "model",
        "model.embed_tokens",
        "model.layers",
        "model.layers[*].self_attn",
        "model.layers[*].mlp",
        "model.norm",
        "model.layers[*].pre_attn_norm",
        "lm_head",
        "config.num_hidden_layers",
        "config.num_attention_heads",
        "config.hidden_size",
        "config.vocab_size",
        "config.head_dim",
    ]

    for attr_path in check_attribute_paths:
        value = get_deep_attr(model, attr_path)
        assert value is not None, f"Attribute {attr_path} should not be None"
