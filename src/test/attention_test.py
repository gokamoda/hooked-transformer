import pytest
import torch
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from hooked_transformer.auto_hooked_model import AutoHookedModelForCausalLM


def attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_dim: int,
    attention_mask: torch.Tensor,
    o_proj: torch.nn.Module,
    precompute_ov: bool = False,
    rope: bool = False,
    position_embeddings: torch.Tensor = None,
):
    if rope:
        assert position_embeddings is not None, (
            "Position embeddings must be provided for RoPE"
        )
    shape_q = (*query_states.shape[:-1], -1, head_dim)
    shape_kv = (*key_states.shape[:-1], -1, head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_kv).transpose(1, 2)

    if rope:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            q=query_states,
            k=key_states,
            sin=sin,
            cos=cos,
        )

    value_states = value_states.view(shape_kv).transpose(1, 2)

    # prepare for gqa.
    # can be done inside spda, but do it here for precompute_ov=True
    batch_size, num_kheads, seq_length, _ = key_states.shape
    _, num_qheads, _, head_dim = query_states.shape
    key_states = repeat_kv(
        key_states,
        n_rep=num_qheads // num_kheads,
    )
    value_states = repeat_kv(
        value_states,
        n_rep=num_qheads // num_kheads,
    )
    if precompute_ov:
        if hasattr(o_proj, "li_weight"):
            weight = o_proj.li_weight.T
        else:
            weight = o_proj.weight.T
        o_proj_by_head = weight.view(num_qheads, head_dim, -1)
        value_states = torch.einsum("bhsi,hid->bhsd", value_states, o_proj_by_head)

    attn_weighted = torch.nn.functional.scaled_dot_product_attention(
        query=query_states,
        key=key_states,
        value=value_states,
        attn_mask=attention_mask,
    )

    if precompute_ov:
        if hasattr(o_proj, "bias") and o_proj.bias is not None:
            return attn_weighted.sum(dim=1) + o_proj.bias
        return attn_weighted.sum(dim=1)
    attn_weighted = attn_weighted.transpose(1, 2).reshape(batch_size, seq_length, -1)
    attn_weighted = o_proj(attn_weighted)
    return attn_weighted


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "openai-community/gpt2",
        "meta-llama/Llama-3.2-1B",
    ],
)
@pytest.mark.parametrize("precompute_ov", [True, False])
def test_attention(model_name_or_path: str, precompute_ov: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hooked_model = AutoHookedModelForCausalLM.from_pretrained(model_name_or_path)
    assert hooked_model is not None, "Model should be loaded successfully"

    hook_config = {
        "model.model.layers[0].self_attn": ["out_hidden_states", "in_attention_mask"],
        "model.model.layers[0].self_attn.q_proj": ["out_hidden_states"],
        "model.model.layers[0].self_attn.k_proj": ["out_hidden_states"],
        "model.model.layers[0].self_attn.v_proj": ["out_hidden_states"],
    }
    if hooked_model.model.config.pos_type == "rope":
        hook_config["model.model.layers[0].self_attn"].append("in_position_embeddings")
    hooked_model.register_hooks(hook_config)

    prompts = ["Tokyo is the capital of the", "Hello"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    generation_args = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
    }

    hooked_model(**inputs, **generation_args)

    result = hooked_model.hook_results()

    reconstructed = attention(
        query_states=result.layers[0].q_proj_output,
        key_states=result.layers[0].k_proj_output,
        value_states=result.layers[0].v_proj_output,
        head_dim=hooked_model.model.config.hidden_size
        // hooked_model.model.config.num_attention_heads,
        attention_mask=result.layers[0].attention_mask,
        o_proj=hooked_model.model.model.layers[0].self_attn.o_proj,
        precompute_ov=precompute_ov,
        rope=hooked_model.model.config.pos_type == "rope",
        position_embeddings=result.layers[0].position_embeddings
        if hooked_model.model.config.pos_type == "rope"
        else None,
    )[:, -1, :]  # -1 to ignore the padding
    original = result.layers[0].attn_out_hidden_states[:, -1, :]
    assert torch.allclose(
        reconstructed,
        original,
        atol=1e-5,
        rtol=1e-5,
    ), f"reconstructed:\n{reconstructed}\noriginal:\n{original}"
