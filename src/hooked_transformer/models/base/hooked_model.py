import re

import torch

from hooked_transformer.auto_model import AutoModelForCausalLMWithAliases
from hooked_transformer.hook import Hook
from hooked_transformer.utils import get_deep_attr

from .data import (
    CausalLMObservationBatchResult,
)
from .hook import (
    AttnHookResult,
    AttnKProjHookResult,
    AttnQProjHookResult,
    AttnVProjHookResult,
    EmbedPositionsHookResult,
    EmbedTokensHookResult,
    Hooks,
    LayerHookResult,
    LMHeadHookResult,
    MLPHookResult,
    NormHookResult,
    _LayerHookResult,
)


def get_result_class_from_position(position: str):
    target = position.split(".")[-1]

    if re.match(r"layers\[\d+\]", target):
        return "layer"


class HookedModelForCausalLM:
    model: AutoModelForCausalLMWithAliases
    hook_config: dict[str, list[str]]
    generation_args: dict = {}
    hooks: Hooks

    def __init__(self, model: AutoModelForCausalLMWithAliases):
        self.model = model

    def __repr__(self):
        return f"HookedModelForCausalLM(model={self.model.__class__.__name__})"

    def set_embed_tokens_hook(self, keys: list[str]):
        self.hooks.embed_tokens = Hook(
            module=self.model.model.embed_tokens,
            result_class=EmbedTokensHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["embed_tokens"]["output"],
            keep=keys,
        )

    def set_embed_positions_hook(self, keys: list[str]):
        self.hooks.embed_positions = Hook(
            module=self.model.model.embed_positions,
            result_class=EmbedPositionsHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["embed_positions"]["output"],
            keep=keys,
        )

    def set_pre_attn_norm_hook(self, layer_index: int, keys: list[str]):
        self.hooks.layers[layer_index].pre_attn_norm = Hook(
            module=self.model.model.layers[layer_index].pre_attn_norm,
            result_class=NormHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["layers"]["pre_attn_norm"]["output"],
            keep=keys,
        )

    def set_pre_mlp_norm_hook(self, layer_index: int, keys: list[str]):
        self.hooks.layers[layer_index].pre_mlp_norm = Hook(
            module=self.model.model.layers[layer_index].pre_mlp_norm,
            result_class=NormHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["layers"]["pre_mlp_norm"]["output"],
            keep=keys,
        )

    def set_self_attn_hook(self, layer_index: int, keys: list[str]):
        self.hooks.layers[layer_index].self_attn = Hook(
            module=self.model.model.layers[layer_index].self_attn,
            result_class=AttnHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["layers"]["self_attn"]["output"],
            keep=keys,
        )

    def set_self_attn_q_proj_hook(self, layer_index: int, keys: list[str]):
        self.hooks.layers[layer_index].self_attn_q_proj = Hook(
            module=self.model.model.layers[layer_index].self_attn.q_proj,
            result_class=AttnQProjHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["layers"]["q_proj"]["output"],
            keep=keys,
        )

    def set_self_attn_k_proj_hook(self, layer_index: int, keys: list[str]):
        self.hooks.layers[layer_index].self_attn_k_proj = Hook(
            module=self.model.model.layers[layer_index].self_attn.k_proj,
            result_class=AttnKProjHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["layers"]["k_proj"]["output"],
            keep=keys,
        )

    def set_self_attn_v_proj_hook(self, layer_index: int, keys: list[str]):
        self.hooks.layers[layer_index].self_attn_v_proj = Hook(
            module=self.model.model.layers[layer_index].self_attn.v_proj,
            result_class=AttnVProjHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["layers"]["v_proj"]["output"],
            keep=keys,
        )

    def set_qkv_proj_hooks(self, layer_index: int, qkv_config: dict[str, list[str]]):
        if qkv_config["q"] != []:
            self.set_self_attn_q_proj_hook(layer_index, qkv_config["q"])
        if qkv_config["k"] != []:
            self.set_self_attn_k_proj_hook(layer_index, qkv_config["k"])
        if qkv_config["v"] != []:
            self.set_self_attn_v_proj_hook(layer_index, qkv_config["v"])

    def set_mlp_hook(self, layer_index: int, keys: list[str]):
        self.hooks.layers[layer_index].mlp = Hook(
            module=self.model.model.layers[layer_index].mlp,
            result_class=MLPHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["layers"]["mlp"]["output"],
            keep=keys,
        )

    def set_layer_hook(self, layer_index: int, layer_config: dict[str, list[str]]):
        qkv_config = {"q": [], "k": [], "v": []}
        for position, keys in layer_config.items():
            if position == "self":
                self.hooks.layers[layer_index].self = Hook(
                    module=self.model.model.layers[layer_index],
                    result_class=LayerHookResult,
                    to_cpu=True,
                    output_keys=self.model.io_keys["layers"]["self"]["output"],
                    keep=keys,
                )
            elif position == "pre_attn_norm":
                self.set_pre_attn_norm_hook(layer_index, keys)
            elif position == "self_attn":
                self.set_self_attn_hook(layer_index, keys)
            elif position == "self_attn.q_proj":
                qkv_config["q"] = keys
            elif position == "self_attn.k_proj":
                qkv_config["k"] = keys
            elif position == "self_attn.v_proj":
                qkv_config["v"] = keys
            elif position == "pre_mlp_norm":
                self.set_pre_mlp_norm_hook(layer_index, keys)
            elif position == "mlp":
                self.set_mlp_hook(layer_index, keys)

        self.set_qkv_proj_hooks(layer_index, qkv_config)

        pass

    def set_layers_hook(self, layers_hook_config: list[dict[str, list[str]]]):
        print(layers_hook_config)
        for layer_index, layer_config in enumerate(layers_hook_config):
            self.hooks.layers.append(_LayerHookResult())
            self.set_layer_hook(
                layer_config=layer_config,
                layer_index=layer_index,
            )

    def set_norm_hook(self, keys: list[str]):
        self.hooks.norm = Hook(
            module=self.model.model.norm,
            result_class=NormHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["norm"]["output"],
            keep=keys,
        )

    def set_lm_head_hook(self, keys: list[str]):
        self.hooks.lm_head = Hook(
            module=self.model.lm_head,
            result_class=LMHeadHookResult,
            to_cpu=True,
            output_keys=self.model.io_keys["lm_head"]["output"],
            keep=keys,
        )

    def set_hook(self, position: str, keys: list[str]):
        if "layers[*]" in position:
            for layer_index in range(self.model.config.num_hidden_layers):
                self.set_hook(
                    position=position.replace("layers[*]", f"layers[{layer_index}]"),
                    keys=keys,
                )
        else:
            result_class = get_result_class_from_position(position)
            hook = Hook(
                module=get_deep_attr(self.model, position),
                result_class=self.result_classes[result_class],
                output_keys=self.model.io_keys[result_class]["output"],
                keep=keys,
            )
            self.hooks.append(hook)

    def convert_configs(self, config: dict[str, list[str]]):
        config = self.flatten_layer_configs(config)

    def initialize_hooks(self):
        self.hooks = Hooks()

    def register_hooks(self, config: dict[str, list[str]]):
        self.initialize_hooks()
        layers_hook_config = []

        specific_layer_pattern = re.compile(r"\[(\d+)\]")

        for position, keys in config.items():
            if position == "model.embed_tokens":
                self.set_embed_tokens_hook(keys)
            elif (
                position == "model.embed_positions"
            ):  # for absolute position embeddings
                self.set_embed_positions_hook(keys)

            elif position == "lm_head":
                self.set_lm_head_hook(keys)
            elif position == "model.norm":
                self.set_norm_hook(keys)
            elif "layers" in position:
                if layers_hook_config == []:  # initialize
                    layers_hook_config = [
                        dict() for _ in range(self.model.config.num_hidden_layers)
                    ]
                if "layers[*]" in position:
                    for layer_index in range(len(layers_hook_config)):
                        layers_hook_config[layer_index][
                            position.split("layers[*].")[1]
                        ] = keys
                else:
                    specific_layer_match = specific_layer_pattern.findall(position)
                    if specific_layer_match:
                        layer_index = int(specific_layer_match[0])
                        if layer_index + 1 > len(layers_hook_config):
                            raise ValueError(
                                f"Layer index {layer_index} exceeds number of layers {len(layers_hook_config)}"
                            )
                        if position.endswith(f"layers[{layer_index}]"):
                            layers_hook_config[layer_index]["self"] = keys
                        else:
                            layers_hook_config[layer_index][
                                position.split(f"layers[{layer_index}].")[1]
                            ] = keys
                    else:
                        raise ValueError(f"Invalid layer position: {position}")

        self.set_layers_hook(layers_hook_config)

    def hook_results(self) -> CausalLMObservationBatchResult:
        return self.hooks.get_results()

    @torch.no_grad()
    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
