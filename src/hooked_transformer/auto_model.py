from transformers import AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig

from hooked_transformer.models.alias_config import ALIAS_CONFIG
from hooked_transformer.utils import get_deep_attr


def add_alias_to_attrs(model: AutoModelForCausalLM):
    if type(model).__name__ in ALIAS_CONFIG:
        alias_dict = ALIAS_CONFIG[type(model).__name__]["alias"]

        for alias_path, target in alias_dict:
            # Regarding alias path
            alias_parts = alias_path.split(".")
            if len(alias_parts) == 1:
                # Top level: set on type(model)
                cls = type(model)
                alias_name = alias_parts[0]
                owner_obj = model
            else:
                # Deeper level: walk through model to reach correct sub-object
                parent_path = ".".join(alias_parts[:-1])
                owner_obj = get_deep_attr(model, parent_path)
                cls = type(owner_obj)
                alias_name = alias_parts[-1]

            # Check if already defined
            if hasattr(cls, alias_name):
                print(f"Alias {alias_name} already exists in {cls.__name__}, skipping.")
                continue

            # Regarding target path
            if isinstance(target, str):
                target_parts = target.split(".")
                assert len(alias_parts) == len(target_parts), (
                    "Alias and target paths must have the same number of parts"
                )

                # Determine where to set the property
                if len(target_parts) == 1:
                    alias_target = target_parts[0]
                else:
                    target_parent_path = ".".join(target_parts[:-1])
                    assert get_deep_attr(model, target_parent_path) is owner_obj, (
                        "Target path must match owner object"
                    )
                    alias_target = target_parts[-1]

                # Define property
                setattr(
                    cls,
                    alias_name,
                    property(
                        lambda self, _target=alias_target: get_deep_attr(self, _target)
                    ),
                )
                print(
                    f"Added alias {alias_name} for {alias_path} -> {alias_target} in {cls.__name__}"
                )
            elif callable(target):
                setattr(
                    cls,
                    alias_name,
                    property(lambda self, _target=target: _target(self)),
                )

    return model  # type: ignore[return]


def add_io_key_info(model: AutoModelForCausalLM):
    if type(model).__name__ in ALIAS_CONFIG:
        io_keys = ALIAS_CONFIG[type(model).__name__]["io_keys"]
        model.io_keys = io_keys
        return model


def add_alias_to_config(config: PretrainedConfig):
    if config.architectures[0] in ["GPT2Config"]:
        config.num_hidden_layers = config.n_layer
        config.num_attention_heads = config.n_head

    config.num_query_heads = config.num_attention_heads
    config.num_key_value_heads = (
        config.num_key_value_heads
        if hasattr(config, "num_key_value_heads")
        else config.num_attention_heads
    )
    return config


class AutoModelForCausalLMWithAliases(AutoModelForCausalLM):
    @staticmethod
    def from_pretrained(
        model_name_or_path: str, **kwargs
    ) -> "AutoModelForCausalLMWithAliases":
        """
        Returns the appropriate universal causal language model class based on the model name or path.

        Args:
            model_name_or_path (str): The name or path of the model.

        Returns:
            class: The appropriate universal causal language model class.
        """
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, **kwargs
        )  # type: ignore
        model = add_alias_to_attrs(model)
        model = add_io_key_info(model)

        return model
