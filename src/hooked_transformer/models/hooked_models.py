from .base.hooked_model import HookedModelForCausalLM
from .gpt2.hooked_model import HookedGPT2LMHeadModel

HOOKED_MODELS = {
    "GPT2LMHeadModel": HookedGPT2LMHeadModel,
    "LlamaForCausalLM": HookedModelForCausalLM,
    # Assuming LlamaForCausalLM is similar to GPT2
}
