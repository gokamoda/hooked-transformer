from .gpt2 import GPT2LMHEAD_CONFIG
from .llama import LLAMA_CONFIG

ALIAS_CONFIG = {
    "GPT2LMHeadModel": GPT2LMHEAD_CONFIG,
    "LlamaForCausalLM": LLAMA_CONFIG,
}
