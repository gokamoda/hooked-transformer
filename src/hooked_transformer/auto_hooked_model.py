from .auto_model import AutoModelForCausalLMWithAliases
from .models.base.hooked_model import HookedModelForCausalLM
from .models.hooked_models import HOOKED_MODELS
from .utils.logger import init_logging

logger = init_logging(__name__)


class AutoHookedModelForCausalLM:
    @staticmethod
    def from_pretrained(
        model_name_or_path: str, **kwargs
    ) -> "AutoHookedModelForCausalLM":
        model = AutoModelForCausalLMWithAliases.from_pretrained(
            model_name_or_path=model_name_or_path, **kwargs
        )

        if model.config.architectures[0] in HOOKED_MODELS:
            hooked_model_class = HOOKED_MODELS[model.config.architectures[0]]

            return hooked_model_class(model)
        else:
            logger.warning(
                f"Model {model.architectures[0]} is not supported for hooking. Returning standard model."
            )
            return HookedModelForCausalLM(model)
