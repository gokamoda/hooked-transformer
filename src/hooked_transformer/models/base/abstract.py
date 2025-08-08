from dataclasses import dataclass, fields, make_dataclass

import torch

from hooked_transformer.utils.logger import init_logging

logger = init_logging(__name__)


@dataclass
class AbstractResult:
    def __repr__(self, indent=1) -> str:
        tab = "\t" * indent
        msg = self.__class__.__name__ + "\n"
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                msg += f"{tab}{k}: {v.shape}\n"
            elif isinstance(v, AbstractResult):
                msg += f"{tab}{k}: {v.__class__.__name__}\n"
            elif isinstance(v, list):
                msg += f"{tab}{k}: {v[0].__class__.__name__} x {len(v)}\n"
            elif isinstance(v, tuple):
                msg += f"{tab}{k}: ("
                for item in v:
                    if isinstance(item, torch.Tensor):
                        msg += f"{item.shape}, "
                    elif isinstance(item, AbstractResult):
                        msg += f"{item.__class__.__name__}, "
                    else:
                        msg += f"{item}, "
                msg += ")\n"
            else:
                msg += f"{tab}{k}: {v}\n"
        return msg

    def __init__(self, **kwargs):
        # Get the field names from the dataclass
        field_names = {f.name for f in fields(self.__class__)}
        for key, value in kwargs.items():
            if key in field_names:
                setattr(self, key, value)
        # Handle ignored/unexpected keys
        ignored_keys = set(kwargs) - field_names
        if ignored_keys:
            logger.warn_once(f"Ignored unexpected keys: {ignored_keys}")

    @classmethod
    def init_all(cls, **kwargs):
        for key, value in kwargs.items():
            setattr(cls, key, value)


@dataclass(repr=False, init=False)
class AbstractBatchResult(AbstractResult):
    def unbatch(self) -> list[AbstractResult]:
        for k, v in self.__dict__.items():
            if isinstance(v, AbstractBatchResult):
                setattr(self, k, v.unbatch())
            elif isinstance(v, (torch.Tensor | list)):
                continue
            else:
                raise ValueError(f"Unexpected type: {type(v)}")

        results = []
        new_class_name = self.__class__.__name__.replace("Batch", "")
        new_class_fields = [(f.name, f.type) for f in fields(self)]
        new_class = make_dataclass(
            new_class_name,
            fields=new_class_fields,
            bases=(AbstractResult,),
            repr=False,
            init=False,
        )

        for i in range(self.get_batch_size()):
            results.append(new_class(**{k: v[i] for k, v in self.__dict__.items()}))

        return results

    def get_batch_size(self) -> int:
        """Get the batch size."""
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
            elif isinstance(v, AbstractBatchResult):
                return v.get_batch_size()
