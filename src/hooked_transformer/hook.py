import contextlib
from typing import Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from hooked_transformer.utils import get_argument_names
from hooked_transformer.utils.logger import init_logging

from .models.base.abstract import AbstractBatchResult

logger = init_logging(__name__)


class Hook:
    """Base class for hooks."""

    hook: RemovableHandle
    result: AbstractBatchResult
    result_class: type[AbstractBatchResult]
    to_cpu: bool
    positional_args_keys: list[str]
    output_keys: list[str]
    with_kwargs: bool
    keep: list[str]

    def __init__(
        self,
        module: nn.Module,
        result_class: AbstractBatchResult,
        to_cpu: bool = True,
        output_keys: list[str] = None,
        keep: None | list[str] = None,
        pre_hook: bool = False,
        post_process: Callable | None = None,
    ) -> None:
        # Register a forward hook
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)

        # Register keys for positional args and output
        input_keys = list(get_argument_names(module.forward))
        input_keys.remove("self")
        self.input_keys = input_keys
        self.output_keys = output_keys
        logger.warn_once(
            f"Hook at {module}\n"
            f"Inputs: {['in_' + k for k in self.input_keys]}\n"
            f"Outputs: {[f'out_{k}' for k in self.output_keys]}"
        )
        self.keep = keep or []

        self.result = None
        self.result_class = result_class
        self.to_cpu = to_cpu
        self.post_process = post_process

    def hook_fn(
        self, module: nn.Module, args: tuple, kwargs: dict, output: tuple
    ) -> None:
        """Forward hook function to capture inputs and outputs.

        Parameters
        ----------
        module : nn.Module
            Module with hook
        args : tuple
            Position arguments passed to the module
        kwargs : dict
            Keyword arguments passed to the module
        output : Tensor
            Output of the module
        """
        hook_result = {}
        pos = -1
        for pos, arg in enumerate(args):
            key = "in_" + self.input_keys[pos]
            if key in self.keep:
                hook_result[key] = (
                    arg.cpu().clone()
                    if self.to_cpu and isinstance(arg, torch.Tensor)
                    else arg
                )

        for key in self.input_keys[pos + 1 :]:
            _key = "in_" + key
            if _key in self.keep:
                hook_result[_key] = (
                    kwargs[key].cpu().clone()
                    if self.to_cpu and isinstance(kwargs[key], torch.Tensor)
                    else kwargs[key]
                )

        # Add output to the hook result
        if self.output_keys is not None:
            if isinstance(output, tuple):
                assert len(output) == len(self.output_keys), (
                    f"Output tuple length {len(output)} does not match expected "
                    f"length {len(self.output_keys)}"
                )
                for k, v in zip(self.output_keys, output):
                    assert k not in hook_result, f"Key {k} already exists in kwargs."
                    key = "out_" + k
                    if key in self.keep:
                        hook_result[key] = (
                            v.cpu().clone()
                            if self.to_cpu and isinstance(v, torch.Tensor)
                            else v
                        )
            else:
                assert len(self.output_keys) == 1, (
                    f"Output keys length {len(self.output_keys)} does not match expected "
                    f"length 1 for single output."
                )
                key = "out_" + self.output_keys[0]
                if key in self.keep:
                    hook_result[key] = (
                        output.cpu().clone()
                        if self.to_cpu and isinstance(output, torch.Tensor)
                        else output
                    )

        if self.post_process is None:
            self.result = self.result_class(**hook_result)
        else:
            self.result = self.result_class(**self.post_process(hook_result))

    def __repr__(self, indent=1) -> str:
        """String representation of the hook."""
        tab = "\t" * indent
        return (
            f"{self.__class__.__name__}\n"
            f"{tab}result: {self.result.__repr__(indent=indent + 1)}"
        )

    #     self.result = self.result_class(**kwargs)

    # def pre_hook_fn(self, module, args, kwargs) -> None:
    #     """Hook function for pre-hook."""
    #     kwargs = {
    #         k: v.cpu().clone() if (isinstance(v, torch.Tensor) and self.to_cpu) else v
    #         for k, v in zip(self.positional_args_keys, args)
    #     }

    #     self.result = self.result_class(**kwargs)

    def remove(self):
        """Remove the hook."""
        self.hook.remove()

    @classmethod
    @contextlib.contextmanager
    def context(cls, hooks: "list[Hook]"):
        """Context manager to use the hook."""
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()
