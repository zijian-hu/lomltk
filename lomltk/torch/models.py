from __future__ import annotations
from contextlib import ContextDecorator, contextmanager
from typing import Any, Dict, ContextManager

from torch.nn import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel

__all__ = [
    "consume_prefix_in_state_dict_if_present",
    "get_module_size",
    "set_model_mode",
    "unwrap_ddp",
]


@contextmanager
def set_model_mode(model: Module, mode: bool) -> ContextManager[None] | ContextDecorator:
    """

    Args:
        model: input model
        mode: True turns on set training mode; False turns on evaluation mode

    Returns:

    """
    prev_mode = model.training
    model.train(mode=mode)

    try:
        yield
    finally:
        model.train(prev_mode)


def consume_prefix_in_state_dict_if_present(
        state_dict: Dict[str, Any],
        prefix: str
) -> None:
    r"""Copied from https://github.com/pytorch/pytorch/blob/57f039b41f940af4f02718fdf967cca1f713d759/torch/nn/modules/utils.py#L43

    Strip the prefix in state_dict in place, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def get_module_size(module: Module) -> int:
    return sum(p.numel() for p in module.parameters())


def unwrap_ddp(module: Module) -> Module:
    if isinstance(module, (DataParallel, DistributedDataParallel)):
        return module.module
    else:
        return module
