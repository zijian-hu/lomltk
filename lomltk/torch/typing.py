from __future__ import annotations
from typing import TypeVar, Union

from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

__all__ = [
    # type
    "DDPModuleType",
    "ModuleType",
]

# see https://stackoverflow.com/a/61737894
ModuleType = TypeVar("ModuleType", bound=Module)
DDPModuleType = Union[ModuleType, DistributedDataParallel]
