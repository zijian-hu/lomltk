from __future__ import annotations
from typing import (
    ContextManager,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from contextlib import ContextDecorator, contextmanager

import numpy as np
import torch
from torch import Size, Tensor
from torch.nn import functional as F, Module

D = TypeVar("D", bound=dict)
TensorInputType = Union[Sequence[Union[int, float]], np.ndarray, Tensor]

__all__ = [
    "multi_hot",
    "random_choice",
    "to_device",
    "to_tensor",
    "toggle_grad",
]


def to_device(
        data: list | tuple | D | Tensor | Module,
        device: torch.device | str,
        non_blocking: bool = True
) -> list | tuple | D | Tensor | Module:
    if isinstance(data, list):
        data = [to_device(d, device) for d in data]

    elif isinstance(data, tuple):
        data = tuple(to_device(d, device) for d in data)

    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = to_device(data[k], device)

    elif isinstance(data, (Tensor, Module)):
        data = data.to(device, non_blocking=non_blocking)

    return data


def to_tensor(inputs: int | float | TensorInputType) -> Tensor:
    if isinstance(inputs, Tensor):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return torch.from_numpy(inputs)
    else:
        return torch.tensor(inputs)


def random_choice(
        choices: TensorInputType,
        size: Size | Sequence[int],
        replacement: bool = True,
        weights: Optional[TensorInputType] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
) -> Tensor:
    """
    See https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/14

    Args:
        choices: Candidates to draw from.
        size: Output shape.
        replacement: Whether to draw with replacement or not.
        weights: The probabilities associated with each candidate.
        dtype: Output data type.
        device: Device of output data.

    Returns:

    """
    choices = to_tensor(choices).to(dtype=dtype, device=device)
    if len(choices.shape) != 1:
        raise ValueError("choices must be a 1D tensor/numpy array/list/tuple")

    if weights is None:
        weights = torch.ones_like(choices, dtype=torch.float)
    else:
        weights = to_tensor(weights).to(dtype=torch.float, device=device)
    if len(weights.shape) != 1:
        raise ValueError("weights must be a 1D tensor/numpy array/list/tuple")

    if weights.shape != choices.shape:
        raise RuntimeError("choices and weights must have the same shape")

    indices = torch.multinomial(
        weights,
        num_samples=np.prod(size).item(),
        replacement=replacement
    )

    outputs: Tensor = torch.index_select(choices, 0, indices)

    return outputs.reshape(size).contiguous()


@contextmanager
def toggle_grad(is_track_grad: bool) -> ContextManager[None] | ContextDecorator:
    if not is_track_grad:
        with torch.no_grad():
            yield
    else:
        yield


def multi_hot(
        inputs: Tensor | Sequence[Tensor],
        num_classes: int = -1
) -> Tensor:
    if isinstance(inputs, Sequence):
        if len(inputs) == 0:
            raise RuntimeError(f"Only non-empty list is accepted.")

        if any(not isinstance(t, Tensor) for t in inputs):
            raise TypeError(f"Only list of torch.Tensor is supported.")

        if any(t.shape[:-1] != inputs[0].shape[:-1] for t in inputs):
            raise RuntimeError(
                f"All tensors must have the same shape except for the last dimension."
            )

        return torch.stack([
            multi_hot(t, num_classes=num_classes) for t in inputs
        ])

    elif isinstance(inputs, Tensor):
        outputs = F.one_hot(inputs, num_classes)

        if len(inputs.shape) >= 1:
            outputs = outputs.sum(dim=max(0, len(inputs.shape) + 1 - 2)).clamp(max=1)

        return outputs

    else:
        raise TypeError(f"Expecting torch.Tensor or list[torch.Tensor], but got {type(inputs)}")
