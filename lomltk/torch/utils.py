from __future__ import annotations
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Size, Tensor

TensorInputType = Union[Sequence[Union[int, float]], np.ndarray, Tensor]

__all__ = [
    "random_choice",
    "to_tensor",
]


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
