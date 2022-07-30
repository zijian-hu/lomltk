from typing import Dict, Union

import torch
from torch import Tensor
import numpy as np
import random

RandomState = Dict[str, Union[Tensor, tuple]]

__all__ = [
    # types
    "RandomState",

    # functions
    "get_random_state",
    "set_random_state",
]


def get_random_state() -> RandomState:
    return dict(
        torch=torch.random.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
    )


def set_random_state(random_state: RandomState) -> None:
    torch.random.set_rng_state(random_state["torch"].cpu())
    np.random.set_state(random_state["numpy"])
    random.setstate(random_state["python"])
