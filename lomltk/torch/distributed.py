from __future__ import annotations
from functools import wraps
import os
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
)
from typing_extensions import Literal, ParamSpec

from torch import distributed, Tensor
from torch.distributed import ProcessGroup, ReduceOp

T = TypeVar("T")
DT = TypeVar("DT")
P = ParamSpec("P")

GroupType = Optional[ProcessGroup]
ReduceOpStrType = Literal["mean", "average", "sum"]

__all__ = [
    "all_reduce",
    "barrier",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "one_rank_only",
    "safe_distributed",
]


def is_distributed() -> bool:
    return distributed.is_available() and distributed.is_initialized()


def safe_distributed(default: DT = None) -> Callable[P, T | DT]:
    """
    Decorator that returns default value when not in distributed mode

    Args:
        default: default value

    Returns:

    """

    def _safe_distributed(func: Callable[P, T]) -> Callable[P, T | DT]:
        @wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> T | DT:
            if is_distributed():
                return func(*args, **kwargs)
            else:
                return default

        return wrapped_func

    return _safe_distributed


@safe_distributed(default=None)
def barrier() -> Optional[Any]:
    return distributed.barrier()


@safe_distributed(default=1)
def get_world_size(group: GroupType = None) -> int:
    return distributed.get_world_size(group)


@safe_distributed(default=0)
def get_local_rank() -> int:
    return os.environ.get("LOCAL_RANK", 0)


@safe_distributed(default=0)
def get_rank(group: GroupType = None) -> int:
    return distributed.get_rank(group)


def get_reduce_operation():
    pass


def all_reduce(
        tensor: Tensor,
        op: ReduceOp | Literal["mean", "average", "sum"] = ReduceOp.SUM,
        group: GroupType = None,
        inplace: bool = False
) -> Tensor:
    outputs = tensor if inplace else tensor.clone()

    if is_distributed():
        distributed.all_reduce(outputs, op=op, group=group)

    return outputs


def one_rank_only(rank: int = 0, synchronize: bool = False) -> Callable[P, Optional[T]]:
    """

    Args:
        rank: target rank to execute the function.
        synchronize: if True, will synchronize with a barrier.

    Returns:

    """

    def _one_rank_only(func: Callable[P, T]) -> Callable[P, Optional[T]]:
        @wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
            output = None
            try:
                if get_rank() == rank:
                    output = func(*args, **kwargs)
            finally:
                if synchronize:
                    barrier()
            return output

        return wrapped_func

    return _one_rank_only