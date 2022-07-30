from __future__ import annotations
import itertools
import time
from functools import wraps
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
)

from tqdm import tqdm

from .typing import ExceptionType, verify_exception_type

__all__ = [
    "flatten",
    "get_progress_bar",
    "retry",
    "sleep",
    "split_chunks",
    "unwrap_tqdm",
]


def split_chunks(inputs: Sequence[Any], chunk_size: int) -> list[list[Any]]:
    assert chunk_size > 0
    return [list(inputs[i:i + chunk_size]) for i in range(0, len(inputs), chunk_size)]


def retry(
        max_retry: int = 0,
        backoff_factor: float = 0,
        exception_type: ExceptionType = Exception
) -> Callable[..., Any]:
    assert max_retry >= 0
    assert verify_exception_type(exception_type)

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for i in range(max_retry + 1):
                try:
                    result = func(*args, **kwargs)
                    return result
                except exception_type as e:
                    if i >= max_retry:
                        raise e

                    else:
                        sleep(backoff_factor)

                    # continue the loop
                    continue

        return wrapper

    return decorate


def flatten(lists: Iterable[Iterable[Any]]) -> list[Any]:
    return list(itertools.chain.from_iterable(lists))


def unwrap_tqdm(iterable: Iterable | tqdm) -> Optional[Iterable]:
    if isinstance(iterable, tqdm):
        iterable = iterable.iterable

    return iterable


def get_progress_bar(
        iterable: Iterable,
        is_tqdm: bool = True,
        desc: Optional[str] = None,
        total: Optional[int | float] = None,
        **kwargs: Any
) -> tqdm | Optional[Iterable]:
    if "disable" in kwargs:
        raise ValueError("is_tqdm is mutually exclusive with disable")

    if not is_tqdm:
        return unwrap_tqdm(iterable)

    elif not isinstance(iterable, tqdm):
        # if is_tqdm and iterable is not tqdm
        return tqdm(iterable, desc=desc, total=total, **kwargs)

    else:
        return iterable


def sleep(seconds: float) -> None:
    """
    Sleep for `seconds` seconds if `seconds` > 0 else do nothing

    Args:
        seconds: number of seconds to sleep

    Returns:

    """
    if seconds > 0:
        time.sleep(seconds)
