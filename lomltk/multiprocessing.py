from __future__ import annotations
from typing import ContextManager, Iterable, TypeVar

from contextlib import ContextDecorator, contextmanager
import os

import joblib
from joblib import Parallel
from tqdm import tqdm

T = TypeVar("T")

__all__ = [
    "cpu_count",
    "resolve_num_workers",
    "tqdm_joblib_context",
]


def cpu_count(only_physical_cores: bool = False) -> int:
    """
    Get number of CPU cores on the current machine. If environment variable `MY_CPU_LIMIT` is set,
        will choose the lower value between `joblib.cpu_count` and `$MY_CPU_LIMIT`

    Args:
        only_physical_cores:

    Returns:

    """
    num_cpus = joblib.cpu_count(only_physical_cores)

    return min(num_cpus, int(os.environ.get("MY_CPU_LIMIT", num_cpus)))


def resolve_num_workers(num_workers: int = -1) -> int:
    if num_workers >= 0:
        return max(1, min(num_workers, cpu_count()))

    elif num_workers == -1:
        # use all cpus
        return cpu_count()

    else:
        # num_workers < -1
        return num_workers + 1 + cpu_count() if abs(num_workers) < cpu_count() else 1


@contextmanager
def tqdm_joblib_context(tqdm_object: tqdm | Iterable[T]) -> ContextManager[tqdm | Iterable[T]] | ContextDecorator:
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.
    See https://stackoverflow.com/a/61689175 and https://stackoverflow.com/a/58936697 for detail

    Args:
        tqdm_object: tqdm object

    Returns:

    """
    if isinstance(tqdm_object, tqdm):
        def tqdm_print_progress(self: Parallel) -> None:
            if self.n_completed_tasks > tqdm_object.n:
                n_completed = self.n_completed_tasks - tqdm_object.n
                tqdm_object.update(n=n_completed)

        old_print_progress = joblib.parallel.Parallel.print_progress
        joblib.parallel.Parallel.print_progress = tqdm_print_progress

        try:
            yield tqdm_object
        finally:
            joblib.parallel.Parallel.print_progress = old_print_progress
            tqdm_object.close()
    else:
        yield tqdm_object
