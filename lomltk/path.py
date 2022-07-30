from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from joblib import delayed, Parallel

from .utils import get_progress_bar
from .multiprocessing import resolve_num_workers, tqdm_joblib_context

__all__ = [
    "get_real_path",
    "is_file",
    "is_dir",
    "find_all_files_helper",
    "find_all_files",
    "safe_delete",
]


def get_real_path(file_path: str | Path) -> Path:
    return Path(os.path.realpath(str(file_path)))


def is_file(file_path: str | Path) -> bool:
    return get_real_path(file_path).is_file()


def is_dir(file_path: str | Path) -> bool:
    return get_real_path(file_path).is_dir()


def find_all_files_helper(
        input_dir: str | Path,
        is_valid_func: Callable[[str | Path], bool] = is_file,
        pattern: str = "*",
        is_recursive: bool = True,
        num_workers: int = 1
) -> tuple[list[Path], list[bool]]:
    input_dir = Path(input_dir)
    num_workers = resolve_num_workers(num_workers)

    file_paths: list[Path] = [
        p for p in get_progress_bar(
            input_dir.rglob(pattern) if is_recursive else input_dir.glob(pattern),
            desc=f"Finding files w/ pattern \"{pattern}\""
        ) if is_file(p)
    ]

    with tqdm_joblib_context(
            get_progress_bar(file_paths, desc="Verifying files")
    ) as progress_bar:
        is_valid_list = Parallel(n_jobs=num_workers)(
            delayed(is_valid_func)(file_path)
            for file_path in progress_bar
        )

    assert len(file_paths) == len(is_valid_list)
    return file_paths, is_valid_list


def find_all_files(
        input_dir: str | Path,
        helper_func: Callable[..., tuple[list[Path], list[bool]]] = find_all_files_helper,
        is_valid_func: Callable[[str | Path], bool] = is_file,
        pattern: str = "*",
        is_recursive: bool = True,
        num_workers: int = 1
) -> list[Path]:
    file_paths, is_valid_list = helper_func(
        input_dir=input_dir,
        is_valid_func=is_valid_func,
        pattern=pattern,
        is_recursive=is_recursive,
        num_workers=num_workers
    )

    return [file_path for file_path, is_valid in zip(file_paths, is_valid_list) if is_valid]


def safe_delete(path: str | Path) -> None:
    try:
        os.remove(str(path))
    except FileNotFoundError:
        return
