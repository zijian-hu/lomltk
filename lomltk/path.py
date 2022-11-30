from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from joblib import delayed, Parallel

from .utils import get_progress_bar
from .multiprocessing import resolve_num_workers, tqdm_joblib_context

__all__ = [
    "find_all_files",
    "find_all_files_helper",
    "get_real_path",
    "is_dir",
    "is_file",
    "safe_delete",
]


def get_real_path(path: str | Path) -> Path:
    return Path(os.path.realpath(str(path)))


def is_file(path: str | Path) -> bool:
    return get_real_path(path).is_file()


def is_dir(path: str | Path) -> bool:
    return get_real_path(path).is_dir()


def find_all_files_helper(
        input_dir: str | Path,
        is_valid_func: Callable[[str | Path], bool] = is_file,
        pattern: str = "*",
        is_recursive: bool = True,
        num_workers: int = 1,
        search_desc: str = "Finding files",
        verify_desc: str = "Verifying files",
        show_progress_bar: bool = True
) -> dict[Path, bool]:
    input_dir = Path(input_dir)
    num_workers = resolve_num_workers(num_workers)

    file_paths: list[Path] = [
        p for p in get_progress_bar(
            input_dir.rglob(pattern) if is_recursive else input_dir.glob(pattern),
            desc=search_desc,
            is_tqdm=show_progress_bar
        ) if is_file(p)
    ]

    with tqdm_joblib_context(
            get_progress_bar(file_paths, desc=verify_desc, is_tqdm=show_progress_bar)
    ) as progress_bar:
        is_valid_list = Parallel(n_jobs=num_workers)(
            delayed(is_valid_func)(file_path)
            for file_path in progress_bar
        )

    assert len(file_paths) == len(is_valid_list)
    return {k: v for k, v in zip(file_paths, is_valid_list)}


def find_all_files(
        input_dir: str | Path,
        helper_func: Callable[..., dict[Path, bool]] = find_all_files_helper,
        is_valid_func: Callable[[str | Path], bool] = is_file,
        pattern: str = "*",
        is_recursive: bool = True,
        num_workers: int = 1,
        search_desc: str = "Finding files",
        verify_desc: str = "Verifying files",
        show_progress_bar: bool = True
) -> list[Path]:
    file_path_validity_dict = helper_func(
        input_dir=input_dir,
        is_valid_func=is_valid_func,
        pattern=pattern,
        is_recursive=is_recursive,
        num_workers=num_workers,
        search_desc=search_desc,
        verify_desc=verify_desc,
        show_progress_bar=show_progress_bar
    )

    return [k for k, v in file_path_validity_dict.items() if v]


def safe_delete(path: str | Path) -> None:
    try:
        os.remove(str(path))
    except FileNotFoundError:
        return
