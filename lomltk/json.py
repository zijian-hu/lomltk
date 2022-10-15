from __future__ import annotations
import json
from json import JSONDecodeError, JSONEncoder
from pathlib import Path
from typing import (
    Any,
    AnyStr,
    IO,
    Optional,
)

from .path import is_file

__all__ = [
    # classes
    "DataEncoder",

    # functions
    "is_json",
    "safe_load",
    "safe_loads",
]


class DataEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return list(obj)

        return super().default(obj)


def safe_load(file: IO[AnyStr], default: Optional[Any] = None, **kwargs: Any) -> Optional[Any]:
    try:
        outputs = json.load(file, **kwargs)
    except JSONDecodeError:
        outputs = default

    return outputs


def safe_loads(string: AnyStr, default: Optional[Any] = None, **kwargs: Any) -> Optional[Any]:
    try:
        outputs = json.loads(string, **kwargs)
    except JSONDecodeError:
        outputs = default

    return outputs


def is_json(file_path: str | Path, encoding: Optional[str] = "utf-8", **kwargs: Any) -> bool:
    if not is_file(file_path):
        return False

    try:
        with open(file_path, mode="r", encoding=encoding, **kwargs) as f:
            json.load(f)
    except JSONDecodeError:
        return False
    else:
        return True
