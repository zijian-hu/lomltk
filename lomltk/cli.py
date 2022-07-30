import math
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

__all__ = [
    "get_file_type_str",
    "get_optional_file_type_str",
    "get_optional_type",
    "int_or_inf",
    "optional_str",
]


def int_or_inf(input_str: str) -> Union[int, float]:
    result = float(input_str)

    if not math.isinf(result):
        result = int(input_str)

    return result


def get_optional_type(func: Callable[[str], Any]) -> Callable[[str], Optional[Any]]:
    def wrapper(input_str: str) -> Optional[Any]:
        if input_str.lower().strip() in {"none", "null"}:
            return None
        else:
            return func(input_str)

    return wrapper


def optional_str(input_str: str) -> Optional[str]:
    return get_optional_type(lambda x: x)(input_str)


def get_file_type_str(*suffix: str) -> Callable[[str], str]:
    suffix = tuple(s.lower().strip() for s in suffix)
    assert all(s.startswith(".") for s in suffix), "suffix must begin with \".\""

    def func(input_str: str) -> str:
        input_str = input_str.strip()
        assert input_str.lower().endswith(suffix), f"\"{input_str}\" does not end with \"{suffix}\""
        return input_str

    return func


def get_optional_file_type_str(*suffix: str) -> Callable[[str], Optional[str]]:
    return get_optional_type(get_file_type_str(*suffix))
