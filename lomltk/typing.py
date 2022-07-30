from typing import (
    Any,
    Tuple,
    Type,
    Union,
)

ExceptionType = Union[Type[Exception], Tuple[Type[Exception], ...]]

__all__ = [
    # types
    "ExceptionType",

    # functions
    "verify_exception_type",
]


def verify_exception_type(exception_type: Any) -> bool:
    """
    Check if exception_type is Type[Exception] | Tuple[Type[Exception], ...]

    Args:
        exception_type: a type object

    Returns:

    """
    if isinstance(exception_type, type):
        return issubclass(exception_type, Exception)

    elif isinstance(exception_type, tuple):
        return all(isinstance(t, type) and issubclass(t, Exception) for t in exception_type)

    else:
        return False
