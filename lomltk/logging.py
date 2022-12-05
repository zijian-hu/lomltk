from __future__ import annotations
from typing import Optional, TextIO
from typing_extensions import Literal

import logging
from logging import (
    FileHandler,
    Formatter,
    getLogger,
    Logger,
    StreamHandler,
)
from pathlib import Path

__all__ = [
    "get_logger",
]


def get_logger(
        name: Optional[str] = None,
        level: int = logging.INFO,
        propagate: Optional[bool] = None,
        stream: Optional[TextIO] = None,
        file_path: Optional[str | Path] = None,
        file_mode: str = "a",
        encoding: Optional[str] = None,
        format_str: Optional[str] = None,
        date_format: Optional[str] = None,
        format_style: Literal["%", "{", "$"] = "%",
        is_create_parents: bool = False
) -> Logger:
    """Create or retrieve logger based on name and configurations

    Args:
        name: logger name.
        level: log level.
        propagate: if propagate to parent loggers.
        stream: logging stream. If not None, will remove all existing handlers.
        file_path: logging file path. If not None, will remove all existing handlers.
        file_mode: logging file mode.
        encoding: logging file encoding.
        format_str: format string. If not None, will be registered to all handlers.
        date_format: date format string. If not None, will be registered to all handlers.
        format_style: formatter style.
        is_create_parents: if True and file_path is not None, will create parent directories
            of the file_path.

    Returns:
        logger

    """
    logger = getLogger(name)
    logger.setLevel(level)

    handlers = []

    if isinstance(propagate, bool):
        logger.propagate = propagate

    if stream is not None:
        handlers.append(StreamHandler(stream))

    if file_path is not None:
        if is_create_parents:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        handlers.append(FileHandler(filename=str(file_path), mode=file_mode, encoding=encoding))

    if len(handlers) > 0:
        # remove old handlers
        if logger.hasHandlers():
            for h in list(logger.handlers):
                logger.removeHandler(h)

        # register new handlers
        for h in handlers:
            logger.addHandler(h)

    if format_str is not None or date_format is not None:
        formatter = Formatter(format_str, datefmt=date_format, style=format_style)

        for h in list(logger.handlers):
            h.setFormatter(formatter)

    return logger
