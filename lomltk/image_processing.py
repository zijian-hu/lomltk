from __future__ import annotations
from pathlib import Path
from typing import (
    Any,
    Optional,
    Sequence,
    Union,
)

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage

from .path import is_file

ImageType = Union[PILImage, np.ndarray]

__all__ = [
    "cv_to_pil",
    "draw_text",
    "pil_open",
    "pil_to_cv",
    "thumbnail",
    "to_cv",
    "to_pil",
]


def pil_open(filename: str | Path, color_mode: Optional[str] = None) -> Image:
    image = Image.open(str(filename))

    if color_mode is not None:
        image = image.convert(color_mode)

    return image


def cv_to_pil(image: np.ndarray) -> PILImage:
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))


def pil_to_cv(image: PILImage) -> np.ndarray:
    return cv.cvtColor(np.asarray(image.convert("RGB")), cv.COLOR_RGB2BGR)


def to_pil(image: ImageType, inplace: bool = False) -> PILImage:
    """
    Convert image to PIL.Image.Image.

    Args:
        image: input image
        inplace: if True, will return the same image if no conversion is needed; else, return a copy

    Returns:

    """
    if isinstance(image, PILImage):
        if inplace:
            return image
        else:
            return image.copy()

    elif isinstance(image, np.ndarray):
        return cv_to_pil(image)

    else:
        raise NotImplementedError(f"{type(image)} is not a supported image type")


def to_cv(image: ImageType, inplace: bool = False) -> np.ndarray:
    """
    Convert image to np.ndarray.

    Args:
        image: input image
        inplace: if True, will return the same image if no conversion is needed; else, return a copy

    Returns:

    """
    if isinstance(image, PILImage):
        return pil_to_cv(image)

    elif isinstance(image, np.ndarray):
        if inplace:
            return image
        else:
            return image.copy()

    else:
        raise NotImplementedError(f"{type(image)} is not a supported image type")


def thumbnail(image: ImageType, max_size: int | float, inplace: bool = False) -> PILImage:
    """
    Limit the image size to be at most (max_size, max_size) while keeping the original ratio

    Args:
        image: input image
        max_size: max size
        inplace: if True, will resize the original image; else, return a copy

    Returns:

    """
    image = to_pil(image, inplace=inplace)
    if not np.isinf(max_size):
        image.thumbnail(size=(max_size, max_size))

    return image


def draw_text(
        image: ImageType,
        text: str,
        coordinates: tuple[int, int],
        font_path: Optional[str | Path] = None,
        font_size: int = 32,
        anchor: Optional[str] = None,
        color: Sequence[int] | str = "red",
        background_color: Optional[Sequence[int] | str] = None,
        **kwargs: Any
) -> ImageType:
    """

    Args:
        image: input image
        text: text to draw
        coordinates: (x, y) coordinates of the input image
        font_path: .ttf or .ttc file
        font_size: font size
        anchor: The text anchor alignment. See https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html#text-anchors
        color: color string or RGB color tuple
        background_color: color string or RGB color tuple for background. If None, no background

    Returns:

    """
    assert font_path is None or is_file(font_path), f"\"{font_path}\" is not a valid font file path"
    output_image = to_pil(image)

    draw = ImageDraw.Draw(output_image)

    # see https://stackoverflow.com/a/50854463
    font = ImageFont.truetype(font_path, font_size) if font_path is not None else draw.getfont()

    if background_color is not None:
        # see https://stackoverflow.com/a/18869779
        x, y = coordinates
        box = font.getbbox(text, anchor=anchor)

        # draw background
        draw.rectangle((box[0] + x, box[1] + y, box[2] + x, box[3] + y), fill=background_color)

    draw.text(
        coordinates,
        text,
        font=font,
        fill=color,
        anchor=anchor,
        **kwargs
    )

    if isinstance(image, np.ndarray):
        output_image = to_cv(output_image)

    return output_image
