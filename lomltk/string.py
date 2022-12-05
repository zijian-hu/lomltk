from typing import Mapping

import re

__all__ = [
    "replace",
]


def replace(text: str, replacements: Mapping[str, str]) -> str:
    # see https://stackoverflow.com/a/6117124
    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    return pattern.sub(lambda m: replacements.get(m.group(0), m.group(0)), text)
