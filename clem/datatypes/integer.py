"""
File name: integer
Author: Fran Moreno
Last Updated: 11/3/2025
Version: 1.0
Description: TOFILL
"""
from .errors import ConversionError
from .utils import remove_alpha_and_special_chars


def string_to_int(data: str) -> int:
    try:
        return int(remove_alpha_and_special_chars(data, allowed_special_chars=",._- "))
    except (TypeError, ValueError):
        raise ConversionError
