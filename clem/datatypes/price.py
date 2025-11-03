"""
File name: price
Author: Fran Moreno
Last Updated: 11/3/2025
Version: 1.0
Description: TOFILL
"""
"""
File name: currency
Author: Fran Moreno
Last Updated: 19 Feb 2025
Version: MVP
Description: Utility functions to handle price formats. It handles multiple currency symbols and locales.
"""

from typing import Optional, Union, List, Tuple

from .errors import ConversionError
from .utils import remove_alpha_and_special_chars


def _find_string_occurrences(string: str, sub_string: str) -> List[int]:
    """
    Finds all occurrences of `sub_string` within `string`.

    UNIT TESTS: test/test_backend/helpers/test_currency.py -> TestFindStringOccurrences

    :param string: string to search in.
    :param sub_string: substring to search for.
    :return: list of positions where `sub_string` is found in `string`.
    """
    if not sub_string:
        return []

    if type(string) is not str or type(sub_string) is not str:
        return []

    return [i for i in range(len(string)) if string.startswith(sub_string, i)]


def _detect_separators(price_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a string containing a price, detects the role for the two possible separators (dot and comma) to
    define the locale it is using.
    If there is no thousand and/or decimal separator, it returns None for the corresponding separator.

    :param price_str: String containing the price to check.
    :return: tuple with the detected thousand separator (first position) and decimal separator (second position).
    """
    dot_idx = _find_string_occurrences(price_str, '.')
    comma_idx = _find_string_occurrences(price_str, ',')

    # No separators found
    if not dot_idx and not comma_idx:
        return None, None

    # Only dot separator found
    if dot_idx and not comma_idx:
        # If just one dot, and less than three digits after it, it is a decimal separator.
        if len(dot_idx) == 1 and len(price_str[dot_idx[0]:-1]) <= 2:
            return None, "."

        # Else, it is a thousand separator.
        return ".", None

    # Only comma separator found
    if not dot_idx and comma_idx:
        # If just one comma, and less than three digits after it, it is a decimal separator.
        if len(comma_idx) == 1 and len(price_str[comma_idx[0]:-1]) <= 2:
            return None, ","

        # Else, it is a thousand separator.
        return ",", None

    # Both separators found
    if len(dot_idx) > 1 and len(comma_idx) > 1:  # Multiple dots and commas
        raise ValueError
    elif len(dot_idx) > 1 and len(comma_idx) == 1:  # Multiple dots, just one comma
        if max(dot_idx) > comma_idx[0]:
            raise ValueError
        return ".", ","
    elif len(dot_idx) == 1 and len(comma_idx) > 1:  # Multiple commas, just one dot
        if max(comma_idx) > dot_idx[0]:
            raise ValueError
        return ",", "."
    else:  # One comma and one dot. Depends on their relative position.
        if dot_idx[0] < comma_idx[0]:
            return ".", ","
        else:
            return ",", "."


def string_to_price(price_str: str) -> Union[float, str]:
    """
    Converts a string containing a price to a float number. It handles both cases where it is a pure number
    contained in a string, or it comes together with a currency sign.

    If an error occurs during conversion, the original value is returned.

    :param price_str: string containing the price.
    :return: corresponding number in float format.
    """
    try:
        price_str = remove_alpha_and_special_chars(price_str)
        thousands_sep, decimal_sep = _detect_separators(price_str)

        if thousands_sep:
            price_str = price_str.replace(thousands_sep, "")
        if decimal_sep:
            price_str = price_str.replace(decimal_sep, ".")

        return float(price_str)

    except (TypeError, ValueError):
        raise ConversionError
