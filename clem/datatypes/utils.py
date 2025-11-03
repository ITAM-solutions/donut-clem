"""
File name: utils
Author: Fran Moreno
Last Updated: 11/3/2025
Version: 1.0
Description: TOFILL
"""
from string import digits


def remove_alpha_and_special_chars(string: str, allowed_special_chars: str = ",. -/_") -> str:
    """
    Removes any special character or letters from a numeric string, keeping only digits and separators.
    Some special characters are allowed, like spaces, punctuations, hyphens, slashes and underscores, so in case of
    being a date, the string is not modified.
    In case that the string does not contain any digit, it is returned as it is.

    :param string: Original string containing the price.
    :param allowed_special_chars: string containing all special characters that are allowed to remain in output.
    :return: String containing only digits and separators.
    """
    if not any(digit in string for digit in digits):
        return string
    return "".join(list(filter(lambda ch: ch in digits + allowed_special_chars, string))).strip()