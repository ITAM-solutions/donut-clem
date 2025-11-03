"""
File name: currency
Author: Fran Moreno
Last Updated: 11/3/2025
Version: 1.0
Description: TOFILL
"""
from .errors import ConversionError


class Currency(str):
    allowed = {"EUR", "Euro", "USD", "CHF", "Dollar", "Pound", "€", "$", "£"}

    def __new__(cls, value):
        if value not in cls.allowed:
            raise ValueError(f"{value} is not a valid {cls.__name__}.")
        return str.__new__(cls, value)


def string_to_currency(data: str):
    try:
        return Currency(data)
    except ValueError:
        raise ConversionError