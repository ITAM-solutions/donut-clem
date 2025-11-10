"""
File name: conversion
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description:
"""
from dataclasses import dataclass
from typing import Any, Callable

from clem.datatypes.errors import ConversionError
from clem.datatypes.price import string_to_price
from clem.datatypes.date import string_to_date
from clem.datatypes.currency import string_to_currency
from clem.datatypes.integer import string_to_int


@dataclass
class ConversionResult:
    value: Any
    failed: bool


class DataTypes:
    """
    Utility class to convert data to a specific datatype. Every conversion is made "safely", meaning that,
    in case that it is not possible to convert, the value will be returned in its original format.
    """
    """ TYPES DEFINITION """

    @classmethod
    def str(cls, data: Any):
        return cls._secure_conversion(data, str)

    @classmethod
    def int(cls, data: Any):
        return cls._secure_conversion(data, string_to_int)

    @classmethod
    def price(cls, data: Any):
        return cls._secure_conversion(data, string_to_price)

    @classmethod
    def date(cls, data: Any):
        return cls._secure_conversion(data, string_to_date)

    @classmethod
    def currency(cls, data: Any):
        return cls._secure_conversion(data, string_to_currency)

    """ END OF TYPES DEFINITION """

    @classmethod
    def as_type(cls, value, datatype):
        return ConversionResult(*datatype(value))

    @staticmethod
    def _secure_conversion(data: Any, conversion_function: Callable):
        if data is None:
            return data, False

        try:
            res, error = conversion_function(data), False
        except ConversionError:
            res, error = data, True

        return res, error


if __name__ == '__main__':
    print(DataTypes)