"""
File name: types
Author: Fran Moreno
Last Updated: 11/3/2025
Version: 1.0
Description: Defines special types to be used through the code.
"""




class Price(str):
    def __new__(cls, value):
        if not cls._is_price(value):
            raise ValueError(f"{value} is not a valid {cls.__name__}.")
        return str.__new__(cls, value)

    @staticmethod
    def _is_price(value: str) -> bool:
        # TODO implement
        return True
