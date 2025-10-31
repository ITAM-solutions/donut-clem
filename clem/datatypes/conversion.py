"""
File name: conversion
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TODO define conversions. Add utility conversion functions to another module in this same subpackage.
"""

class DataTypes:
    """
    Utility class to convert data to a specific datatype. Every conversion is made "safely", meaning that,
    in case that it is not possible to convert, the value will be returned in its original format.
    """
    @classmethod
    def as_type(cls, value, datatype):
        return datatype(value)

    @staticmethod
    def str(data):
        try:
            return str(data)
        except ValueError:
            return data

    @staticmethod
    def int(data):
        try:
            return int(data)
        except ValueError:
            return data

    @staticmethod
    def float(data):
        pass

    @staticmethod
    def date(data):
        pass

    @staticmethod
    def currency(data):
        pass
