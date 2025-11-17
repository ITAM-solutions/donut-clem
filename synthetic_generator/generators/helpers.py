"""
File name: helpers
Author: Fran Moreno
Last Updated: 9/3/2025
Version: 1.0
Description: TOFILL
"""
import re

from numpy.random import choice as np_choice
from typing import Dict, Tuple

from synthetic_generator.utils.config import DocxConfig
from synthetic_generator.utils.translator import translate
from synthetic_generator.utils.datatypes import FieldType

def unpack_tuples_to_dict(data: dict):
    """
    This function solves a common data structure conversion based on the code structure.

    Expected input format:
    {
        'key1': ('val1', 'lbl1'),
        'key2': ('val2', 'lbl2'),
        ...
    }
    Output format:
    {
        'key1': {'lbl': 'lbl1', 'val': 'val1'},
        'key2': {'lbl': 'lbl2', 'val': 'val2'},
    }
    :param data:
    :return:
    """
    return {
        id_: {"lbl": label, "val": value}
        for id_, (value, label) in data.items()
    }


def is_blank(value: str, p: float):
    """
    This helper method will return the given value or make it empty based on the given probability `p`.
    The probability `p` relates to the empty (blank) value.

    :param value: value to operate with.
    :param p: probability of becoming an empty (blank) value.
    :return: resulting decision.
    """
    blank = ' '
    return str(np_choice([value, blank], p=[1-p, p]))


def normalize_string(value: str, dtype: FieldType, is_blank_p: float, config: DocxConfig):
    value = is_blank(value, is_blank_p)

    if dtype not in (
        FieldType.Currency,
        FieldType.InvoiceID,
        FieldType.PONumber,
        FieldType.IDLong,
        FieldType.IDShort,
        FieldType.IDHuge,
        FieldType.UUID,
        FieldType.IDTax,
        FieldType.IDBank,
        FieldType.Price,
        FieldType.CompanyName,
        FieldType.Number,
        FieldType.Percentage,
        FieldType.PhoneNumber,
        FieldType.Email,
        FieldType.PersonName,
        FieldType.Signature,
        FieldType.Address,
        FieldType.BankName,
        FieldType.IBAN,
        FieldType.Swift,
        FieldType.Index,
        FieldType.ProductName,
        FieldType.ProductDesc,
        FieldType.ProductDescShort,
        FieldType.Street,
        FieldType.City,
        FieldType.Country,
        FieldType.PostalCode,
        FieldType.Url,
        # Keep expanding if FieldType is updated with new types.
    ):
        value = translate(value, config.lang)
    return value

def has_blank_value(data: dict, p: float):
    for k, v in data.items():
        data[k] = str(np_choice([v, ''], p=[1-p, p]))
    return data


def has_blank_value_tuple(data: Dict[str, Tuple[str, str]], p: float) -> Dict[str, Tuple[str, str]]:
    for k, v in data.items():
        data[k] = (str(np_choice([v[0], ''], p=[1-p, p])), v[1])
    return data


def date_format_to_regex(date_format: str) -> str:
    regex_maps = {
        "%Y": r"\d{4}",
        "%m": r"\d{2}",
        "%d": r"\d{2}",
        "%B": r"[A-Za-z]+",
        "%b": r"[A-Za-z]+",
    }

    for format_id, regex_map in regex_maps.items():
        date_format = date_format.replace(format_id, regex_map)

    return date_format


def find_dates_in_string(text: str, date_format: str):
    regex_pattern = date_format_to_regex(date_format)

    matches = re.findall(regex_pattern, text)
    return matches