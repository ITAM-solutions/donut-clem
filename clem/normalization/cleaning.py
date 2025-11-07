"""
File name: cleaning
Author: Fran Moreno
Last Updated: 11/4/2025
Version: 1.0
Description: TOFILL
"""
from typing import Optional
import re


def clean_value(value: Optional[str]) -> str:
    """

    :param value:
    :return:
    """
    if value is None:
        return ''

    clean = value.strip().lower()
    clean = re.sub(r'[^\w\s]', '', clean)
    clean = re.sub(r'[^a-z0-9]', '', clean)
    return clean
