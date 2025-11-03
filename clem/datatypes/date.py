"""
File name: date
Author: Fran Moreno
Last Updated: 11/3/2025
Version: 1.0
Description: TOFILL
"""
import datetime

from dateutil.parser import parse as date_parse, ParserError

from .errors import ConversionError


def parse_date(date_str: str, expected_format: str) -> str:
    date_parsed = date_parse(date_str, fuzzy=True, default=datetime.datetime(year=1, month=1, day=1))
    if date_parsed.year == 1:
        expected_format = expected_format.replace("%Y", "")
    return date_parsed.strftime(expected_format)


def string_to_date(date_str: str, expected_format: str = "%m/%d/%Y") -> datetime.datetime:
    """
    Transforms a date format into the expected one. If the given string is not a date, it returns the original value.
    # TODO: try to cover cases like this: 17-MRT-2023 (Month is in Dutch acronym).

    UNIT TESTS: test/test_backend/clem_fields/datatypes/test_date.py -> TestGetFormattedDate

    :param date_str: value to be transformed.
    :param expected_format: expected format for the date.
    :return: transformed date.
    """
    try:
        date_parsed = parse_date(date_str, expected_format)
        datetime_obj = datetime.datetime.strptime(date_parsed, expected_format)
        return datetime_obj
    except (ParserError, ValueError):
        raise ConversionError
