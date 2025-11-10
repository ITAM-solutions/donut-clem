"""
File name: __init__
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from clem.conditions.DataTypesCondition import DataTypesCondition
from clem.conditions.PageNumberCondition import PageNumberCondition
from clem.conditions.PagePositionCondition import PagePositionCondition
from clem.conditions.ValueRepetitionCondition import ValueRepetitionCondition

conditions = [
    DataTypesCondition,
    PageNumberCondition,
    PagePositionCondition,
    ValueRepetitionCondition,  # Must always be the last one, as it reduces candidates to uniques.
]
