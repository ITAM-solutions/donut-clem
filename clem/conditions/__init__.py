"""
File name: __init__
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from clem.conditions.PagePositionCondition import PagePositionCondition
from clem.conditions.DataTypesCondition import DataTypesCondition
from clem.conditions.ValueRepetitionCondition import ValueRepetitionCondition

conditions = [
    PagePositionCondition,
    DataTypesCondition,
    ValueRepetitionCondition
]
