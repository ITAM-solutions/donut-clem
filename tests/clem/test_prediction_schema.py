"""
File name: test_prediction_schema
Author: Fran Moreno
Last Updated: 10/29/2025
Version: 1.0
Description: TOFILL
"""
import unittest

from parameterized import parameterized
from unittest.mock import patch

from clem.prediction_schema import ProductsSchema, PredictionSchema


class TestProductsSchema(unittest.TestCase):
    def test_singleStringsAsValues_passesValidation(self):
        data = {
            'name': 'foo1',
            'sku': 'foo2',
            'met': 'foo3',
            'metgr': 'foo4',
            'qty': 'foo5',
            'unpr': 'foo6',
            'totpr': 'foo7',
            'dfrom': 'foo8',
            'dto': 'foo9'
        }

        # Assertion
        ProductsSchema(**data)

    def test_listsOfStringsAsValues_passesValidation(self):
        data = {
            'name': ['foo1', 'another'],
            'sku': ['foo2', 'another'],
            'met': ['foo3', 'another'],
            'metgr': ['foo4', 'another'],
            'qty': ['foo5', 'another'],
            'unpr': ['foo6', 'another'],
            'totpr': ['foo7', 'another'],
            'dfrom': ['foo8', 'another'],
            'dto': ['foo9', 'another'],
        }

        # Assertion
        ProductsSchema(**data)

    def test_listWithStringsAndNone_passesValidation(self):
        data = {
            'name': ['foo1', None],
            'sku': ['foo2', None],
            'met': ['foo3', None],
            'metgr': ['foo4', None],
            'qty': ['foo5', None],
            'unpr': ['foo6', None],
            'totpr': ['foo7', None],
            'dfrom': ['foo8', None],
            'dto': ['foo9', None],
        }

        # Assertion
        ProductsSchema(**data)

    def test_noneValues_passesValidation(self):
        data = {
            'name': None,
            'sku': None,
            'met': None,
            'metgr': None,
            'qty': None,
            'unpr': None,
            'totpr': None,
            'dfrom': None,
            'dto': None,
        }

        # Assertion
        ProductsSchema(**data)

    @parameterized.expand([
        ["000", {"a": "1", "b": "2"}, {"a": ["1"], "b": ["2"]}],
        ["001", {"a": ["1", "2"], "b": "3"}, {"a": ["1", "2"], "b": ["3"]}],
        ["002", {"a": None, "b": "3"}, {"a": [None], "b": ["3"]}],
        ["003", {"a": None, "b": None}, {"a": [None], "b": [None]}],
        ["004", {"a": "None"}, {"a": [None]}]
    ])
    def test_convertValuesToList(self, _test_idx, data, expected):
        ProductsSchema._convert_values_to_list(data)
        self.assertEqual(data, expected)

    @parameterized.expand([
        ["000", {"a": ["1", "2"], "b": ["3"]}, {"a": ["1", "2"], "b": ["3", None]}],
        ["001", {"a": ["1"], "b": ["2", "3", "4"]}, {"a": ["1", None, None], "b": ["2", "3", "4"]}],
        ["002", {"a": [None], "b": [None, None]}, {"a": [None, None], "b": [None, None]}],
        ["003", {"a": [], "b": ["1", "2"]}, {"a": [None, None], "b": ["1", "2"]}],
        ["004", {"a": ["1", "2"], "b": []}, {"a": ["1", "2"], "b": [None, None]}],
        ["005", {"a": [], "b": []}, {"a": [], "b": []}]
    ])
    def test_makeListsEqualInLength(self, _test_idx, data, expected):
        ProductsSchema._make_lists_equal_in_length(data)
        self.assertEqual(data, expected)


if __name__ == '__main__':
    unittest.main()
