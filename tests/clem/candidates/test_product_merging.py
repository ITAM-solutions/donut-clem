"""
File name: test_product_merging
Author: Fran Moreno
Last Updated: 11/7/2025
Version: 1.0
Description: TOFILL
"""
import unittest

from parameterized import parameterized
from unittest.mock import Mock

import clem.candidates.product_merging as module
from clem.candidates.collector import ProductCandidate, FieldCandidate
from clem.datatypes import DataTypes

FIELD_NAMES = {'name', 'sku', 'qty', 'met', 'metgr', 'dfrom', 'dto', 'unpr', 'totpr'}


def mock_field_candidate(value):
    return Mock(spec=FieldCandidate, value=value, is_none=(value is None))


def mock_product_candidate(values_dict: dict):
    return ProductCandidate(**{
        fname: mock_field_candidate(val) for fname, val in values_dict.items()
    })


class TestFindAndCombinePartialProducts(unittest.TestCase):


    @parameterized.expand([
        (
            # Consecutive partial products, with no common fields. Can be combined.
            "000",
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                    qty=FieldCandidate('b1', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a2', DataTypes.str),
                ),
                ProductCandidate(
                    qty=FieldCandidate('b2', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a3', DataTypes.str),
                    qty=FieldCandidate('b3', DataTypes.str),
                ),
            ],
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                    qty=FieldCandidate('b1', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a2', DataTypes.str),
                    qty=FieldCandidate('b2', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a3', DataTypes.str),
                    qty=FieldCandidate('b3', DataTypes.str),
                ),
            ]
        ),
        (
            # Consecutive partial products, but both fill common field. Cannot be combined.
            "001",
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                    qty=FieldCandidate('b1', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a2', DataTypes.str),
                    qty=FieldCandidate('b2', DataTypes.str),
                ),
                ProductCandidate(
                    qty=FieldCandidate('b3', DataTypes.str),
                    totpr=FieldCandidate('c3', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a3', DataTypes.str),
                    qty=FieldCandidate('b3', DataTypes.str),
                ),
            ],
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                    qty=FieldCandidate('b1', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a2', DataTypes.str),
                    qty=FieldCandidate('b2', DataTypes.str),
                ),
                ProductCandidate(
                    qty=FieldCandidate('b3', DataTypes.str),
                    totpr=FieldCandidate('c3', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a3', DataTypes.str),
                    qty=FieldCandidate('b3', DataTypes.str),
                ),
            ],
        ),
        (
            # Multiple partial products pairs that can be combined into one single product.
            "002",
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                ),
                ProductCandidate(
                    qty=FieldCandidate('b1', DataTypes.str),
                ),
                ProductCandidate(
                    totpr=FieldCandidate('c1', DataTypes.str),
                ),
                ProductCandidate(
                    unpr=FieldCandidate('d1', DataTypes.str),
                ),
            ],
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                    qty=FieldCandidate('b1', DataTypes.str),
                    totpr=FieldCandidate('c1', DataTypes.str),
                    unpr=FieldCandidate('d1', DataTypes.str),
                ),
            ],
        ),
        (
            # Just one product, cannot merge anything.
            "003",
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                ),
            ],
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                ),
            ],
        ),
        (
            # Empty list of products, cannot merge anything.
            "004",
            [
            ],
            [
            ],
        ),
        (
            # Multiple partial products, but they are not consecutive, so cannot merge.
            "005",
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                    qty=FieldCandidate('b1', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a2', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a3', DataTypes.str),
                    qty=FieldCandidate('b3', DataTypes.str),
                ),
                ProductCandidate(
                    qty=FieldCandidate('b4', DataTypes.str),
                ),
            ],
            [
                ProductCandidate(
                    name=FieldCandidate('a1', DataTypes.str),
                    qty=FieldCandidate('b1', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a2', DataTypes.str),
                ),
                ProductCandidate(
                    name=FieldCandidate('a3', DataTypes.str),
                    qty=FieldCandidate('b3', DataTypes.str),
                ),
                ProductCandidate(
                    qty=FieldCandidate('b4', DataTypes.str),
                ),
            ],
        ),

    ])
    def test_twoConsecutivePartialProducts_shouldMergeThem(self, _test_idx, products, expected):
        actual_merge = module.find_and_combine_partial_products(products)

        self.assertEqual(len(expected), len(actual_merge))

        for p1, p2 in zip(expected, actual_merge):
            for field_name in FIELD_NAMES:
                f1 = getattr(p1, field_name)
                f2 = getattr(p2, field_name)
                self.assertEqual(f1.value, f2.value)


class TestArePartials(unittest.TestCase):

    def test_bothProductsAreFullyFilled_returnsFalse(self):
        product1_fields = {
            'name': 'a',
            'sku': 'b',
            'qty': 'c',
            'met': 'd',
            'metgr': 'e',
            'dfrom': 'f',
            'dto': 'g',
            'unpr': 'h',
            'totpr': 'i'
        }

        product2_fields = {
            'name': 'a2',
            'sku': 'b2',
            'qty': 'c2',
            'met': 'd2',
            'metgr': 'e2',
            'dfrom': 'f2',
            'dto': 'g2',
            'unpr': 'h2',
            'totpr': 'i2'
        }

        product1 = mock_product_candidate(product1_fields)
        product2 = mock_product_candidate(product2_fields)

        self.assertFalse(module.are_partials(product1, product2))

    def test_noIntersectingValuesBetweenProducts_returnsTrue(self):
        product1_fields = {
            'name': 'a',
            'sku': 'b',
            'qty': None,
            'met': None,
            'metgr': 'e',
            'dfrom': None,
            'dto': 'g',
            'unpr': 'h',
            'totpr': None
        }

        product2_fields = {
            'name': None,
            'sku': None,
            'qty': 'c2',
            'met': 'd2',
            'metgr': None,
            'dfrom': 'f2',
            'dto': None,
            'unpr': None,
            'totpr': 'i2'
        }

        product1 = mock_product_candidate(product1_fields)
        product2 = mock_product_candidate(product2_fields)

        self.assertTrue(module.are_partials(product1, product2))

    def test_noIntersectionWithEmptyFieldsInBoth_returnsTrue(self):
        product1_fields = {
            'name': 'a',
            'sku': None,
            'qty': None,
            'met': None,
            'metgr': None,
            'dfrom': None,
            'dto': None,
            'unpr': None,
            'totpr': None
        }

        product2_fields = {
            'name': None,
            'sku': None,
            'qty': 'c2',
            'met': None,
            'metgr': None,
            'dfrom': None,
            'dto': None,
            'unpr': None,
            'totpr': None
        }

        product1 = mock_product_candidate(product1_fields)
        product2 = mock_product_candidate(product2_fields)

        self.assertTrue(module.are_partials(product1, product2))


class TestMergeTwoPartialProducts(unittest.TestCase):
    @parameterized.expand([
        ("000", {'name': 'a'}, {'qty': 'b'}),
        ("001", {'name': 'a', 'sku': 'e'}, {'qty': 'b', 'unpr': 'c', 'totpr': 'd'}),
        ("002", {'name': 'a', 'sku': 'b', 'qty': 'c', 'met': 'd', 'metgr': 'e'}, {'dfrom': 'f', 'dto': 'g', 'unpr': 'h', 'totpr': 'i'}),
    ])
    def test_multipleCases(self, _test_idx, p1_fields: dict, p2_fields: dict):
        p1_empty_fields = set(FIELD_NAMES).difference(set(p1_fields))
        p2_empty_fields = set(FIELD_NAMES).difference(set(p2_fields))

        p1_all_fields = p1_fields.copy()
        p2_all_fields = p2_fields.copy()

        p1_all_fields.update({f: None for f in p1_empty_fields})
        p2_all_fields.update({f: None for f in p2_empty_fields})

        product1 = mock_product_candidate(p1_all_fields)
        product2 = mock_product_candidate(p2_all_fields)

        expected_fields = p1_fields.copy()
        expected_fields.update(p2_fields)
        p_expected_empty_fields = set(FIELD_NAMES).difference(set(expected_fields))
        expected_fields.update({f: None for f in p_expected_empty_fields})
        product_expected = mock_product_candidate(expected_fields)

        p_actual = module.merge_two_partial_products(product1, product2)

        self.assertEqual(product_expected.filled_fields.keys(), p_actual.filled_fields.keys())



if __name__ == '__main__':
    unittest.main()