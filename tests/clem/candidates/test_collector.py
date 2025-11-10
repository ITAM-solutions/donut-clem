"""
File name: test_collector
Author: Fran Moreno
Last Updated: 11/3/2025
Version: 1.0
Description: TOFILL
"""
import unittest

from unittest.mock import patch
from parameterized import parameterized
from datetime import datetime

import clem.candidates.collector as module
from clem.datatypes import DataTypes
from clem.datatypes.currency import Currency

class TestFieldCandidate(unittest.TestCase):

    @parameterized.expand([
        ["000", "foo", DataTypes.str, "foo", str],
        ["001", "123", DataTypes.int, 123, int],
        ["002", "123.45", DataTypes.price, 123.45, float],
        ["003", "EUR", DataTypes.currency, "EUR", Currency],
        ["004", "01/30/2025", DataTypes.date, datetime(year=2025, month=1, day=30), datetime],
    ])
    def test_datatypesConversion_validValues(self, _test_idx, data, dtype, expected_value, expected_type):
        candidate = module.FieldCandidate(
            value=data,
            datatype=dtype
        )

        actual_value = candidate.value
        actual_type = type(actual_value)

        self.assertEqual(expected_value, actual_value)
        self.assertEqual(expected_type, actual_type)

    def test_datatypesConversion_valueRawRemainsIntact(self):
        raw_val = "123"

        candidate = module.FieldCandidate(
            value=raw_val,
            datatype=DataTypes.int
        )

        actual_raw_val = candidate.value_raw
        expected_raw_val = raw_val

        self.assertEqual(expected_raw_val, actual_raw_val)


    @parameterized.expand([
        ["001", "invalid_int", DataTypes.int],
        ["002", "invalid_price", DataTypes.price],
        ["003", "invalid_currency", DataTypes.currency],
        ["004", "invalid_date", DataTypes.date],
    ])
    def test_datatypesConversion_conversionFailedSetsFlag(self, _test_idx, data, dtype):
        candidate = module.FieldCandidate(
            value=data,
            datatype=dtype
        )

        actual_value = candidate.value
        expected_value = data
        conversion_error = candidate.failed_conversion

        self.assertEqual(expected_value, actual_value)
        self.assertTrue(conversion_error)

    def test_emptyValue_isNoneFlagSetToTrue(self):
        candidate = module.FieldCandidate(
            value=None,
            datatype=DataTypes.int
        )

        self.assertTrue(candidate.is_none)


class TestProductCandidate(unittest.TestCase):
    def test_productAsDict(self):
        candidate = module.ProductCandidate(
            name=module.FieldCandidate('foo1', DataTypes.str),
            sku=module.FieldCandidate('foo2', DataTypes.str),
            qty=module.FieldCandidate('foo3', DataTypes.str),
            met=module.FieldCandidate('foo4', DataTypes.str),
            metgr=module.FieldCandidate('foo5', DataTypes.str),
            dfrom=module.FieldCandidate('foo6', DataTypes.str),
            dto=module.FieldCandidate('foo7', DataTypes.str),
            unpr=module.FieldCandidate('foo8', DataTypes.str),
            totpr=module.FieldCandidate('foo9', DataTypes.str),
        )

        expected_dict = {
            "name": "foo1",
            "sku": "foo2",
            "qty": "foo3",
            "met": "foo4",
            "metgr": "foo5",
            "dfrom": "foo6",
            "dto": "foo7",
            "unpr": "foo8",
            "totpr": "foo9"
        }

        self.assertEqual(expected_dict, candidate.dict)

    def test_productIsComplete(self):
        candidate = module.ProductCandidate(
            name=module.FieldCandidate('foo', DataTypes.str),
            sku=module.FieldCandidate('foo', DataTypes.str),
            qty=module.FieldCandidate('foo', DataTypes.str),
            met=module.FieldCandidate('foo', DataTypes.str),
            metgr=module.FieldCandidate('foo', DataTypes.str),
            dfrom=module.FieldCandidate('foo', DataTypes.str),
            dto=module.FieldCandidate('foo', DataTypes.str),
            unpr=module.FieldCandidate('foo', DataTypes.str),
            totpr=module.FieldCandidate('foo', DataTypes.str),
        )

        self.assertTrue(candidate.is_complete)


class TestCandidateCollector(unittest.TestCase):

    @patch('clem.prediction_schema.PredictionSchema')
    @patch('clem.prediction_schema.ProductsSchema')
    def test_addCandidate_expectedDatatypes(self, mock_products_schema, mock_prediction_schema):
        # Define mocks
        mock_prediction_schema.id_ = 'foo1'
        mock_prediction_schema.date_ = 'foo2'
        mock_prediction_schema.po = 'foo3'
        mock_prediction_schema.cur = 'foo4'
        mock_prediction_schema.vendor = 'foo5'
        mock_prediction_schema.corp = 'foo6'

        mock_products_schema.name = ['foo1_1', 'foo1_2']
        mock_products_schema.sku = ['foo2_1', 'foo2_2']
        mock_products_schema.met = ['foo3_1', 'foo3_2']
        mock_products_schema.metgr = ['foo4_1', 'foo4_2']
        mock_products_schema.qty = ['1', '2']
        mock_products_schema.unpr = ['1.1', '2.2']
        mock_products_schema.totpr = ['11.1', '22.2']
        mock_products_schema.dfrom = ['01/31/2025', '02/31/2025']
        mock_products_schema.dto = ['03/31/2025', '04/31/2025']
        mock_products_schema.num_products = 2

        mock_prediction_schema.products = mock_products_schema

        metadata = {'foo': 'bar'}

        candidate_collector = module.CandidateCollector()
        candidate_collector.add(mock_prediction_schema, metadata=metadata)

        from pathlib import Path
        candidate_collector.log(Path('.'))
        # Checks
        invoice_fields_expected_actual_pairs = (
            (candidate_collector.id_, ('foo1', DataTypes.str)),
            (candidate_collector.date_, ('foo2', DataTypes.date)),
            (candidate_collector.po, ('foo3', DataTypes.str)),
            (candidate_collector.cur, ('foo4', DataTypes.currency)),
            (candidate_collector.vendor, ('foo5', DataTypes.str)),
            (candidate_collector.corp, ('foo6', DataTypes.str)),
        )

        for candidate_field, (candidate_value, candidate_dtype) in invoice_fields_expected_actual_pairs:
            self.assertEqual(
                candidate_field,
                [module.FieldCandidate(
                    value=candidate_value,
                    datatype=candidate_dtype,
                    metadata=metadata,
                    score=0
                )])

        product_fields_expected_actual_pairs = (
            ('name', (('foo1_1', 'foo1_2'), DataTypes.str)),
            ('sku', (('foo2_1', 'foo2_2'), DataTypes.str)),
            ('met', (('foo3_1', 'foo3_2'), DataTypes.str)),
            ('metgr', (('foo4_1', 'foo4_2'), DataTypes.str)),
            ('qty', (('1', '2'), DataTypes.int)),
            ('unpr', (('1.1', '2.2'), DataTypes.price)),
            ('totpr', (('11.1', '22.2'), DataTypes.price)),
            ('dfrom', (('01/31/2025', '02/31/2025'), DataTypes.date)),
            ('dto', (('03/31/2025', '04/31/2025'), DataTypes.date)),
        )

        for product_field_name, (candidates_values, candidate_dtype) in product_fields_expected_actual_pairs:
            for idx, product in enumerate(candidate_collector.products):
                self.assertEqual(
                    getattr(product, product_field_name),
                    module.FieldCandidate(
                        value=candidates_values[idx],
                        datatype=candidate_dtype,
                        metadata=metadata,
                        score=0
                    ))

    def test_getBestCandidates_multipleBestCandidatesCanBeRetrieved(self):
        candidate_collector = module.CandidateCollector(id_=[
            module.FieldCandidate(value="foo1", datatype=DataTypes.str, score=10.0),
            module.FieldCandidate(value="foo2", datatype=DataTypes.str, score=10.0),
            module.FieldCandidate(value="foo3", datatype=DataTypes.str, score=8.0),
        ])

        expected_subset = {'id_': ['foo1', 'foo2']}

        actual = candidate_collector.get_best_candidates()
        self.assertEqual(actual, {**actual, **expected_subset})

    def test_getBestCandidates_emptyListsWhenNoCandidates(self):
        candidate_collector = module.CandidateCollector()
        empty_best_candidates = candidate_collector.get_best_candidates()
        for field_name in empty_best_candidates:
            if field_name != "products":
                self.assertEqual(empty_best_candidates[field_name], [])
            else:
                self.assertEqual(empty_best_candidates[field_name], {})

    def test_getBestCandidates_retrievesRawValues(self):
        # Testing with all possible datatypes.
        candidate_collector = module.CandidateCollector(id_=[
            module.FieldCandidate(value="sample-text", datatype=DataTypes.str, score=0.0),
            module.FieldCandidate(value="01/02/25", datatype=DataTypes.date, score=0.0),
            module.FieldCandidate(value="12.34", datatype=DataTypes.price, score=0.0),
            module.FieldCandidate(value="123", datatype=DataTypes.int, score=0.0),
            module.FieldCandidate(value="EUR", datatype=DataTypes.currency, score=0.0),
        ])
        best_candidates = candidate_collector.get_best_candidates()
        expected_values = ['sample-text', '01/02/25', '12.34', '123', 'EUR']
        for expected, actual in zip(expected_values, best_candidates['id_']):
            self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()


