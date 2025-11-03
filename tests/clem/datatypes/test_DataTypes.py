"""
File name: test_DataTypes
Author: Fran Moreno
Last Updated: 11/3/2025
Version: 1.0
Description: TOFILL
"""
import unittest

from parameterized import parameterized

from clem.datatypes import DataTypes


class TestDataTypes(unittest.TestCase):

    @parameterized.expand([
        ["000", "invalid_int", DataTypes.int],
        ["000", "invalid_price", DataTypes.price],
        ["000", "invalid_date", DataTypes.date],
        ["000", "invalid_currency", DataTypes.currency],
    ])
    def test_invalidData_returnsOriginalDataAndErrorFlag(self, _test_idx, data, dtype):
        actual_value, raised_error = dtype(data)
        expected = data

        self.assertEqual(expected, actual_value)
        self.assertTrue(raised_error)

    @parameterized.expand([
        ["000", None , DataTypes.int],
        ["000", None , DataTypes.price],
        ["000", None , DataTypes.date],
        ["000", None , DataTypes.currency],
    ])
    def test_emptyData_returnsNone(self, _test_idx, data, dtype):
        actual_value, raised_error = dtype(data)
        expected = None

        self.assertEqual(expected, actual_value)
        self.assertFalse(raised_error)


if __name__ == '__main__':
    unittest.main()