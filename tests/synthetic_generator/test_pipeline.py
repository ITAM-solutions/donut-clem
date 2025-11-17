"""
File name: test_pipeline
Author: Fran Moreno
Last Updated: 11/14/2025
Version: 1.0
Description: TOFILL
"""
import unittest
import sys

from unittest.mock import MagicMock
from parameterized import parameterized

# Mock modules that are not needed for this tests





class TestSyntheticDataGenerator(unittest.TestCase):

    @parameterized.expand([
        ("001", [{'a': 'bar1'}], 'bar1', ['a'], []),
        ("002", [{'a': 'bar1', 'b': 'bar2', 'c': 'bar4'}, {'a': 'bar1', 'b': 'bar3', 'c': 'bar5'}], 'bar1 bar3 bar5', ['b'], [{'a': 'bar1', 'b': 'bar3', 'c': 'bar5'}]),
        ("003", [{'a': 'bar3', 'b': 'bar2'}, {'a': 'bar1', 'b': 'bar4'}], 'bar1 bar2', ['b'], [{'a': 'bar1', 'b': None}]),
        ("004", [{'a': 'bar1', 'b': 'bar2', 'c': 'bar3'}], 'bar1 bar2', [], [{'a': 'bar1', 'b': 'bar2', 'c': None}]),
        ("005", [{'a': 'bar1', 'b': 'bar2'}, {'a': 'bar3', 'b': 'bar2'}], 'bar1 bar2 bar3', [], [{'a': 'bar1', 'b': 'bar2'}, {'a': 'bar3', 'b': 'bar2'}]),
    ])

    def test_cleanNonIncludedProducts_multiple_cases(self, _test_idx, products, text_in_page, shared_fields, expected):

        # Mock modules that are not needed
        sys.modules["synthetic_generator.factory"] = MagicMock()
        sys.modules["synthetic_generator.template_parser"] = MagicMock()
        sys.modules['clem.model'] = MagicMock()

        from synthetic_generator.pipeline import SyntheticDataGenerator

        actual = SyntheticDataGenerator._clean_non_included_products(products, text_in_page, shared_fields)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
