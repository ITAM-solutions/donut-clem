"""
File name: test_ValueRepetitionCondition
Author: Fran Moreno
Last Updated: 11/5/2025
Version: 1.0
Description: TOFILL
"""
import unittest
import uuid
from unittest.mock import MagicMock
from unittest.mock import patch
from parameterized import parameterized

from clem.conditions.ValueRepetitionCondition import ValueRepetitionCondition


class TestValueRepetitionCondition(unittest.TestCase):

    # @parameterized.expand([
    #     ("000", {'a': [1, 2, 3], 'b': [4, 5], 'c': [2, 6]}, {'a': [1, 2, 3, 6], 'b': [4, 5]}),
    #     ("001", {'a': [1, 2], 'b': [3, 4], 'c': [5]}, {'a': [1, 2], 'b': [3, 4], 'c': [5]}),
    #     ("002", {'a': [1, 2, 3], 'b': [2, 5], 'c': [3, 6]}, {'a': [1, 2, 3, 5, 6]}),
    #     ("003", {'a': [1], 'b': [1], 'c': [1]}, {'a': [1]}),
    #     ("004", {'a': [1, 2], 'b': [3, 4], 'c': [2, 3]}, {'a': [1, 2, 3, 4]}),
    #     ("005", {'a': [1, 2, 3], 'b': [4, 5], 'c': [2, 4], 'd': [5, 2]}, {'a': [1, 2, 3, 4, 5]}),
    #     ("006", {'a': [1, 2, 3], 'b': [4, 5], 'c': [1, 2], 'd': [5, 6], 'e': [2, 6]}, {'a': [1, 2, 3, 4, 5, 6]})
    # ])
    # def test_mergeUniquesWithCommonCandidates_expectedMerge(self, _test_idx, original_map, expected):
    #     actual = ValueRepetitionCondition._merge_uniques_with_common_candidates(original_map)
    #     self.assertDictEqual(expected, actual)
    #
    # @parameterized.expand([
    #     ("000", {"a": [0, 1, 2], "b": [3, 4]}, ['c1', 'c2', 'c3', 'c4', 'c5'], [3.0, 3.0, 3.0, 2.0, 2.0])
    # ])
    # def test_assignScoresForRepetition_expectedScoresAssigned(self, _test_idx, uniques_map_vals, candidate_vals, expected):
    #     candidates = [
    #         MagicMock(value_clean=val, _uuid=uuid.uuid4(), score=0.0)
    #         for val in candidate_vals
    #     ]
    #     uniques_map = {k: [candidates[idx] for idx in indexes] for k, indexes in uniques_map_vals.items()}
    #     ValueRepetitionCondition._assign_scores_for_repetition(uniques_map, candidates)
    #
    #     for idx, candidate in enumerate(candidates):
    #         self.assertEqual(expected[idx], candidate.score)

    @parameterized.expand([
        ("000", 'val1', 'val2', 'foo')
    ])
    def test_isAMatch(self, _test_idx, val1, val2, expected):
        actual = ValueRepetitionCondition._is_a_match(val1, val2)
        print(actual)


if __name__ == '__main__':
    unittest.main()
