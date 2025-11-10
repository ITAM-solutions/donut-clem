"""
File name: test_merger
Author: Fran Moreno
Last Updated: 11/10/2025
Version: 1.0
Description: TOFILL
"""
import unittest

from clem.candidates.merger import CandidateSelector
from clem.candidates.collector import FieldCandidate, CandidateCollector
from clem.datatypes import DataTypes

class TestCandidateSelector(unittest.TestCase):

    def test_merge_invoiceFields_realCase1(self):
        candidates_values = [
            # id_candidates
            [
                FieldCandidate(value="0210595557", datatype=DataTypes.str, metadata={"page": 0, "section": 1}),
                FieldCandidate(value="210595557", datatype=DataTypes.str, metadata={"page": 1, "section": 0}),
                FieldCandidate(value="210595557", datatype=DataTypes.str, metadata={"page": 2, "section": 0}),
            ],
            # date_candidates
            [
                FieldCandidate(value="23-JUL-2024", datatype=DataTypes.date, metadata={"page": 0, "section": 1}),
                FieldCandidate(value="23-JUL-2024", datatype=DataTypes.date, metadata={"page": 1, "section": 0}),
                FieldCandidate(value="23-JUL-2024", datatype=DataTypes.date, metadata={"page": 2, "section": 0}),
            ],
            # po_candidates
            [
                FieldCandidate(value="NL001269318B01", datatype=DataTypes.str, metadata={"page": 0, "section": 1}),
                FieldCandidate(value=None, datatype=DataTypes.str, metadata={"page": 1}),
                FieldCandidate(value=None, datatype=DataTypes.str, metadata={"page": 2}),
            ],
            # cur_candidates
            [
                FieldCandidate(value="EUR", datatype=DataTypes.currency, metadata={"page": 0, "section": 2}),
                FieldCandidate(value="21.00%", datatype=DataTypes.currency, metadata={"page": 1, "section": 3}),
                FieldCandidate(value=None, datatype=DataTypes.currency, metadata={"page": 2}),
            ],
            # vendor_candidates
            [
                FieldCandidate(value="INSIGHT ENTERPRISES NETHERLANDS B.V.", datatype=DataTypes.str,
                    metadata={"page": 0, "section": 0}),
                FieldCandidate(value="INSIGHT", datatype=DataTypes.str, metadata={"page": 1, "section": 0}),
                FieldCandidate(value=None, datatype=DataTypes.str, metadata={"page": 2, }),
            ],
            # corp_candidates
            [
                FieldCandidate(value=None, datatype=DataTypes.str, metadata={"page": 0}),
                FieldCandidate(value=None, datatype=DataTypes.str, metadata={"page": 1}),
                FieldCandidate(value=None, datatype=DataTypes.str, metadata={"page": 2}),
            ]
        ]

        candidates = CandidateCollector(
            id_=candidates_values[0],
            date_=candidates_values[1],
            po=candidates_values[2],
            cur=candidates_values[3],
            vendor=candidates_values[4],
            corp=candidates_values[5]
        )

        candidates_with_scores = CandidateSelector.merge(candidates)

        expected_final_candidates = [
            # id_candidates
            [
                FieldCandidate(value="0210595557", datatype=DataTypes.str, score=11.0),
                FieldCandidate(value="210595557", datatype=DataTypes.str, score=3.0),
                FieldCandidate(value="210595557", datatype=DataTypes.str, score=3.0),
            ],
            # date_candidates
            [
                FieldCandidate(value="23-JUL-2024", datatype=DataTypes.date, score=5.0),
                FieldCandidate(value="23-JUL-2024", datatype=DataTypes.date, score=13.0),
                FieldCandidate(value="23-JUL-2024", datatype=DataTypes.date, score=8.0),
            ],
            # po_candidates
            [
                FieldCandidate(value="NL001269318B01", datatype=DataTypes.str, score=11.0),
                FieldCandidate(value=None, datatype=DataTypes.str, score=3.333),
                FieldCandidate(value=None, datatype=DataTypes.str, score=0.0),
            ],
            # cur_candidates
            [
                FieldCandidate(value="EUR", datatype=DataTypes.currency, score=10.5),
                FieldCandidate(value="21.00%", datatype=DataTypes.currency, score=0.5),
                FieldCandidate(value=None, datatype=DataTypes.currency, score=6.667),
            ],
            # vendor_candidates
            [
                FieldCandidate(value="INSIGHT ENTERPRISES NETHERLANDS B.V.", datatype=DataTypes.str, score=8.5),
                FieldCandidate(value="INSIGHT", datatype=DataTypes.str, score=5.5),
                FieldCandidate(value=None, datatype=DataTypes.str, score=1.667),
            ],
            # corp_candidates
            [
                FieldCandidate(value=None, datatype=DataTypes.str, score=8.0),
                FieldCandidate(value=None, datatype=DataTypes.str, score=0.0),
                FieldCandidate(value=None, datatype=DataTypes.str, score=0.0),
            ]
        ]

        # Check final candidates scores
        for actual_candidates, expected_candidates in zip(candidates_values, expected_final_candidates):
            for actual_candidate, expected_candidate in zip(actual_candidates, expected_candidates):
                self.assertEqual(actual_candidate.value, expected_candidate.value)
                self.assertEqual(actual_candidate.score, expected_candidate.score)

        # Check best candidates
        expected_best_candidates = {
            "id_": "0210595557",
            "date_": "23-JUL-2024",
            "po": "NL001269318B01",
            "cur": "EUR",
            "vendor": "INSIGHT ENTERPRISES NETHERLANDS B.V.",
            "corp": None,
            "products": {}
        }
        actual_best_candidates = candidates_with_scores.get_best_candidates()
        self.assertDictEqual(expected_best_candidates, actual_best_candidates)


if __name__ == '__main__':
    unittest.main()