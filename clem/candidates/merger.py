"""
File name: merger
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from typing import List

from clem.candidates.collector import CandidateCollector, ProductCandidate
from clem.candidates.product_merging import find_and_combine_partial_products
from clem.conditions import conditions, ValueRepetitionCondition


class CandidateSelector:
    def __init__(self, candidates: CandidateCollector):
        self._candidates = candidates

    @classmethod
    def merge(cls, candidates: CandidateCollector) -> CandidateCollector:

        # Apply general conditions
        for condition in conditions:
            condition.apply(candidates)

        # Reduce products if needed
        candidates.products = find_and_combine_partial_products(candidates.products)

        return candidates



if __name__ == '__main__':
    from clem.candidates.collector import FieldCandidate
    from clem.datatypes import DataTypes

    id_candidates = [
        FieldCandidate(value='foo1', datatype=DataTypes.str),
        FieldCandidate(value='foo1', datatype=DataTypes.str),
        FieldCandidate(value='foo1', datatype=DataTypes.str),
        FieldCandidate(value='foo3', datatype=DataTypes.str),
        FieldCandidate(value='foo2', datatype=DataTypes.str),
        FieldCandidate(value='foo5', datatype=DataTypes.str),
    ]

    candidates = CandidateCollector(id_=id_candidates)

    scored_candidates = CandidateSelector.merge(candidates)
    print(scored_candidates.get_best_candidates())
