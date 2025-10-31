"""
File name: merger
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from typing import List

from clem.candidates.collector import CandidateCollector, ProductCandidate
from clem.conditions import conditions


class CandidateSelector:
    def __init__(self, candidates: CandidateCollector):
        self._candidates = candidates

    @classmethod
    def merge(cls, candidates: CandidateCollector):

        # Select best invoice field values
        for condition in conditions:
            condition.apply(candidates)

        # Reduce products if needed
        candidates.products = cls._merge_products(candidates.products)

        return candidates.get_best_candidates()

    @staticmethod
    def _merge_products(products: List[ProductCandidate]) -> List[ProductCandidate]:
        # TODO implement a product merge mechanism based on metadat info.
        return products

# if __name__ == '__main__':
#     from clem.candidates.collector import FieldCandidate
#
#     id_candidates = [
#         FieldCandidate(value='foo1', metadata={'page': 0, 'section': 0}),
#         FieldCandidate(value='foo2', metadata={'page': 0, 'section': 2}),
#         FieldCandidate(value='foo3', metadata={'page': 1, 'section': 0}),
#         FieldCandidate(value='foo4', metadata={'page': 2, 'section': 4}),
#     ]
#
#     candidates = CandidateCollector(id_=id_candidates)
#
#     scored_candidates = CandidateSelector.merge(candidates)
#     print(scored_candidates.get_best_candidates())
