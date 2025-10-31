"""
File name: merger
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from clem.candidates.collector import CandidateCollector
from clem.conditions import conditions


class CandidateSelector:
    def __init__(self, candidates: CandidateCollector):
        self._candidates = candidates

    @classmethod
    def merge(cls, candidates: CandidateCollector):
        for condition in conditions:
            condition.apply(candidates)

        return cls(candidates)

    def get_best_candidates(self):
        best = dict()

        # Invoice fields
        for f_name, options in self._candidates.invoice_fields.items():
            best[f_name] = max(options, key=lambda x: x.score).value if options else None

        # Products
        best['products'] = self._candidates.products

        return best


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
