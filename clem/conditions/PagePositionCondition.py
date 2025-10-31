"""
File name: PagePosition
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from clem.conditions.BaseCondition import Condition
from clem.candidates.collector import CandidateCollector

class PagePositionCondition(Condition):

    @classmethod
    def apply(cls, candidates: CandidateCollector) -> None:
        for id_candidate in candidates.id_:
            if id_candidate.metadata.get('page') == 0:
                id_candidate.score += 1
            if id_candidate.metadata.get('section') in (0, 4):
                id_candidate.score += 1


if __name__ == '__main__':
    from clem.candidates.collector import FieldCandidate

    id_candidates = [
        FieldCandidate(value='foo1', metadata={'page': 0, 'section': 0}),
        FieldCandidate(value='foo2', metadata={'page': 0, 'section': 2}),
        FieldCandidate(value='foo3', metadata={'page': 1, 'section': 0}),
        FieldCandidate(value='foo4', metadata={'page': 2, 'section': 4}),
    ]

    candidates = CandidateCollector(id_=id_candidates)
    print("Initial candidates ID:", id(candidates))
    PagePositionCondition.apply(candidates)
    print("Updates candidates ID:", id(candidates))

