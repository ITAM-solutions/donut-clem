"""
File name: PageNumberCondition
Author: Fran Moreno
Last Updated: 11/7/2025
Version: 1.0
Description: TOFILL
"""
from clem.conditions.BaseCondition import WeakCondition
from clem.candidates.collector import CandidateCollector
from clem.datatypes import DataTypes


class PageNumberCondition(WeakCondition):
    """ Assign higher scores to values that are located in places where they are more commonly found. """
    name = "PageNumber[W]"

    @classmethod
    def apply(cls, candidates: CandidateCollector) -> None:
        fields_for_which_page_idx_matters = [
            'id_',
            'corp',
            'vendor',
            'po',
            'cur',
        ]

        for field_name in fields_for_which_page_idx_matters:
            for candidate in getattr(candidates, field_name):
                if candidate.metadata.get('page') == 0:
                    candidate.update_score(cls.weight, cls.name)


if __name__ == '__main__':
    from clem.candidates.collector import FieldCandidate

    id_candidates = [
        FieldCandidate(value='foo1', datatype=DataTypes.str, metadata={'page': 0}),
        FieldCandidate(value='foo2', datatype=DataTypes.str, metadata={'page': 0}),
        FieldCandidate(value='foo3', datatype=DataTypes.str, metadata={'page': 1}),
        FieldCandidate(value='foo4', datatype=DataTypes.str, metadata={'page': 2}),
    ]

    corp_candidates = [
        FieldCandidate(value='foo1', datatype=DataTypes.str, metadata={'page': 1}),
        FieldCandidate(value='foo2', datatype=DataTypes.str, metadata={'page': 2}),
        FieldCandidate(value='foo3', datatype=DataTypes.str, metadata={'page': 0}),
        FieldCandidate(value='foo4', datatype=DataTypes.str, metadata={'page': 0}),
    ]

    po_candidates = [
        FieldCandidate(value='foo1', datatype=DataTypes.str, metadata={'page': 0}),
        FieldCandidate(value='foo2', datatype=DataTypes.str, metadata={'page': 0}),
        FieldCandidate(value='foo3', datatype=DataTypes.str, metadata={'page': 0}),
        FieldCandidate(value='foo4', datatype=DataTypes.str, metadata={'page': 2}),
    ]

    candidates = CandidateCollector(
        id_=id_candidates,
        corp=corp_candidates,
        po=po_candidates
    )

    PageNumberCondition.apply(candidates)
    print(candidates)