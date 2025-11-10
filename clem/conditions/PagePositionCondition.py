"""
File name: PagePosition
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from clem.conditions.BaseCondition import WeakCondition
from clem.candidates.collector import CandidateCollector
from clem.datatypes import DataTypes


class PagePositionCondition(WeakCondition):
    """ Assign higher scores to values that are located in places where they are more commonly found. """
    name = "PagePosition[W]"

    @classmethod
    def apply(cls, candidates: CandidateCollector) -> None:
        fields_for_which_page_position_matters = {
            'id_': [0, 1],
            'corp': [0, 1],
            'vendor': [0, 1],
            'po': [0, 1],
            'date_': [0, 3],
            'cur': [1, 2, 3]
        }

        for field_name, common_sections in fields_for_which_page_position_matters.items():
            for candidate in getattr(candidates, field_name):
                if candidate.metadata.get('section') in common_sections:
                    candidate.update_score(cls.weight, cls.name)


if __name__ == '__main__':
    from clem.candidates.collector import FieldCandidate

    id_candidates = [
        FieldCandidate(value='foo1', datatype=DataTypes.str, metadata={'page': 0, 'section': 0}),
        FieldCandidate(value='foo2', datatype=DataTypes.str, metadata={'page': 0, 'section': 2}),
        FieldCandidate(value='foo3', datatype=DataTypes.str, metadata={'page': 1, 'section': 0}),
        FieldCandidate(value='foo4', datatype=DataTypes.str, metadata={'page': 2, 'section': 4}),
    ]

    cur_candidates = [
        FieldCandidate(value='foo1', datatype=DataTypes.str, metadata={'page': 0, 'section': 0}),
        FieldCandidate(value='foo2', datatype=DataTypes.str, metadata={'page': 0, 'section': 2}),
        FieldCandidate(value='foo3', datatype=DataTypes.str, metadata={'page': 1, 'section': 0}),
        FieldCandidate(value='foo4', datatype=DataTypes.str, metadata={'page': 2, 'section': 4}),
    ]

    candidates = CandidateCollector(id_=id_candidates, cur=cur_candidates)
    print("Initial candidates ID:", id(candidates))
    PagePositionCondition.apply(candidates)
    print("Updates candidates ID:", id(candidates))

