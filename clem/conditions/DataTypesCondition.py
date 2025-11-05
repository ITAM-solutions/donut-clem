"""
File name: DataTypesCondition
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from clem.conditions.BaseCondition import StrongCondition
from clem.candidates.collector import CandidateCollector
from clem.datatypes import DataTypes


class DataTypesCondition(StrongCondition):
    """ Assign higher scores to values that were successfully converted to their expected datatype. """

    @classmethod
    def apply(cls, candidates: CandidateCollector) -> None:
        for invoice_field, field_candidates in candidates.invoice_fields.items():
            for field_candidate in field_candidates:
                # TODO refine assigned values after defining more conditions.
                if not field_candidate.failed_conversion:
                    if field_candidate.datatype == DataTypes.str:
                        field_candidate.score += 1
                    else:
                        field_candidate.score += 5
                else:
                    field_candidate.score -= 1


if __name__ == '__main__':
    print(type(DataTypesCondition))

#     from clem.candidates.collector import FieldCandidate
#
#     id_candidates = [
#         FieldCandidate(value='foo1', datatype=DataTypes.str),
#         FieldCandidate(value='foo2', datatype=DataTypes.str),
#         FieldCandidate(value='foo3', datatype=DataTypes.str),
#         FieldCandidate(value='foo4', datatype=DataTypes.str),
#     ]
#
#     currency_candidates = [
#         FieldCandidate(value='EUR', datatype=DataTypes.currency),
#         FieldCandidate(value='USD', datatype=DataTypes.currency),
#         FieldCandidate(value='foo1', datatype=DataTypes.currency),
#         FieldCandidate(value='foo2', datatype=DataTypes.currency),
#     ]
#
#     candidates = CandidateCollector(id_=id_candidates, cur=currency_candidates)
#     print("Initial candidates ID:", id(candidates))
#     DataTypesCondition.apply(candidates)
#     print("Updates candidates ID:", id(candidates))