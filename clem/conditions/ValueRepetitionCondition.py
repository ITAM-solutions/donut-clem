"""
File name: ValueRepetitionCondition
Author: Fran Moreno
Last Updated: 11/4/2025
Version: 1.0
Description: TOFILL
"""
from typing import Dict, List
from collections import defaultdict

from clem.conditions.BaseCondition import StrongCondition
from clem.candidates.collector import CandidateCollector, FieldCandidate
from clem.utils.similarity import is_a_match


class ValueRepetitionCondition(StrongCondition):
    """ Assign higher scores to values that are repeated multiple times between different candidates. """

    @classmethod
    def apply(cls, candidates: CandidateCollector) -> None:
        """

        :param candidates:
        :return:
        """

        # First iteration: over invoice fields. Each of them have a list of candidates.
        for invoice_field_id, field_candidates in candidates.invoice_fields.items():
            unique_values_map = cls._find_best_unique_values(field_candidates)
            cls._assign_scores_for_repetition(unique_values_map, field_candidates)

    @classmethod
    def _find_best_unique_values(cls, candidates: List[FieldCandidate]):
        unique_values_map = defaultdict(list)

        for candidate in candidates:
            unique_vals_matches = [
                unique_value for unique_value in unique_values_map if is_a_match(unique_value, candidate.value_clean)
            ]

            if not unique_vals_matches:
                unique_values_map[candidate.value_clean].append(candidate)
            else:
                for unique_value in unique_vals_matches:
                    unique_values_map[unique_value].append(candidate)

        # Reduce duplicate matches in the mapping and return the result.
        val = cls._merge_uniques_with_common_candidates(unique_values_map)
        return val

    @classmethod
    def _assign_scores_for_repetition(cls, uniques_map: Dict[str, List], actual_candidates: List[FieldCandidate]):
        total_num_candidates = len(actual_candidates)

        for unique_id, candidates in uniques_map.items():
            num_reps = len(candidates)
            for candidate in candidates:
                weight = cls.weight * (num_reps / total_num_candidates)  # noqa
                actual_candidates[actual_candidates.index(candidate)].score += weight

    @classmethod
    def _merge_uniques_with_common_candidates(cls, uniques: Dict[str, list]) -> Dict[str, list]:
        """
        :param uniques:
        :return:
        """
        # Runs twice to ensure that all
        return cls._merge_duplicates(cls._merge_duplicates(uniques))

    @staticmethod
    def _merge_duplicates(uniques: dict):
        new_uniques = dict()
        keys = list(uniques.keys())

        for i in range(len(keys)):
            key_i = keys[i]
            candidates_i = uniques[key_i]
            if candidates_i:  # Not None
                new_uniques[key_i] = uniques[key_i]
            else:  # Has been merged to a previous list
                continue

            for j in range(i + 1, len(keys)):  # The remaining lists to compare with
                key_j = keys[j]
                candidates_j = uniques[key_j]

                for candidate in candidates_i:
                    if candidate in candidates_j: # Merge with "i" list
                        new_uniques[key_i] = list(set(new_uniques[key_i]).union(set(candidates_j)))
                        uniques[key_j] = []
                        break

        return new_uniques


if __name__ == '__main__':
    from clem.candidates.collector import FieldCandidate
    from clem.datatypes import DataTypes

    id_candidates = [
        FieldCandidate(value='foo1', datatype=DataTypes.str),
        FieldCandidate(value='foo1', datatype=DataTypes.str),
        FieldCandidate(value='foo2', datatype=DataTypes.str),
        FieldCandidate(value='foo1', datatype=DataTypes.str),
        FieldCandidate(value='foo1', datatype=DataTypes.str),
        FieldCandidate(value='foo2', datatype=DataTypes.str),
        FieldCandidate(value='foo3', datatype=DataTypes.str),
        FieldCandidate(value='foo4', datatype=DataTypes.str),
    ]

    candidates = CandidateCollector(id_=id_candidates)
    print("Initial candidates ID:", id(candidates))
    ValueRepetitionCondition.apply(candidates)
    print("Updates candidates ID:", id(candidates))




