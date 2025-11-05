"""
File name: utils
Author: Fran Moreno
Last Updated: 11/4/2025
Version: 1.0
Description: TOFILL
"""
from typing import List
from clem.candidates.collector import FieldCandidate


def compute_similarity(candidate1: FieldCandidate, candidate2: FieldCandidate):
    return 1.0


def get_unique_candidates(candidates: List[FieldCandidate]):
    """

    :param candidates:
    :return:
    """
    unique_subsets = dict()

    th = 0.99

    for candidate in candidates:
        for unique_subset in unique_subsets:
            for key, candidate2 in unique_subset.items():
                similarity_ratio = compute_similarity(candidate, candidate2)
                if similarity_ratio >= th and already_indexed_in:
                    # We can assume that they contain the same value.
                    unique_subset[key].append(candidate)
                    already_indexed_in = key

        # No previous matches. Create its own unique subset.
        unique_subsets[candidate.value_clean] = [candidate]
