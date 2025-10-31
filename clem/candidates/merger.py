"""
File name: merger
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from clem.candidates.collector import CandidateCollector
from clem.conditions import conditions


def merge(candidates: CandidateCollector):
    for condition in conditions:
        condition(candidates)

    return candidates
