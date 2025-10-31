"""
File name: BaseCondition
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
from abc import ABC, abstractmethod
from clem.candidates.collector import CandidateCollector

class Condition(ABC):

    @classmethod
    @abstractmethod
    def apply(cls, candidates: CandidateCollector) -> None:
        """
        Receives a set of candidates, and updates its scores in place by the
        rules defined in the given condition.

        :param candidates: initial set of candidates for every prediction field.
        :return: None (updates candidates object in place).
        """
        pass