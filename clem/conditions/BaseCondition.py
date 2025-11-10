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
    """ Base Condition. Abstract class. Do not create instances from this class! """

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

    @property
    @abstractmethod
    def weight(self) -> float:
        """ Defines how important this condition is. """
        pass

    @property
    @abstractmethod
    def name(self) -> float:
        """ Printable name to use for this condition"""
        pass


class WeakCondition(Condition, ABC):
    """
    Use this subclass for conditions that are weakly defined, give partial value to the selection or are not
    accurate enough.
    """
    weight: float = 3


class StrongCondition(Condition, ABC):
    """
    Use this subclass for conditions that define critical details about the values that are found.
    """
    weight: float = 5
