"""
File name: collector
Author: Fran Moreno
Last Updated: 10/29/2025
Version: 1.0
Description: TOFILL
"""
from dataclasses import dataclass, field
from typing import List

from prediction_schema import PredictionSchema
from candidates import FieldCandidate, ProductCandidate


@dataclass
class CandidateCollector:
    """ Mirrors the PredictionSchema structure, but with sets of Candidates instead of field values. """

    id_: List[FieldCandidate] = field(default_factory=lambda: [])
    date_: List[FieldCandidate] = field(default_factory=lambda: [])
    po: List[FieldCandidate] = field(default_factory=lambda: [])
    cur: List[FieldCandidate] = field(default_factory=lambda: [])
    vendor: List[FieldCandidate] = field(default_factory=lambda: [])
    corp: List[FieldCandidate] = field(default_factory=lambda: [])
    products: List[ProductCandidate] = field(default_factory=lambda: [])

    def add(self, prediction: PredictionSchema):
        self.id_.append(FieldCandidate(prediction.id_, prediction.metadata))
        self.date_.append(FieldCandidate(prediction.date_, prediction.metadata))
        self.po.append(FieldCandidate(prediction.po, prediction.metadata))
        self.cur.append(FieldCandidate(prediction.cur, prediction.metadata))
        self.vendor.append(FieldCandidate(prediction.vendor, prediction.metadata))
        self.corp.append(FieldCandidate(prediction.corp, prediction.metadata))
        self.products.extend([
            ProductCandidate(**{
                field_name: vals[idx] for field_name, vals in prediction.products.model_dump().items()
            }) for idx in range(prediction.products.num_products)
        ])

    def merge(self) -> "CandidateCollector":
        return ScoreSystem.compute_scores(self)


class ScoreSystem:
    """ Collects all scoring conditions and defines a method to execute them sequentially. """

    @classmethod
    def compute_scores(cls, candidates: CandidateCollector) -> CandidateCollector:
        conditions = [
            cls.condition_1,
            cls.condition_2,
            cls.condition_3,
            cls.condition_4,
            # Keep expanding if needed
        ]

        for condition in conditions:
            candidates = condition(candidates)
        return candidates

    @staticmethod
    def condition_1(candidates: CandidateCollector) -> CandidateCollector:
        return candidates

    @staticmethod
    def condition_2(candidates: CandidateCollector) -> CandidateCollector:
        return candidates

    @staticmethod
    def condition_3(candidates: CandidateCollector) -> CandidateCollector:
        return candidates

    @staticmethod
    def condition_4(candidates: CandidateCollector) -> CandidateCollector:
        return candidates
