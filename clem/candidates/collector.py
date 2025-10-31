"""
File name: collector
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
import dataclasses

from dataclasses import dataclass, field
from typing import Optional, List

from clem.prediction_schema import PredictionMetadata, PredictionSchema


@dataclass
class FieldCandidate:
    value: Optional
    metadata: dict = field(default_factory=lambda: dict())
    score: int = 0
    score_norm: float = 0.0


@dataclass
class ProductCandidate:
    name: FieldCandidate
    sku: FieldCandidate
    qty: FieldCandidate
    met: FieldCandidate
    metgr: FieldCandidate
    dfrom: FieldCandidate
    dto: FieldCandidate
    unpr: FieldCandidate
    totpr: FieldCandidate


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

    def add(self, prediction: PredictionSchema, metadata: dict):
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

    @property
    def invoice_fields(self):
        as_dict = dict()
        for f in dataclasses.fields(self):  # noqa
            if f.name != 'products':
                as_dict[f.name] = getattr(self, f.name)
        return as_dict


# if __name__ == '__main__':
#     from prediction_schema import ProductsSchema
#
#     json_str1 = '{"id": "pred1v1", "products": {"name": ["prod1v1", "prod2v1"], "sku": ["prod1v2", "prod2v2"]}}'
#     json_str2 = '{"id": "pred2v1", "products": {"name": ["prod3v1", "prod4v1", "prod5v1"], "sku": ["prod3v2", "prod4v2", "prod5v2"]}}'
#
#     data1 = {
#         'id': 'pred1v1',
#         'products': {
#             'name': ['a', 'b'],
#             'sku': ['c', 'd'],
#         }
#     }
#     pr1 = PredictionSchema(**data1)
#     # pr1 = PredictionSchema.model_validate_json(json_str1)
#     pr2 = PredictionSchema.model_validate_json(json_str2)
#
#     candidate_collector = CandidateCollector()
#     candidate_collector.add(pr1)
#     candidate_collector.add(pr2)
#     print(candidate_collector)
