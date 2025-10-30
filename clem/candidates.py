"""
File name: candidates
Author: Fran Moreno
Last Updated: 10/29/2025
Version: 1.0
Description: TOFILL
"""
from dataclasses import dataclass
from typing import Optional

from prediction_schema import PredictionMetadata


@dataclass
class FieldCandidate:
    value: Optional
    metadata: PredictionMetadata
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