"""
File name: collector
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
import dataclasses

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any
from collections import defaultdict

from clem.datatypes.conversion import DataTypes
from clem.prediction_schema import PredictionSchema

@dataclass
class FieldCandidate:
    value: Optional
    datatype: Callable
    metadata: dict = field(default_factory=lambda: dict())
    score: int = 0
    score_norm: float = 0.0
    _value_raw: Any = None

    def __post_init__(self):
        self._value_raw = self.value
        self.value = DataTypes.as_type(self.value, self.datatype)

    @property
    def value_raw(self):
        return self._value_raw


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

    @property
    def dict(self) -> dict:
        as_dict = dict()
        for f in dataclasses.fields(self):  # noqa
            field_candidate: FieldCandidate = getattr(self, f.name)
            as_dict[f.name] = field_candidate.value
        return as_dict

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
        self.id_.append(FieldCandidate(prediction.id_, DataTypes.str, metadata))
        self.date_.append(FieldCandidate(prediction.date_, DataTypes.str, metadata))
        self.po.append(FieldCandidate(prediction.po, DataTypes.str, metadata))
        self.cur.append(FieldCandidate(prediction.cur, DataTypes.currency, metadata))
        self.vendor.append(FieldCandidate(prediction.vendor, DataTypes.str, metadata))
        self.corp.append(FieldCandidate(prediction.corp, DataTypes.str, metadata))

        num_products = prediction.products.num_products if prediction.products else 0
        self.products.extend([
            ProductCandidate(
                name=FieldCandidate(prediction.products.name[idx], DataTypes.str, metadata ),
                sku=FieldCandidate(prediction.products.sku[idx], DataTypes.str, metadata ),
                met=FieldCandidate(prediction.products.met[idx], DataTypes.str, metadata ),
                metgr=FieldCandidate(prediction.products.metgr[idx], DataTypes.str, metadata ),
                qty=FieldCandidate(prediction.products.qty[idx], DataTypes.int, metadata ),
                unpr=FieldCandidate(prediction.products.unpr[idx], DataTypes.float, metadata ),
                totpr=FieldCandidate(prediction.products.totpr[idx], DataTypes.float, metadata ),
                dfrom=FieldCandidate(prediction.products.dfrom[idx], DataTypes.date, metadata ),
                dto=FieldCandidate(prediction.products.dto[idx], DataTypes.float, metadata ),
            )
            for idx in range(num_products)
        ])

    def get_best_candidates(self):
        best = dict()

        # Invoice fields
        for f_name, options in self.invoice_fields.items():
            best[f_name] = max(options, key=lambda x: x.score).value if options else None

        # Products
        best['products'] = self.products_fields

        return best

    @property
    def invoice_fields(self):
        as_dict = dict()
        for f in dataclasses.fields(self):  # noqa
            if f.name != 'products':
                as_dict[f.name] = getattr(self, f.name)
        return as_dict

    @property
    def products_fields(self):
        as_dict = defaultdict(list)
        for product in self.products:
            for field_name, value in product.dict.items():
                as_dict[field_name].append(value)
        return dict(as_dict)  # defaultdict to dict


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


if __name__ == '__main__':
    # products_candidates = ProductCandidate(
    #     name=FieldCandidate('foo1'),
    #     sku=FieldCandidate('foo2'),
    #     qty=FieldCandidate('foo3'),
    #     met=FieldCandidate('foo4'),
    #     metgr=FieldCandidate('foo5'),
    #     dfrom=FieldCandidate('foo6'),
    #     dto=FieldCandidate('foo7'),
    #     unpr=FieldCandidate('foo8'),
    #     totpr=FieldCandidate('foo9'),
    # )

    # print(products_candidates.dict)
    # candidate = FieldCandidate('1', lambda x: int(x))
    # print(type(candidate.value))

    # class Currency(str):
    #     allowed = {"EUR", "USD"}
    #
    #     def __new__(cls, value):
    #         if value not in cls.allowed:
    #             raise ValueError(f"{value} is not a valid {cls.__name__}.")
    #         return str.__new__(cls, value)

    candidate = FieldCandidate('10', DataTypes.int)

    a = 1