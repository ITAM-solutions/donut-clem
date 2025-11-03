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

from clem.datatypes import DataTypes
from clem.prediction_schema import PredictionSchema

@dataclass
class FieldCandidate:
    value: Optional
    datatype: Callable
    metadata: dict = field(default_factory=lambda: dict())
    score: int = 0

    _value_raw: Any = None
    _is_none: bool = False
    _failed_conversion: bool = False

    def __post_init__(self):
        self._value_raw = self.value
        self._is_none = self.value is None

        conversion = DataTypes.as_type(self.value, self.datatype)
        self.value = conversion.value
        self._failed_conversion = conversion.failed

    @property
    def value_raw(self):
        return self._value_raw

    @property
    def failed_conversion(self):
        return self._failed_conversion

    @property
    def is_none(self):
        return self._is_none


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

    @property
    def is_complete(self) -> bool:
        return all(self.dict.values())

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
        self.date_.append(FieldCandidate(prediction.date_, DataTypes.date, metadata))
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
                unpr=FieldCandidate(prediction.products.unpr[idx], DataTypes.price, metadata ),
                totpr=FieldCandidate(prediction.products.totpr[idx], DataTypes.price, metadata ),
                dfrom=FieldCandidate(prediction.products.dfrom[idx], DataTypes.date, metadata ),
                dto=FieldCandidate(prediction.products.dto[idx], DataTypes.date, metadata ),
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
