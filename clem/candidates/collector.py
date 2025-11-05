"""
File name: collector
Author: Fran Moreno
Last Updated: 10/31/2025
Version: 1.0
Description: TOFILL
"""
import dataclasses
import uuid
from pathlib import Path
import csv
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, Dict
from collections import defaultdict

from clem.datatypes import DataTypes
from clem.prediction_schema import PredictionSchema
from clem.normalization.cleaning import clean_value


@dataclass
class FieldCandidate:
    value: Optional
    datatype: Callable
    metadata: dict = field(default_factory=lambda: dict())

    score: float = .0
    _value_raw: Any = None
    _value_clean: Any = None
    _is_none: bool = False
    _failed_conversion: bool = False
    _passed_conditions: List = field(default_factory=lambda: list())

    _uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        self._value_raw = self.value
        self._is_none = self.value is None

        # Convert to its expected datatype
        conversion = DataTypes.as_type(self.value, self.datatype)
        self.value = conversion.value
        self._failed_conversion = conversion.failed

        # Normalize value for later comparison
        self._value_clean = clean_value(self._value_raw)

    def add_passed_condition(self, condition):
        self._passed_conditions.append(condition)

    @property
    def value_raw(self):
        return self._value_raw

    @property
    def value_clean(self):
        return self._value_clean

    @property
    def failed_conversion(self):
        return self._failed_conversion

    @property
    def is_none(self):
        return self._is_none

    def __hash__(self):
        return hash(self._uuid)

    def __eq__(self, other):
        if not isinstance(other, FieldCandidate):
            return False
        return self._uuid == other._uuid


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

    def log(self, dir_path: Path):
        # Invoice fields log
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        with open(dir_path / 'invoice_fields.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            header = ['Field', 'Candidate#', 'Value', 'Value (raw)', 'Is None?', 'Failed Conversion?', 'Score']
            writer.writerow(header)

            for f_name, candidates in self.invoice_fields.items():
                for idx, candidate in enumerate(candidates):
                    new_row = [f_name, idx, candidate.value, candidate.value_raw, candidate.is_none, candidate.failed_conversion, candidate.score]
                    writer.writerow(new_row)

        # Products log
        with open(dir_path / 'products.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            header = ['Field', 'Product#', 'Value']
            writer.writerow(header)

            for f_name, values in self.products_fields.items():
                for idx, value in enumerate(values):
                    writer.writerow([f_name, idx, value])

    @property
    def invoice_fields(self) -> Dict[str, List[FieldCandidate]]:
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
