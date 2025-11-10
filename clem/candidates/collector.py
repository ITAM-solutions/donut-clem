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
from typing import Optional, List, Callable, Any, Dict, Set
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

    _score_log: List = field(default_factory=lambda: list())
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

    def update_score(self, inc, condition_name: str) -> None:
        self.score += round(inc, 3)
        self._score_log.append(f"{'+' if inc>=0 else '-'}{condition_name}")

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

    def __repr__(self):
        score_logs = f", log={', '.join(self._score_log)}" if self._score_log else ''
        return f"FieldCandidate(value={self.value}, score={self.score}{score_logs})"


def get_empty_field(field_name: str) -> FieldCandidate:
    datatypes_mapping = {
        'name': DataTypes.str,
        'sku': DataTypes.str,
        'qty': DataTypes.int,
        'met': DataTypes.str,
        'metgr': DataTypes.str,
        'dfrom': DataTypes.date,
        'dto': DataTypes.date,
        'unpr': DataTypes.price,
        'totpr': DataTypes.price,
    }
    return FieldCandidate(value=None, datatype=datatypes_mapping.get(field_name, DataTypes.str))


@dataclass
class ProductCandidate:
    name: FieldCandidate = field(default_factory=lambda: get_empty_field('name'))
    sku: FieldCandidate = field(default_factory=lambda: get_empty_field('sku'))
    qty: FieldCandidate = field(default_factory=lambda: get_empty_field('qty'))
    met: FieldCandidate = field(default_factory=lambda: get_empty_field('met'))
    metgr: FieldCandidate = field(default_factory=lambda: get_empty_field('metgr'))
    dfrom: FieldCandidate = field(default_factory=lambda: get_empty_field('dfrom'))
    dto: FieldCandidate = field(default_factory=lambda: get_empty_field('dto'))
    unpr: FieldCandidate = field(default_factory=lambda: get_empty_field('unpr'))
    totpr: FieldCandidate = field(default_factory=lambda: get_empty_field('totpr'))

    @property
    def dict(self) -> dict:
        as_dict = dict()
        for f in dataclasses.fields(self):  # noqa
            field_candidate: FieldCandidate = getattr(self, f.name)
            as_dict[f.name] = field_candidate.value
        return as_dict

    @property
    def dict_raw(self) -> dict:
        as_dict = dict()
        for f in dataclasses.fields(self):  # noqa
            field_candidate: FieldCandidate = getattr(self, f.name)
            as_dict[f.name] = field_candidate.value_raw
        return as_dict

    @property
    def is_complete(self) -> bool:
        return all(self.dict.values())

    @property
    def filled_fields(self) -> Dict[str, Any]:
        return {
            f.name: field_candidate
            for f in dataclasses.fields(self) if not (field_candidate := getattr(self, f.name)).is_none
        }

    def __repr__(self):
        return (f"ProductCandidate(name={self.name.value_raw}, sku={self.sku.value_raw}, qty={self.qty.value_raw}, "
                f"met={self.met.value_raw}, metgr={self.metgr.value_raw}, dfrom={self.dfrom.value_raw}, "
                f"dto={self.dto.value_raw}, unpr={self.unpr.value_raw}, totpr={self.totpr.value_raw}")


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
        self.id_.append(FieldCandidate(prediction.id_, DataTypes.str, metadata=metadata))
        self.date_.append(FieldCandidate(prediction.date_, DataTypes.date, metadata=metadata))
        self.po.append(FieldCandidate(prediction.po, DataTypes.str, metadata=metadata))
        self.cur.append(FieldCandidate(prediction.cur, DataTypes.currency, metadata=metadata))
        self.vendor.append(FieldCandidate(prediction.vendor, DataTypes.str, metadata=metadata))
        self.corp.append(FieldCandidate(prediction.corp, DataTypes.str, metadata=metadata))

        num_products = prediction.products.num_products if prediction.products else 0
        self.products.extend([
            ProductCandidate(
                name=FieldCandidate(prediction.products.name[idx], DataTypes.str, metadata=metadata),
                sku=FieldCandidate(prediction.products.sku[idx], DataTypes.str, metadata=metadata),
                met=FieldCandidate(prediction.products.met[idx], DataTypes.str, metadata=metadata),
                metgr=FieldCandidate(prediction.products.metgr[idx], DataTypes.str, metadata=metadata),
                qty=FieldCandidate(prediction.products.qty[idx], DataTypes.int, metadata=metadata),
                unpr=FieldCandidate(prediction.products.unpr[idx], DataTypes.price, metadata=metadata),
                totpr=FieldCandidate(prediction.products.totpr[idx], DataTypes.price, metadata=metadata),
                dfrom=FieldCandidate(prediction.products.dfrom[idx], DataTypes.date, metadata=metadata),
                dto=FieldCandidate(prediction.products.dto[idx], DataTypes.date, metadata=metadata),
            )
            for idx in range(num_products)
        ])

    def get_best_candidates(self):
        """
        :return:
        """

        best = dict()

        # Invoice fields
        for f_name, options in self.invoice_fields.items():
            # Get just the options with the higher score
            filtered_options = list(filter(
                lambda option: option.score == max(options, key=lambda x: x.score).score,
                options
            ))
            best[f_name] = [option.value_raw for option in filtered_options]

        # Products
        best['products'] = self.product_fields_raw

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
    def products_fields(self) -> dict:
        as_dict = defaultdict(list)
        for product in self.products:
            for field_name, value in product.dict.items():
                as_dict[field_name].append(value)
        return dict(as_dict)  # defaultdict to dict

    @property
    def product_fields_raw(self) -> dict:
        as_dict = defaultdict(list)
        for product in self.products:
            for field_name, value in product.dict_raw.items():
                as_dict[field_name].append(value)
        return dict(as_dict)  # defaultdict to dict

if __name__ == '__main__':
    product = ProductCandidate(
        name=FieldCandidate('a', DataTypes.str),
        sku=FieldCandidate('b', DataTypes.str),
    )
    print(product.filled_fields)
