"""
File name: output
Author: Fran Moreno
Last Updated: 10/27/2025
Version: 1.0
Description: This file defines the expected output schema using a Pydantic model. It also defines custom exceptions
for cases where the output schema is not valid.
"""
import json

from pydantic import BaseModel, Field, AliasChoices, model_validator
from typing import List, Union
from collections import defaultdict


class ProductsSchema(BaseModel):
    name: Union[List[str], str, None]
    sku: Union[List[str], str, None]
    met: Union[List[str], str, None]
    metgr: Union[List[str], str, None]
    qty: Union[List[str], str, None]
    unpr: Union[List[str], str, None]
    totpr: Union[List[str], str, None]
    dfrom: Union[List[str], str, None] = Field(validation_alias=AliasChoices('dfrom', 'drom'))
    dto: Union[List[str], str, None]

    @model_validator(mode='after')
    def normalize_lists(self):
        """ Ensures that each field is always a list of strings. """
        for field_name, value in self.model_dump().items():
            if isinstance(value, str):
                if value == 'None':  # Normalizes empty values:
                    value = None
                setattr(self, field_name, [value])
        return self


class PredictionSchema(BaseModel):
    id_: Union[str, None] = Field(alias="id")
    date_: Union[str, None] = Field(alias="date")
    po: Union[str, None]
    cur: Union[str, None]
    vendor: Union[str, None]
    corp: Union[str, None] = Field(validation_alias=AliasChoices('corp', 'company'))
    products: Union[ProductsSchema, List[ProductsSchema], str, None]

    @model_validator(mode='after')
    def normalize_products(self):
        """ Ensures that products is a dict of lists"""
        if isinstance(self.products, list):
            products_d = defaultdict(list)
            for product in self.products:
                for k, v in product.model_dump().items():
                    products_d[k].extend(v)
            self.products = ProductsSchema.model_construct(**products_d)
        elif isinstance(self.products, str):
            self.products = None

        return self

    @model_validator(mode='after')
    def normalize_empty_values(self):
        for field_name, value in self.model_dump().items():
            if value == 'None':
                setattr(self, field_name, None)
        return self


def get_empty_prediction() -> PredictionSchema:
    content = {
        "id": None,
        "date": None,
        "po": None,
        "cur": None,
        "vendor": None,
        "corp": None,
        "products": None
    }
    return PredictionSchema.model_validate_json(json.dumps(content))


# if __name__ == '__main__':
#     # json_str = '{"products": {"drom": ["a", "b"]}}'
#     json_str = '{"products": [{"name": "foo1", "drom": "None"}, {"name": "foo2", "drom": "b"}]}'
#     s = PredictionSchema.model_validate_json(json_str)
#     print(s)
