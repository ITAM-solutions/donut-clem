"""
File name: output
Author: Fran Moreno
Last Updated: 10/27/2025
Version: 1.0
Description: This file defines the expected output schema using a Pydantic model. It also defines custom exceptions
for cases where the output schema is not valid.
"""
from pydantic import BaseModel, Field, AliasChoices
from typing import List, Union

class PredictionSchemaNotValidException(Exception):
    pass


class ProductsSchema(BaseModel):
    name: Union[List[str], str]
    sku: Union[List[str], str]
    met: Union[List[str], str]
    metgr: Union[List[str], str]
    qty: Union[List[str], str]
    unpr: Union[List[str], str]
    totpr: Union[List[str], str]
    dfrom: Union[List[str], str] = Field(validation_alias=AliasChoices('dfrom', 'drom'))
    dto: Union[List[str], str]


class PredictionSchema(BaseModel):
    id_: str = Field(alias="id")
    date_: str = Field(alias="date")
    po: str
    cur: str
    vendor: str
    corp: str = Field(validation_alias=AliasChoices('corp', 'company'))
    products: Union[ProductsSchema, List[ProductsSchema], str]


# if __name__ == '__main__':
#     json_str = '{"id": "abc", "products": {"drom": ["a", "b"]}}'
#     PredictionSchema.model_validate_json(json_str)
