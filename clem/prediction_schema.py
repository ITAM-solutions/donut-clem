"""
File name: output
Author: Fran Moreno
Last Updated: 10/27/2025
Version: 2.0
Description: This file defines the expected output schema using a Pydantic model. It also defines custom exceptions
for cases where the output schema is not valid.
"""
from collections import defaultdict
# from enum import Enum
# from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator, AliasChoices
from typing import List, Union, Optional


# @dataclass
# class DataSource:
#     pass


# @dataclass
# class Page(DataSource):
#     idx: int
#     # TODO Keep adding things here


# class SectionName(Enum):
#     HEADER = 0
#     BODY = 0
#     FOOTER = 0


# @dataclass
# class Section(DataSource):
#     part: SectionName
#     # TODO Keep adding things here
#
#
# @dataclass
# class PredictionMetadata:
#     raised_error: bool
#     source: DataSource
#     # TODO Keep adding things here



class ProductsSchema(BaseModel):
    """ Data structure for Product (or Products) information obtained from DonutCLEM. """

    name: Union[List[Optional[str]], str, None]
    sku: Union[List[Optional[str]], str, None]
    met: Union[List[Optional[str]], str, None]
    metgr: Union[List[Optional[str]], str, None]
    qty: Union[List[Optional[str]], str, None]
    unpr: Union[List[Optional[str]], str, None]
    totpr: Union[List[Optional[str]], str, None]
    dfrom: Union[List[Optional[str]], str, None] = Field(validation_alias=AliasChoices('dfrom', 'drom'))
    dto: Union[List[Optional[str]], str, None]

    @model_validator(mode='after')
    def normalize_lists(self):
        """ Ensures that each field is always a list of strings. """

        model_data = self.model_dump()  # Returns the model data in dictionary format
        self._convert_values_to_list(model_data)
        self._make_lists_equal_in_length(model_data)

        # Update instance
        for k, v in model_data.items():
            setattr(self, k, v)

        return self  # model_validator methods always need to return self (check pydantic docs).

    @property
    def num_products(self) -> int:
        """ Returns the number of products in this ProductsSchema set. """
        if isinstance(self.name, list):
            return len(self.name)
        elif isinstance(self.name, str):
            return 1
        else:
            return 0

    @staticmethod
    def _convert_values_to_list(model_data: dict) -> None:
        """
        Converts all the values in `model_data` to a list in case that they were strings. It also normalizes
        "empty" (None) values from str to NoneType.

        :param model_data: dictionary containing the model data
        :return: None (updates model_data in place).
        """
        for field_name, value in model_data.items():
            if isinstance(value, str) or value is None:
                if value == 'None':  # Normalizes empty values:
                    value = None
                model_data[field_name] = [value]

    @staticmethod
    def _make_lists_equal_in_length(model_data: dict) -> None:
        """
        Finds the maximum length in the model data values, and expands shorter lists to match that length.
        This method assumes that all the values are already lists. So make sure that `_convert_values_to_lists`
        executes before calling this method.

        :param model_data: dictionary containing the model data
        :return: None (updates model_data in place).
        """
        # TODO refine to avoid indexing values to the wrong product in list.
        max_items = max([len(l) for l in model_data.values()])
        for field_name, value_list in model_data.items():
            model_data[field_name] = value_list + [None] * (max_items - len(value_list))


class PredictionSchema(BaseModel):
    """ Data structure that results from DonutCLEM output. """

    id_: Union[str, None] = Field(alias="id")  # id
    date_: Union[str, None] = Field(alias="date")  # date
    po: Union[str, None]
    cur: Union[str, None]
    vendor: Union[str, None]
    corp: Union[str, None] = Field(validation_alias=AliasChoices('corp', 'company'))
    products: Union[ProductsSchema, List[ProductsSchema], str, None]

    # metadata: won't be returned with model.fields
    # _raised_error: bool = False
    # _source: DataSource = Page(0)
    # Keep adding things here

    @model_validator(mode='after')
    def normalize_products(self):
        """ Ensures that products is a dict of lists """
        if isinstance(self.products, list):
            products_d = defaultdict(list)
            for product in self.products:
                for k, v in product.model_dump().items():
                    products_d[k].extend(v)
            self.products = ProductsSchema.model_construct(**products_d)
        elif isinstance(self.products, str):  # No products found.
            self.products = None

        return self

    @model_validator(mode='after')
    def normalize_empty_values(self):
        """ DonutCLEM outputs 'None' string as an empty value. Normalize those values to real `None` values. """
        for field_name, value in self.model_dump().items():
            if value == 'None':
                setattr(self, field_name, None)
        return self

    @property
    def raised_error(self):
        """ Property that tells if the resulting output was produced after an error was raised. """
        return self._raised_error

    # @property
    # def metadata(self) -> dict:
    #     # return PredictionMetadata(
    #     #     raised_error=self._raised_error,
    #     #     source=self._source
    #     # )
    #     return pass


def get_empty_prediction(raised_error: bool = False) -> PredictionSchema:
    """
    Utility method to produce an empty normalized prediction.

    :param raised_error: if True, means that this empty product results from a prediction error.
    :return: normalized DonutCLEM output as a PredictionSchema instance.
    """
    content = {
        "id": None,
        "date": None,
        "po": None,
        "cur": None,
        "vendor": None,
        "corp": None,
        "products": None,
        "_raised_error": raised_error
    }
    return PredictionSchema(**content)


if __name__ == '__main__':
    # # # json_str = '{"products": {"drom": ["a", "b"]}}'
    # # json_str = '{"products": [{"name": "foo1", "drom": "None"}, {"name": "foo2", "drom": "b"}]}'
    # # s = PredictionSchema.model_validate_json(json_str)
    # # print(s)
    # json_str = '{"name": ["val1", "val2"], "sku": ["val3"]}'
    #
    # p = ProductsSchema.model_validate_json(json_str)
    # print(p)

    data = {
        'id': 'sample_id',
        'date': 'sample_date',
        'po': 'sample_po',
        'cur': 'sample_cur',
        'vendor': 'sample_vendor',
        'corp': 'sample_corp',
        'products': {
            'name': ['name1', 'name2'],
            'sku': ['sku1', 'sku2'],
            'met': ['met1', 'met2'],
            'metgr': ['metgr1', 'metgr2'],
            'qty': ['qty1', 'qty2'],
            'unpr': ['unpr1', 'unpr2'],
            'totpr': ['totpr1', 'totpr2'],
            'dfrom': ['dfrom1', 'dfrom2'],
            'dto': ['dto1', 'dto2']
        }
    }

    s = PredictionSchema(**data)
    a = 1
