"""
File name: generator_fabric
Author: Fran Moreno
Last Updated: 8/28/2025
Version: 1.0
Description: TOFILL
"""
from typing import Callable, Dict, List
from numpy.random import choice as np_choice
import random
from copy import deepcopy
from pathlib import Path
import synthetic_generator.generators.data_generators as generators

import synthetic_generator.word_bank as word_bank

from synthetic_generator.utils.config import DocxConfig
from synthetic_generator.utils.func_parsing import collect_funcs_from_module
from synthetic_generator.utils.text_format import as_rich


class GeneratorFactory:
    def __init__(self, template_path: Path):
        self.template_path = template_path

        self._generators_registry: Dict[str, Callable] = collect_funcs_from_module(
            generators,
            suffix="gen",
            default=lambda *args, **kwargs: {}
        )

        self._base_config = self._preload_document_base_config()
        self._doc_sections = []
        self._existing_vals = set()
        # self._inv_fields = self._preload_invoice_fields()
        print(self._base_config.lang)

    def generate(self, name: str, **kwargs):
        """
        Dispatch to the generator registered under 'name'.

        :param name:
        :param kwargs:
        :return:
        """
        if name not in self._generators_registry:
            return None

        # data: dict => keys('raw', 'type_': options -> ['kv', 'table', 'text'])
        data = self._generators_registry.get(name, "__default__")(
            self._base_config,
            doc_sections=self._doc_sections,
            # inv_fields=self._inv_fields,
            field_names=word_bank.INVOICE_MAIN_FIELDS,
            **kwargs)

        # data['formatted'] = modifiers.to_text_lines(data)
        data['rich'] = self._to_rich_format(data)

        return data

    def generate_color_palette(self):
        c1, c2, c3, c4 = self._pondered_choice(word_bank.COLOR_PALETTES)
        print(f"Colors: {c1}, {c2}, {c3}, {c4}")
        return {
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
        }

    def _preload_document_base_config(self):
        """
        Will prepare the base configurations for the document to be generated. This includes:
        - Language: `en` (English) or `nl` (Dutch).
        - Color Palette: contains primary, secondary and tertiary.
        - Date Format
        :return:
        """
        return DocxConfig(
            lang = self._pondered_choice(word_bank.LANGUAGES),
            colors = self._pondered_choice(word_bank.COLOR_PALETTES),
            date_format = self._pondered_choice(word_bank.DATE_FORMATS),
            date_range_sep=self._pondered_choice(word_bank.DATE_RANGE_SEPARATORS),
            price_format=random.choice(list(word_bank.PriceFormat)),
            price_show_currency=random.choice([True, False]),
            price_currency_position=random.choice([True, False]),
            number_format=random.choice(list(word_bank.NumberFormat)),
            page_idx_format=random.choice(word_bank.PAGE_IDX_FORMATS),
            pct_format=random.choice([True, False]),
            font_family = self._pondered_choice(word_bank.FONT_FAMILIES),
            font_size = self._pondered_choice(word_bank.FONT_SIZES),
            font_size_tables=word_bank.FONT_SIZES[0],
            font_color=self._pondered_choice(word_bank.FONT_COLORS),
            product_field_set=self._select_product_field_group()
        )

    @staticmethod
    def _pondered_choice(data: tuple):
        if not data:
            raise ValueError('No data to choose from.')

        if not isinstance(data[0], tuple):
            return random.choice(list(data))
        else:
            options = [i[0] for i in data]
            probs = [i[1] for i in data]
            choice_idx = np_choice(list(range(len(options))), p=probs)
            return options[choice_idx]

    def _to_rich_format(self, data: dict):
        raw_data = deepcopy(data['raw'])
        type_ = data['type']

        if type_ == "kv":
            font_size = self._base_config.font_size

            rich_fields = {
                f_id: {
                    "lbl": as_rich(f["lbl"], self._base_config, font_size),
                    "val": as_rich(f["val"], self._base_config, font_size),
                    "empty": f["val"] == ' '
                } for f_id, f in raw_data.items()
            }
            rich_content = [f for f in rich_fields.values() if not f["empty"]]
            raw_content_all = [f for f in raw_data.values()]
            raw_content = [f for f in raw_data.values() if f['val'] != ' ']
            table_data = [f"{f['lbl']}: {f['val']}" for f in rich_fields.values()]
            as_text = "\n".join([f"{i['lbl']}: {i['val']}" for i in raw_content])
            return {
                "data": rich_content,
                "f": rich_fields,
                "text": raw_data,
                "table": table_data,
                "raw": raw_content,
                "raw_all": raw_content_all,
                "as_text": as_text,
            }
        elif type_ == "table":
            num_cols = len(data['raw']['labels'])
            print("Num of cols in table:", num_cols)
            font_size = word_bank.TABLE_FONT_SIZES.get(num_cols)

            rich_content = {
                "labels": {
                    "raw": data['raw']['labels'],
                    "rich": [as_rich(lbl, self._base_config, font_size) for lbl in data['raw']['labels']],
                    "kv": {k: v for k, v in zip(data['raw']['ids'], data['raw']['labels'])}
                },
                "tb_items": [
                    {
                        "fields": {
                            "raw": tb_item['fields'],
                            "rich": [as_rich(i, self._base_config, font_size) for i in tb_item['fields']]
                        },
                        "kv": {
                            key: {"lbl": lbl, "val": val}
                            for key, (lbl, val) in zip(data['raw']['ids'], zip(data['raw']['labels'], tb_item['fields']))
                        }
                    }
                    for tb_item in data['raw']['tb_items']
                ]
            }

            return rich_content
        elif type_ == "text":
            content = data['raw']
            oneline = data['raw'].replace('\n', ', ')
            rich_content = {
                "rich": as_rich(content, self._base_config, self._base_config.font_size),
                "text": content,
                "oneline": oneline,
            }
            return rich_content

    def _select_product_field_group(self) -> Dict[str, tuple]:
        template_code = self.template_path.stem.split('_')[1]
        if template_code in word_bank.PRODUCT_FIELD_GROUPS_SPECIAL:
            field_selection_keys = word_bank.PRODUCT_FIELD_GROUPS_SPECIAL[template_code]
        else:
            field_selection_keys = random.choice(word_bank.PRODUCT_FIELD_GROUPS)

        fields_selection = {}
        for field_key in field_selection_keys:
            if field_key in word_bank.PRODUCT_MAIN_FIELDS:
                fields_selection[field_key] = word_bank.PRODUCT_MAIN_FIELDS[field_key]
            elif field_key in word_bank.PRODUCT_ADDITIONAL_FIELDS:
                fields_selection[field_key] = word_bank.PRODUCT_ADDITIONAL_FIELDS[field_key]
            else:
                print(f"Asked for non-existing product field: {field_key}. Will be skipped.")
        fields_selection = {k: (v[0], random.choice(v[1])) for k, v in fields_selection.items()}
        return fields_selection



    # def _preload_invoice_fields(self) -> dict:
    #     """
    #     Executes the main generator function that produces random values for the important invoice fields that
    #     needs to be tracked.
    #
    #     Notice that this function will only generate values for the "shared" invoice fields, not products/item fields.
    #
    #     :return:
    #     """
    #     return {
    #         k: (generate_value(v[0], config=self._base_config), random.choice(v[1])) for k, v in word_bank.INVOICE_MAIN_FIELDS.items()
    #     }

    @staticmethod
    def _to_varname(func_name: str, suffix: str) -> str:
        """
        Takes a Generator function name and normalizes it to camel case format. The result will
        be used as the identifier in the Jinja templates to select this generator function.

        :param func_name: generator function name.
        :return: function name in camel case without the initial 'get' word.
        """
        name_parts = func_name.split('_')

        if not name_parts:
            raise ValueError(f"Generator function '{func_name}' does not specify any target data in its name.")

        if name_parts[0] != suffix:
            raise ValueError(f"Function '{func_name}' does not follow the expected name pattern for generator functions.")

        name_parts.pop(0)
        if not name_parts or not name_parts[0]:
            raise ValueError(f"Function '{func_name}' does not follow the expected name pattern for generator functions.")

        varname = name_parts.pop(0)

        varname += ''.join([i.capitalize() for i in name_parts])
        return varname

    @property
    def generators(self) -> List[str]:
        """ List of all registered generator names. """
        return list(self._generators_registry.keys())

    @property
    def docx_config(self) -> DocxConfig:
        return self._base_config

    @property
    def expected_product_fields(self) -> List[str]:
        return [key for key in self._base_config.product_field_set if key in word_bank.PRODUCT_MAIN_FIELDS]

    # @property
    # def inv_fields(self) -> dict:
    #     """ List of Important invoice fields to be included in the context. """
    #     inv_fields = {
    #         'lbl': {},
    #         'val': {}
    #     }
    #     for field_, (value, name) in self._inv_fields.items():
    #         inv_fields['lbl'][field_] = rich(name, self._base_config)
    #         inv_fields['val'][field_] = rich(value, self._base_config)
    #     return inv_fields
