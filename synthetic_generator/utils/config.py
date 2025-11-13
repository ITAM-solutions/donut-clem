"""
File name: config
Author: Fran Moreno
Last Updated: 9/1/2025
Version: 1.0
Description: TOFILL
"""
from dataclasses import dataclass
from faker import Faker
from typing import Tuple, Dict
from synthetic_generator.word_bank import PriceFormat, NumberFormat


@dataclass
class DocxConfig:
    lang: str
    colors: Tuple[str, str, str, str]  # Palette name, set of bg, border and font colors, and probability.
    date_format: str
    date_range_sep: str
    price_format: PriceFormat
    price_show_currency: bool
    price_currency_position: bool
    number_format: NumberFormat
    page_idx_format: str
    pct_format: bool
    font_family: str
    font_size: int
    font_size_tables: int
    font_color: str
    product_field_set: Dict[str, tuple]
    faker: Faker = None

    def __post_init__(self):
        self.faker = Faker(locale=self.lang)
