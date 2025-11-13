"""
File name: utils
Author: Fran Moreno
Last Updated: 8/28/2025
Version: 1.0
Description: TOFILL
"""
from string import ascii_uppercase, digits
import random
import time
import sys
from numpy.random import choice as np_choice
from datetime import datetime
from typing import Tuple

import synthetic_generator.word_bank as word_bank
import synthetic_generator.generators.helpers as helpers

from synthetic_generator.utils.datatypes import FieldType
from synthetic_generator.utils.func_parsing import collect_funcs_from_module
from synthetic_generator.utils.config import DocxConfig


def get_id_invoice(config: DocxConfig, **kwargs):
    formats = [
        'Q-000000-00',
        'Q-00000-000',
        'INV-000000',
        'INV-00000',
        'QAA-000000',
        'QAA-00000',
        'INVAA-000000',
        'INVAA-00000',
        'QAA-AA0000',
        '0000000000',
        '000000000000',
        '000000000',
        'A0000AAAAA',
        'AA0000AAA',
        'AA-000000',
    ]

    template = random.choice(formats)
    text = ''

    for character in template:
        if character == 'A':
            text += random.choice(ascii_uppercase)
        elif character == '0':
            text += random.choice(digits)
        else:
            text += character

    return text


def get_po_number(config: DocxConfig, **kwargs):
    formats = [
        '000000000000000',
        '00000000000000',
        '0000000000000',
        '000000000000',
        '00000000000',
        '0000000000',
        'PO-000000000000',
        'PO-00000000000',
        'PO-0000000000',
        'PO-000000000',
        'A00000000000AAAA',
        'AA0000000000AA',
        'AA-AA-00000000',
        'AA-AA-00000',
        'PO#000000000000',
        'PO#00000000000',
        'PO#0000000000',
        'PO#00000000',
        'PO#0000000',
        'PO_000000000000',
        'PO_00000000000',
    ]

    template = random.choice(formats)
    text = ''

    for character in template:
        if character == 'A':
            text += random.choice(ascii_uppercase)
        elif character == '0':
            text += random.choice(digits)
        else:
            text += character

    return text


def get_id_long(config: DocxConfig, **kwargs) -> str:
    length_range: Tuple[int, int] = (10, 15)
    vocab = ascii_uppercase + digits + '-'

    text = ''.join(random.choice(vocab) for _ in range(random.randint(length_range[0], length_range[1])))
    return text


def get_id_short(config: DocxConfig, **kwargs) -> str:
    length_range: Tuple[int, int] = (3, 5)
    vocab = ascii_uppercase + digits + '-'

    text = ''.join(random.choice(vocab) for _ in range(random.randint(length_range[0], length_range[1])))
    return text


def get_id_huge(config: DocxConfig, **kwargs) -> str:
    length_range: Tuple[int, int] = (20, 30)
    vocab = ascii_uppercase + digits + '-'

    text = ''.join(random.choice(vocab) for _ in range(random.randint(length_range[0], length_range[1])))
    return text


def get_uuid(config: DocxConfig, **kwargs) -> str:
    num_groups = random.randint(4, 6)
    group_length = random.randint(4, 6)
    vocab = ascii_uppercase + digits

    sep = random.choice([' ', '-'])
    text = f'{sep}'.join(''.join(random.choice(vocab) for _ in range(group_length)) for _ in range(num_groups)) # Removes ending hyphen
    return text


def get_id_numeric(config: DocxConfig, **kwargs) -> str:
    length_range = (5, 9)
    vocab = digits
    text = ''.join(random.choice(vocab) for _ in range(random.randint(length_range[0], length_range[1])))
    return text


def get_id_tax(config: DocxConfig, **kwargs) -> str:
    return "sample_id_tax"


def get_id_bank(config: DocxConfig, **kwargs) -> str:
    return "sample_id_bank"


def get_price(config: DocxConfig, **kwargs) -> str:
    currency = kwargs.get("currency", '')
    whole = random.randint(0, 1000000)
    decimals = random.randint(0, 99)
    sign = np_choice(['', '-'], p=[0.9, 0.1])
    price = '0,00'

    if config.price_format == word_bank.PriceFormat.NO_TH_DOT_DEC:
        price = f'{sign}{whole}.{decimals}'
    elif config.price_format == word_bank.PriceFormat.NO_TH_COMMA_DEC:
        price = f'{sign}{whole},{decimals}'
    elif config.price_format == word_bank.PriceFormat.COMMA_TH_DOT_DEC:
        whole_str = f'{whole:,}'
        price = f'{sign}{whole_str}.{decimals}'
    elif config.price_format == word_bank.PriceFormat.DOT_TH_COMMA_DEC:
        whole_str = f'{whole:,}'.replace(',', '.')
        price = f'{sign}{whole_str},{decimals}'
    elif config.price_format == word_bank.PriceFormat.SPACE_TH_DOT_DEC:
        whole_str = f'{whole:,}'.replace(',', ' ')
        price = f'{sign}{whole_str},{decimals}'

    if config.price_show_currency:
        price = f'{currency} {price}' if config.price_currency_position else f'{price} {currency}'

    return price


def get_text(config: DocxConfig, **kwargs) -> str:
    return "sample_text"


def get_product_name(config: DocxConfig, **kwargs) -> str:
    return random.choice(word_bank.PRODUCT_NAMES)


def get_product_desc_short(config: DocxConfig, **kwargs) -> str:
    product_desc = ' '.join(config.faker.words(nb=random.randint(3, 5)))
    return product_desc


def get_product_desc(config: DocxConfig, **kwargs) -> str:
    product_desc = ' '.join(config.faker.words(nb=random.randint(7, 15)))
    return product_desc

def get_short_text(config: DocxConfig, **kwargs) -> str:
    return "sample_short_text"


def get_long_text(config: DocxConfig, **kwargs) -> str:
    return "sample_long_text"


def get_metric(config: DocxConfig, **kwargs) -> str:
    return random.choice(word_bank.METRICS)


def get_metric_group(config: DocxConfig, **kwargs) -> str:
    return random.choice(word_bank.METRIC_GROUPS)


def get_billing_term(config: DocxConfig, **kwargs) -> str:
    return random.choice(word_bank.BILLING_TERMS)


def get_bank_payment_method(config: DocxConfig, **kwargs) -> str:
    selection = random.choice(word_bank.BANK_PAYMENT_METHODS)
    return helpers.normalize_string(selection, FieldType.PaymentMethod, is_blank_p=0.0, config=config)


def get_bank_name(config: DocxConfig, **kwargs) -> str:
    selection = random.choice(word_bank.BANK_NAMES)
    return selection


def get_iban(config: DocxConfig, **kwargs) -> str:
    return config.faker.iban()


def get_swift(config: DocxConfig, **kwargs) -> str:
    return config.faker.swift()


def get_payment_term(config: DocxConfig, **kwargs) -> str:
    return random.choice(word_bank.PAYMENT_TERMS)


def get_payment_method(config: DocxConfig, **kwargs) -> str:
    selection = random.choice(word_bank.BANK_PAYMENT_METHODS)
    return helpers.normalize_string(selection, FieldType.PaymentMethod, is_blank_p=0.0, config=config)


def get_percentage(config: DocxConfig, **kwargs) -> str:
    pct = random.random()
    if config.pct_format:
        return f"{pct:.2f}"
    else:
        return f"{(pct * 100):.2f} %"


def get_date(config: DocxConfig, date_from: str = None, date_to: str = None, **kwargs) -> str:
    def str_time_prop(start: str, end: str, time_format: str, prop):
        """Get a time at a proportion of a range of two formatted times.

        start and end should be strings specifying times formatted in the
        given format (strftime-style), giving an interval [start, end].
        prop specifies how a proportion of the interval to be taken after
        start.  The returned time will be in the specified format.
        """

        stime = time.mktime(time.strptime(start, time_format))
        etime = time.mktime(time.strptime(end, time_format))

        ptime = stime + prop * (etime - stime)

        return time.strftime(time_format, time.localtime(ptime))

    date_format = config.date_format

    if not date_from:
        date_from = datetime(2020, 1, 1).strftime(date_format)
    if not date_to:
        date_to = datetime(2030, 12, 31).strftime(date_format)

    date_text = str_time_prop(date_from, date_to, date_format, random.random())
    return date_text


def get_date_range(config: DocxConfig, **kwargs):
    date_from = get_date(config)
    date_to = get_date(config, date_from=date_from)

    sep = config.date_range_sep
    prefix = 'from ' if sep in ['to', 'till'] else ''
    return f"{prefix}{date_from} {sep} {date_to}"


def get_currency(config: DocxConfig, **kwargs):
    return random.choice(word_bank.CURRENCY_TYPE)


def get_doc_type(config: DocxConfig, **kwargs):
    selection = random.choice(word_bank.DOCUMENT_TYPE)
    return helpers.normalize_string(selection, FieldType.DocType, is_blank_p=0.0, config=config)


def get_company_name(config: DocxConfig, **kwargs):
    return random.choice(word_bank.COMPANY_NAMES)


def get_number(config: DocxConfig, range: tuple = None, **kwargs):
    if range:
        number = random.randint(range[0], range[1])
    else:
        number = random.randint(0, 10000)

    if config.number_format == word_bank.NumberFormat.NO_TH:
        number = f'{number:,}'.replace(',', ' ')
    elif config.number_format == word_bank.NumberFormat.DOT_TH:
        number = f'{number:,}'.replace(',', '.')
    elif config.number_format == word_bank.NumberFormat.COMMA_TH:
        number = f'{number:,}'

    return number


def get_phone_number(config, **kwargs):
    prefix = str(random.randint(0, 99)).zfill(2)
    number = ''.join([str(random.randint(0, 9)) for _ in range(random.randint(8, 10))])

    return f"+{prefix} {number}"


def get_email(config: DocxConfig, **kwargs):
    return config.faker.company_email()


def get_name(config: DocxConfig, **kwargs):
    return config.faker.name()


def get_signature(config: DocxConfig, **kwargs):
    return ''


def get_work_title(config: DocxConfig, **kwargs):
    return config.faker.job()


def get_address(config: DocxConfig, **kwargs):
    return f'{config.faker.address()}\n{config.faker.city()}, {config.faker.country()}'


def get_index(config: DocxConfig, idx, **kwargs):
    return str(idx)


""" IMPORTANT: Do not declare any getter function after this line """
# sys.modules[__name__] gets a reference to the current module.
_REGISTRY = collect_funcs_from_module(sys.modules[__name__], suffix="get", default=lambda: print("Random generic data"))


def generate_value(datatype: FieldType, config, **kwargs):
    return _REGISTRY[datatype.value](config, **kwargs)
