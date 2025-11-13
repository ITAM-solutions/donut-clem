"""
File name: generators
Author: Fran Moreno
Last Updated: 8/27/2025
Version: 1.0
Description: TOFILL
"""
from enum import Enum
import random
from typing import List

from synthetic_generator.generators import value_generators
from synthetic_generator.utils.config import DocxConfig
import synthetic_generator.utils.translator as translator
import synthetic_generator.word_bank as word_bank
import synthetic_generator.generators.helpers as helpers
from synthetic_generator.utils.datatypes import FieldType


class DataFormat(str, Enum):
    KV = 'kv'
    TABLE = 'table'
    TEXT_LINE = 'text_line'


def gen_addr(config: DocxConfig, **kwargs):
    return {
        "raw": value_generators.generate_value(FieldType.Address, config),
        "type": "text"
    }

def gen_addr_inline(config: DocxConfig, **kwargs):
    return {
        "raw": value_generators.generate_value(FieldType.Address, config).replace('\n', ', '),
        "type": "text"
    }

def gen_doc_ids(config: DocxConfig, **kwargs):
    data = {
        f_id: {
            'lbl': helpers.normalize_string(random.choice(labels), FieldType.Text, 0.0, config),
            'val': helpers.normalize_string(value_generators.generate_value(dtype, config), dtype, 0.0, config)
        }
        for f_id, (dtype, labels) in word_bank.DOCUMENT_IDS.items()
    }

    return {
        "raw": data,
        "type": "kv",
    }


def gen_billing_terms(config: DocxConfig, **kwargs):
    """ billingTerms: text_line """

    return {
        "raw": translator.translate(value_generators.get_billing_term(config), config.lang),
        "type": "text_line"
    }


def gen_payment_terms(config: DocxConfig, **kwargs):
    """ billingTerms: text_line """

    return {
        "raw": translator.translate(value_generators.get_payment_term(config), config.lang),
        "type": "text_line"
    }


def _generate_main_invoice_fields(config: DocxConfig, all_inv_main_fields: dict, qty: int = None):
    """
    Generate random values for each main invoice field, and randomly set those values as blank with low probability.
    The second step is performed because sometimes these fields may appear empty in invoices, but this is not a
    common thing.

    :param config:
    :param all_inv_main_fields:
    :param qty:
    :return:
    """
    as_blank_probability = 0.05

    main_fields = {}
    for field_id, (dtype, labels) in all_inv_main_fields.items():
        selected_label = random.choice(labels)
        generated_value = value_generators.generate_value(dtype, config)
        normalized_label = helpers.normalize_string(selected_label, FieldType.Text, 0.0, config)
        normalized_value = helpers.normalize_string(generated_value, dtype, as_blank_probability, config)
        main_fields[field_id] = (normalized_value, normalized_label)

    return helpers.unpack_tuples_to_dict(main_fields)


def _generate_additional_invoice_fields(config: DocxConfig, qty: int = None):
    additional_fields = word_bank.INVOICE_ADDITIONAL_FIELDS.copy()
    selected_fields = {}

    # If given quantity, use that. Else, select random quantity.
    qty = min(qty, len(additional_fields)) if qty else random.choice(list(range(len(additional_fields))))
    for _ in range(qty):
        field_id, (dtype, labels) = random.choice(list(additional_fields.items()))
        selected_tag = random.choice(labels)

        selected_fields[field_id] = (
            helpers.normalize_string(value_generators.generate_value(datatype=dtype, config=config),
                dtype, 0.3, config),
            helpers.normalize_string(selected_tag, FieldType.Text, 0.0, config)
        )

        additional_fields.pop(field_id)

    selected_fields.update({field_id: (' ', random.choice(labels)) for field_id, (_, labels) in additional_fields.items()})

    return helpers.unpack_tuples_to_dict(selected_fields)


def gen_inv_info(config: DocxConfig, quantity: int, gt_data: dict, **kwargs) -> dict:
    """  invInfo: kv

    Generates the invoice information section. This section usually contains the ground-truth general info of the
    invoice, and optionally some additional fields.

    This function will save the main invoice info fields into the ground-truth structure received as part of `**kwargs`.

    :param config:
    :param inv_fields:
    :param gt_data: to be filled with the resulting main invoice field values, indexed by their IDs.
    :param quantity: will dictate how many fields can be generated IN TOTAL.
    :return:
    """
    all_main_invoice_fields = word_bank.INVOICE_MAIN_FIELDS

    fields_qty = additional_fields_qty = None
    if quantity:
        fields_qty = int(quantity * 0.6)
        additional_fields_qty = quantity - fields_qty

    gt_data.update({k: '' for k in all_main_invoice_fields})

    main_fields = _generate_main_invoice_fields(config, all_main_invoice_fields, fields_qty)
    additional_fields = _generate_additional_invoice_fields(config, qty=additional_fields_qty)

    gt_data.update({field_id: field_info['val'] for field_id, field_info in main_fields.items()})

    raw_data = {
        **main_fields,
        **additional_fields,
    }

    return {
        'raw': raw_data,
        'type': 'kv'
    }


def _generate_product(
        config: DocxConfig,
        fields_group: dict,
        gt_products: list,
        empty_product: bool=False,
        **kwargs,
) -> dict:

    if empty_product:
        fields_values = {
            k: (
                ' ',
                helpers.normalize_string(v[1], FieldType.Text, 0.0, config)
            )
            for k, v in fields_group.items()}
    else:
        fields_values = {
            k: (
                helpers.normalize_string(value_generators.generate_value(v[0], config=config, **kwargs), v[0], 0.0, config),
                helpers.normalize_string(v[1], FieldType.Text, 0.0, config)
            )
            for k, v in fields_group.items()
        }

        gt_product = {k: '' for k in word_bank.PRODUCT_MAIN_FIELDS}
        gt_product.update({k: str(v[0]) for k, v in fields_values.items() if k in word_bank.PRODUCT_MAIN_FIELDS})

        # Special case: date range to dfrom and dto fields
        if fields_values.get('subscription_period'):
            dfrom, dto = _get_dates_from_date_range(fields_values['subscription_period'][0], config.date_format)
            gt_product['dfrom'] = dfrom
            gt_product['dto'] = dto

        gt_products.append(gt_product)

    return fields_values


def _get_dates_from_date_range(date_string, date_format):
    dates = helpers.find_dates_in_string(date_string, date_format)
    if len(dates) == 2:
        dfrom, dto = dates
    elif len(dates) == 1:
        dfrom, dto = dates[0], None
    else:
        dfrom, dto = None, None
    return dfrom, dto


def _get_random_dict_subset(config, dictionary: dict, qty=None):
    subset_size = qty if qty is not None else random.randint(0, len(dictionary))
    return dict(random.sample(sorted(dictionary.items()), subset_size))


def gen_tbl_products(config: DocxConfig, quantity: int, missing: int, gt_data: dict, **kwargs):
    """ tblProducts: table

    :param config:
    :param gt_data:
    :param kwargs:
    :return:
    """
    quantity = quantity if quantity is not None else 0
    raw_data = {
        'ids': [],
        'labels': [],
        'tb_items': []
    }

    gt_currency = gt_data.get("cur", '')
    gt_products = gt_data.get('products', [])

    fields_selection = config.product_field_set
    raw_data['ids'] = list(fields_selection.keys())
    raw_data['labels'] = [
        helpers.normalize_string(i[1], FieldType.Text, 0.0, config)
        for i in fields_selection.values()
    ]
    for idx in range(quantity):
        product_data = _generate_product(config, fields_selection, gt_products, currency=gt_currency, idx=idx)
        raw_data['tb_items'].append(product_data)

    print("Products:", quantity)
    print("Empty rows:", missing)
    for _ in range(quantity, quantity + missing):
        empty_product = _generate_product(config, fields_selection, gt_products, empty_product=True, currency=gt_currency)
        raw_data['tb_items'].append(empty_product)

    gt_data['products'] = gt_products

    raw_data['tb_items'] = [
        {
            'lbl': f'item_{i}',
            'fields': [f[0] for f in item.values()],
        }
        for i, item in enumerate(raw_data['tb_items'])
    ]

    return {
        'raw': raw_data,
        'type': 'table'
    }


def gen_products_shared(config: DocxConfig, **kwargs):
    data = {
        f_id: {
            'lbl': helpers.normalize_string(random.choice(labels), FieldType.Text, 0.0, config),
            'val': helpers.normalize_string(value_generators.generate_value(dtype, config), dtype, 0.0, config)
        }
        for f_id, (dtype, labels) in word_bank.PRODUCT_SHARED_FIELDS.items()
    }

    return {
        "raw": data,
        "type": "kv",
    }


def gen_pr_metric(config: DocxConfig, **kwargs):
    return {
        "raw": value_generators.generate_value(FieldType.Metric, config),
        "type": "text"
    }


def gen_pr_metric_group(config: DocxConfig, **kwargs):
    return {
        "raw": value_generators.generate_value(FieldType.MetricGroup, config),
        "type": "text"
    }


def gen_pr_date_from(config: DocxConfig, **kwargs):
    return {
        "raw": value_generators.generate_value(FieldType.Date, config),
        "type": "text"
    }


def gen_pr_date_to(config: DocxConfig, **kwargs):
    return {
        "raw": value_generators.generate_value(FieldType.Date, config),
        "type": "text"
    }


def gen_pr_unpr(config: DocxConfig, **kwargs):
    return {
        "raw": value_generators.generate_value(FieldType.Price, config),
        "type": "text"
    }


def gen_pr_totpr(config: DocxConfig, **kwargs):
    return {
        "raw": value_generators.generate_value(FieldType.Price, config),
        "type": "text"
    }


def gen_signature(config: DocxConfig, blank_probability: float = None, **kwargs):
    """ signature: kv

    :param config:
    :param blank_probability:
    :param kwargs:
    :return:
    """
    data = {
        f_id: {
            'lbl': helpers.normalize_string(random.choice(labels), FieldType.Text, 0.0, config),
            'val': helpers.normalize_string(value_generators.generate_value(dtype, config), dtype, 0.6, config)
        }
        for f_id, (dtype, labels) in word_bank.SIGNATURE_INFO_FIELDS.items()
    }

    return {
        "raw": data,
        "type": "kv",
    }


def gen_signature_empty(config: DocxConfig, **kwargs):
    """ signatureEmpty: kv

    - Authorized by
    - Authorized signature
    - Signature date
    - Additional signature
    - Some text

    :param config:
    :param kwargs:
    :return:
    """
    return gen_signature(config, blank_probability=0.0)


def gen_bank_info(config: DocxConfig, **kwargs):
    """

    - Bank Name
    - Bank Addresss
    - Bank Country
    - Bank Account owner (name)
    - Bank Account Number
    - IBAN & SWIFT Code
    - VAT Code
    - Company registration number

    :param config:
    :param kwargs:
    :return:
    """
    data = {
        f_id: {'lbl': random.choice(labels), 'val': value_generators.generate_value(dtype, config)}
        for f_id, (dtype, labels) in word_bank.BANK_INFO_FIELDS.items()
    }

    return {
        "raw": data,
        "type": "kv",
    }


def gen_contact_info(config: DocxConfig, **kwargs) -> dict:
    """ contactInfo: kv
    - Contact name
    - Contact email address
    - Contact phone
    -

    :param config:
    :param kwargs:
    :return:
    """
    p = 0.3

    data = {
        f_id: {
            'lbl': helpers.normalize_string(random.choice(labels), FieldType.Text, 0.0, config),
            'val': helpers.normalize_string(value_generators.generate_value(dtype, config), dtype, p, config),
        }
        for f_id, (dtype, labels) in word_bank.CONTACT_INFO_FIELDS.items()
    }

    return {
        "raw": data,
        "type": "kv",
    }


def get_text_lines(config: DocxConfig, quantity: int, **kwargs):
    """

    Quantity here is used to specify the number of separate lines to generate.

    :param config:
    :param quantity:
    :param kwargs:
    :return:
    """
    pass


def _get_text(num_sentences: int, language: str) -> List[str]:
    sentences = [
        translator.translate(
            ' '.join([random.choice(word_bank.WORDS) for _ in range(random.randint(10, 25))]),
            language
        ) for _ in range(num_sentences)
    ]
    return sentences


def gen_text_prose(config: DocxConfig, quantity: int, **kwargs):
    """

    Quantity here is used to specify the number of sentences in the text. The number of words in each sentence will
    be randomly selected.

    :param config:
    :param quantity:
    :param kwargs:
    :return:
    """
    sentences = _get_text(quantity, config.lang)
    prose = '. '.join([s.capitalize() for s in sentences])

    return {
        "raw": prose,
        "type": "text"
    }

def gen_text_paragraphs(config: DocxConfig, quantity: int, **kwargs):
    """

    :param config:
    :param quantity:
    :param kwargs:
    :return:
    """
    data = '\n\n'.join(
        [
            '. '.join([s.capitalize() for s in _get_text(random.randint(3, 8), config.lang)])
            for _ in range(quantity)
        ]
    )

    return {
        "raw": data,
        "type": "text",
    }


def gen_section_title(config: DocxConfig, doc_sections: List[str], **kwargs):
    available_sections = list(set(word_bank.DOCUMENT_SECTIONS).difference(set(doc_sections)))
    next_idx = len(doc_sections) + 1

    current_section = translator.translate(random.choice(available_sections), config.lang)
    text = f"{next_idx}. {current_section}"

    doc_sections.append(text)
    return {
        "raw": text,
        "type": "text"
    }


def gen_text_sections(config: DocxConfig, quantity: int, doc_sections: List[str], **kwargs):
    """
    textSections: kv

    :param config:
    :param quantity: tells how much sections to include.
    :param kwargs:
    :return:
    """


    sections = {
        idx + 1: {
            "lbl": gen_section_title(config, doc_sections, **kwargs)['raw'],
            "val": gen_text_prose(config, random.randint(3, 6))["raw"]
        }
        for idx in range(quantity)
    }

    return {
        "raw": sections,
        "type": "kv"
    }



def gen_page_idx(config: DocxConfig, **kwargs):
    total_pages = int(value_generators.generate_value(FieldType.Number, config, range=(1, 20), **kwargs))
    current_page = random.randint(1, total_pages)

    base_text = config.page_idx_format
    text = base_text.replace('#x', str(current_page)).replace('#y', str(total_pages))
    text = translator.translate(text, config.lang)
    return {
        "raw": text,
        "type": "text",
    }


def gen_id_short(config: DocxConfig, **kwargs):
    text = value_generators.generate_value(FieldType.IDShort, config)
    return {
        "raw": text,
        "type": "text",
    }

def gen_price(config: DocxConfig, **kwargs):
    text = value_generators.generate_value(FieldType.Price, config)
    return {
        "raw": text,
        "type": "text",
    }


def get_text_sections(config: DocxConfig, quantity: int, **kwargs):
    pass
