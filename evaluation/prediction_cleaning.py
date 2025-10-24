"""
File name: prediction_cleaning
Author: Fran Moreno
Last Updated: 10/23/2025
Version: 1.0
Description: TOFILL
"""
from collections import defaultdict


def normalize_products_structure2(data: dict) -> dict:
    """
    Converts the products data structure to a dictionary of lists in case it is
    a list of dictionaries. That is the real format that donut is usng as input
    and output.

    :param data:
    :return:
    """
    products = data.get('products')

    if isinstance(products, list):
        products_d = defaultdict(list)
        for product in products:
            for field in product:
                products_d[field].append(product[field])
        data['products'] = dict(products_d)
        return data
    elif isinstance(products, dict):
        # When just one product, values can be a string instead of list. Normalize that as well
        for field, values in products.items():
            if isinstance(values, str):
                products[field] = [values]
        return data
    elif isinstance(products, str):
        data['products'] = {}
        return data
    else:  # data['products'] = None
        data['products'] = {}
        return data


def normalize_products_structure(data: dict) -> dict:
    """
    Converts the products data structure to list in case it is a dictionary, to be aligned with the
    structure followed in the ground-truth.

    :param data:
    :return:
    """
    products = data.get('products')

    if products and type(products) == dict:

        # Find maximum number of products
        max_num_products = 1
        for value in products.values():
            if type(value) == list and len(value) > max_num_products:
                max_num_products = len(value)

        # Extend missing values to keep output normalized
        product_list = [dict() for _ in range(max_num_products)]
        for key, org_value in products.items():
            if type(org_value) == str:
                values = [org_value] + [None] * (max_num_products - 1)
            else:
                values = org_value + [None] * (max_num_products - len(org_value))

            # Update the output with the new values
            for idx, value in enumerate(values):
                product_list[idx][key] = value

        data['products'] = product_list
        return data

    # If products is not dict, return the original data.
    return data

def remove_unused_fields(data: dict) -> dict:
    """
    The output may contain some fields that are not in the ground-truth. To be sure that the metrics are as
    accurate as possible, those unused fields will be removed.

    :param data:
    :return:
    """
    unused_shared_keys = ['ltype', 'ctype', 'atype', 'pub']
    unused_product_keys = ['issub']

    for k in unused_shared_keys:
        if k in data:
            del data[k]

    products = data.get('products', {})
    for pk in unused_product_keys:
        if pk in products:
            del products[pk]

    # Remove fields that were named differently
    if 'drom' in products:
        products['dfrom'] = products['drom']
        del products['drom']

    return data


def normalize_empty_values(data: dict) -> dict:
    for k, v in data.items():
        if v == 'None':
            data[k] = None

    products = data.get('products', {})
    data['products'] = {
        field: [None if value == 'None' else value for value in values]
        for field, values in products.items()
    }

    return data


def get_empty_response() -> dict:
    return {
        "id": None,
        "date": None,
        "cur": None,
        "vendor": None,
        "po": None,
        "corp": None,
        "products": {},
    }
