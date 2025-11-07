"""
File name: product_merging
Author: Fran Moreno
Last Updated: 11/7/2025
Version: 1.0
Description: TOFILL
"""
from typing import List

from clem.candidates.collector import ProductCandidate, FieldCandidate


def find_and_combine_partial_products(products: List[ProductCandidate]):
    """

    :return:
    """

    keeps_merging = True
    while keeps_merging:

        merged_list = []
        current_idx = 0

        while current_idx < len(products):
            if current_idx + 1 < len(products):  # Last product in list.
                if are_partials(products[current_idx], products[current_idx + 1]):
                    merged_list.append(merge_two_partial_products(products[current_idx], products[current_idx + 1]))
                    current_idx += 2
                else:
                    merged_list.append(products[current_idx])
                    current_idx += 1
            else:  # Last product in list
                merged_list.append(products[current_idx])
                current_idx += 1

        if len(merged_list) < len(products):
            products = merged_list.copy()
        else:
            keeps_merging = False

    return products


def are_partials(product1: ProductCandidate, product2: ProductCandidate) -> bool:
    """
    TODO define
    :param product1:
    :param product2:
    :return:
    """
    fields_filled_product1 = set(product1.filled_fields)
    fields_filled_product2 = set(product2.filled_fields)

    common_filled_fields = fields_filled_product1.intersection(fields_filled_product2)
    return len(common_filled_fields) == 0


def merge_two_partial_products(partial1: ProductCandidate, partial2: ProductCandidate) -> ProductCandidate:
    """
    TODO define
    :param partial1:
    :param partial2:
    :return:
    """
    fields_partial1 = partial1.filled_fields
    fields_partial2 = partial2.filled_fields

    return ProductCandidate(
        **fields_partial1,
        **fields_partial2
    )