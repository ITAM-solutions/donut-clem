"""
File name: output_template
Author: Fran Moreno
Last Updated: 6/4/2025
Version: 1.0
Description: TOFILL
"""
import json

from pathlib import Path

shared_fields_mapping = {
    "Purchase invoice reference": "id",
    "Orderform/Quote": "id",
    "Purchase Date": "date",
    "Purchase date": "date",
    "Purchase currency": "cur",
    "Purchase currency (Invoice)": "cur",
    "Purchase currency (Snow)": "cur",
    "Purchase currency (snow)": "cur",
    "Purchase Currency (SNOW)": "cur",
    "Vendor/Reseller": "vendor",
    "Vendor/reseller": "vendor",
    "Sales Order Number": "po",
    "Legal Organisation": "corp",
    "Legal organisation": "corp",
    "Supplier": "pub",
    "Manufacturer": "pub",
    "License Type": "ltype",
    "License type": "ltype",
    "Agreement type": "ctype",
    "Assignment Type": "atype",
    "Assignment type": "atype",
}

non_shared_fields_mapping = {
    "Application name": "name",
    "Product Description": "name",
    "Product description": "name",
    "SKU": "sku",
    "Metric": "met",
    "Specific Metric": "metgr",
    "Specified Metric": "metgr",
    "Amount of licenses": "qty",
    "License quantity": "qty",
    "Number of licenses": "qty",
    "Number of licenses ": "qty",
    "Quantity": "qty",
    "Unit Price": "unpr",
    "Purchase Price": "totpr",
    "Purchase Price (Invoice)": "totpr",
    "Purchase Price (SNOW)": "totpr",
    "Purchase Price (Snow)": "totpr",
    "Purchase price": "totpr",
    "Purchase price in EUR": "totpr",
    "purchase price (snow)": "totpr",
    "Subscription valid from": "drom",
    "Valid from": "drom",
    "Subscription valid to": "dto",
    "Valid to": "dto",
    "Is subscription": "issub",
    "Subscription agreement": "issub"
}


readable_names = {
    'id': 'Invoice ID',
    'date': 'Invoice Date',
    'cur': 'Currency',
    'vendor': 'Vendor/Reseller',
    'po': 'Purchase Order Number',
    'corp': 'Corporate Unit Name',
    'pub': 'Publisher',
    'ltype': 'License Type',
    'ctype': 'Contract Type',
    'atype': 'Agreement Type',
    'name': 'Product Name',
    'sku': 'SKU',
    'met': 'Metric',
    'metgr': 'Metric Group',
    'qty': 'Quantity',
    'unpr': 'Unit Price',
    'totpr': 'Total Price',
    'drom': '(Subscription) valid From',
    'dto': '(Subscription) valid To',
    'issub': 'Is a Subscription?'
}


class OutputJSON:
    def __init__(self):
        self.template = {
            'id': None,
            'date': None,
            'cur': None,
            'vendor': None,
            'po': None,
            'corp': None,
            'pub': None,
            'ltype': None,
            'ctype': None,
            'atype': None,
            'products': []
        }

        self.item_template = {
            'name': None,
            'sku': None,
            'met': None,
            'metgr': None,
            'qty': None,
            'unpr': None,
            'totpr': None,
            'drom': None,
            'dto': None,
            'issub': None
        }

    def add_item(self, item_data: dict):
        new_item = self.item_template.copy()
        for name_raw, value in item_data.items():
            name = shared_fields_mapping.get(name_raw)
            if name and not self.template.get(name):
                self.template[name] = value

            else:  # Check if part of item values
                name = non_shared_fields_mapping.get(name_raw)
                if name:
                    new_item[name] = value
                # Else, name_raw is not in the mapping, so it is not important

        self.template['products'].append(new_item)

    def save(self, target_path: Path):
        with open(target_path, 'w') as fp:
            json.dump(self.template, fp, indent=4)

    def load_existing_json(self, image_path: Path):
        json_path = image_path.with_suffix('.json')

        if not json_path.exists():
            return

        with open(json_path, 'r') as fp:
            data = json.load(fp)

        self.template = data
