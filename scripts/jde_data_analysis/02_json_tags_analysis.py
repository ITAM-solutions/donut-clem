"""
File name: json_tags_analysis
Author: Fran Moreno
Last Updated: 6/3/2025
Version: 1.0
Description:

Run after executing `excel_analysis.py` to collect a set of all the unique keys found in the JSON files that were
extracted from the original Dataset excel set.

NOTE: This script is part of the JDE dataset analysis and preparation for Donut.
"""
import json
import tqdm
from pathlib import Path


def main():
    DATASET_PATH = Path(r'C:\Users\FranMoreno\datasets\c_and_o_outputs')

    json_files = DATASET_PATH.glob('**/*.json')

    tags_set = set()
    for json_file in tqdm.tqdm(json_files):
        with open(json_file, 'r') as fp:
            content = json.load(fp)
            extracted_values = content['values']

            if 'Supplier' in extracted_values.keys():
                print(json_file)

            tags_set |= set(extracted_values.keys())

    tags_set.remove('null')
    print(len(tags_set))
    tags_set = sorted(tags_set)
    print(sorted(tags_set))

if __name__ == '__main__':
    main()


