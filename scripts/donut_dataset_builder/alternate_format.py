"""
File name: alternate_format
Author: Fran Moreno
Last Updated: 7/29/2025
Version: 1.0
Description: TOFILL
"""
from collections import defaultdict
from pathlib import Path
import json

def alternate_json_lists_format(d):
    if isinstance(d, dict):
        return {k: alternate_json_lists_format(v) for k, v in d.items()}
    elif isinstance(d, list):
        d_mod = defaultdict(list)
        for el in d:
            for k in el:
                d_mod[k].append(el[k])
        return dict(d_mod)
    else:
        return d


if __name__ == '__main__':
    DATASET_PATH = Path(r"C:\Users\FranMoreno\datasets\tagged_data_extractions\donut_54_samples_json_format_fixed\dataset")

    json_files = [i for i in DATASET_PATH.glob('**/*.json') if i.stem != '.file_metadata']
    for json_file in json_files:
        with open(json_file, 'r') as fp:
            json_content = json.load(fp)
        with open(json_file, 'w') as fp:
            json_content_mod = alternate_json_lists_format(json_content)
            json.dump(json_content_mod, fp, indent=4)
