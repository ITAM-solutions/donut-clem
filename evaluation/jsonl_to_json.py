"""
File name: jsonl_to_json
Author: Fran Moreno
Last Updated: 10/24/2025
Version: 1.0
Description: TOFILL
"""
import json
import os

from pathlib import Path

if __name__ == '__main__':
    DATA_PATH = Path(r'C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\test\synthetic')

    jsonl_file = DATA_PATH / 'metadata.jsonl'

    with open(jsonl_file, 'r', encoding='utf-8') as fp:
        file_content = fp.readlines()

    for line in file_content:
        json_obj = json.loads(line)
        f_name = json_obj['file_name']
        content = json.loads(json_obj['ground_truth'])['gt_parse']

        f_path = (DATA_PATH / f_name).with_suffix('.json')
        with open(f_path, 'w', encoding='utf-8') as fp:
            json.dump(content, fp, indent=4)


