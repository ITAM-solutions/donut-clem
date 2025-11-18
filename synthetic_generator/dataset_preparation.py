"""
File name: dataset_preparation
Author: Fran Moreno
Last Updated: 11/18/2025
Version: 1.0
Description: TOFILL
"""
import os
import shutil
import random
import json
from typing import Dict
from pathlib import Path
from tqdm import tqdm

RESULTS_PATH = Path(r"output")
OUTPUT_PATH = Path("gathered")

def gather_by_template_name(results_path: Path):
    template_names = [i.stem for i in Path(r"templates").glob('**/*.docx')]

    sample_id = 0
    for template_name in template_names:
        im_samples = [i for i in results_path.glob(f'**/{template_name}/*.png') if i.with_suffix('.json').exists()]
        json_samples = [i.with_suffix('.json') for i in im_samples]

        template_new_folder = OUTPUT_PATH / template_name
        template_new_folder.mkdir(parents=True, exist_ok=True)
        for im_sample, json_sample in zip(im_samples, json_samples):
            base_name = OUTPUT_PATH / template_name / str(sample_id).zfill(6)
            shutil.move(im_sample, base_name.with_suffix(im_sample.suffix))
            shutil.move(json_sample, base_name.with_suffix(json_sample.suffix))
            sample_id += 1

    # Some logging
    total_train, total_val, total_test = 0, 0, 0
    for folder in os.listdir(OUTPUT_PATH):
        num_samples = len(list((OUTPUT_PATH / folder).glob('*.png')))
        train, val, test = int(num_samples * 0.75), int(num_samples * 0.10), int(num_samples * 0.15)
        print(f"{folder}: {len(list((OUTPUT_PATH / folder).glob('*.png')))} samples ({train}, {val}, {test})")
        total_train += train
        total_val += val
        total_test += test
    print("======")
    print("TOTAL FOR TRAIN:", total_train)
    print("TOTAL FOR VALIDATION:", total_val)
    print("TOTAL FOR TEST:", total_test)


def normalize_json_data(json_data: dict) -> dict:
    for key, value in json_data.items():
        if value == '':
            json_data[key] = None
        elif type(value) == str:
            json_data[key] = value.strip()

    if json_data.get('products'):
        for product in json_data['products']:
            for pkey, pvalue in product.items():
                if pvalue == '':
                    product[pkey] = None
                elif type(pvalue) == str:
                    product[pkey] = pvalue.strip()

    return json_data


def to_donut_dataset(data_path: Path, output_path: Path, splits: Dict[str, float]):
    for template in tqdm(os.listdir(data_path)):
        samples = list((data_path / template).glob('*.png'))
        random.shuffle(samples)  # To randomize

        start_idx = 0
        for split_name, split_pct in splits.items():
            split_len = int(len(samples) * split_pct)
            split_path = output_path / split_name

            gt_list = []
            im_list = []

            for sample in samples[start_idx: start_idx + split_len]:
                gt = sample.with_suffix('.json')

                with open(gt, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)

                gt_data = normalize_json_data(gt_data)

                im_list.append(sample)
                jsonl_line = {"gt_parse": gt_data}
                gt_list.append(jsonl_line)

            split_path.mkdir(parents=True, exist_ok=True)
            with open(split_path / "metadata.jsonl", "a+", encoding="utf-8") as f:
                for image, gt_parse in zip(im_list, gt_list):
                    line = {"file_name": image.name, "ground_truth": json.dumps(gt_parse)}
                    f.write(json.dumps(line) + '\n')
                    shutil.copyfile(image, split_path / image.name)

            start_idx = split_len


if __name__ == '__main__':
    # gather_by_template_name(RESULTS_PATH)
    to_donut_dataset(
        data_path=Path("gathered"),
        output_path=Path(r"C:\Users\FranMoreno\datasets\train_dataset_18Nov\synth_data"),
        splits={
            "train": 0.75,
            "validation": 0.10,
            "test": 0.15
        }
    )

