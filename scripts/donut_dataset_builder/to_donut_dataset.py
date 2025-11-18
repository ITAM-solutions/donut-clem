"""
File name: to_donut_dataset
Author: Fran Moreno
Last Updated: 8/12/2025
Version: 1.0
Description: TOFILL
"""
import os
from pathlib import Path
import shutil
import json
import random


ORG_DATA_PATH = Path(r'C:\Users\FranMoreno\ITAM_software\repositories\CLEM\remote\CLEM-ai\donut\synthetic_generator\output_prepared\synth_dataset_to_be_formatted')
DEST_DATA_PATH = Path(r"C:\Users\FranMoreno\datasets\train_dataset_16Oct\synth_data")

pct = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}


def remove_keys(data: dict) -> None:
    keys_to_remove = (
    )

    for k in keys_to_remove:
        if k in data.keys():
            data.pop(k)


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


def from_synthetic_data(shuffle: bool = True):
    samples = [ORG_DATA_PATH / sample for sample in os.listdir(ORG_DATA_PATH)]

    # Shuffle to make sure that data is mixed as much as possible
    if shuffle:
        random.shuffle(samples)
        print("First samples to make sure it is shuffled correctly:\n", samples[:10])

    num_samples = len(samples)
    print("Samples to process:", num_samples)

    start_idx = 0
    for split in pct:
        split_len = int(num_samples * pct[split])

        split_path = DEST_DATA_PATH / split
        gt_list = []
        images = []

        for sample_path in samples[start_idx:start_idx + split_len]:
            im_path = sample_path / (sample_path.stem + '.png')
            json_path = sample_path / (sample_path.stem + '.json')

            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            json_data = normalize_json_data(json_data)
            # remove_keys(json_data)

            images.append(im_path)
            jsonl_line = {"gt_parse": json_data}
            gt_list.append(jsonl_line)

        os.makedirs(split_path)
        with open(split_path / "metadata.jsonl", "w", encoding='utf-8') as f:
            for image, gt_parse in zip(images, gt_list):
                line = {"file_name": image.name, "ground_truth": json.dumps(gt_parse)}
                f.write(json.dumps(line) + '\n')
                shutil.copyfile(image, split_path / image.name)

        start_idx = split_len


def from_real_data():
    samples = [Path(ORG_DATA_PATH) / i for i in os.listdir(ORG_DATA_PATH)]
    num_samples = len(samples)
    print(num_samples)

    start_idx = 0
    for split in pct:
        split_len = int(num_samples * pct[split])

        split_path = DEST_DATA_PATH / split
        gt_list = []
        images = []

        for sample_path in samples[start_idx:start_idx + split_len]:
            for im_path in sample_path.glob("*.png"):
                json_path = im_path.with_suffix('.json')
                if not json_path.exists():
                    continue  # Skip sample, no ground-truth json file found.

                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                remove_keys(json_data)

                images.append(im_path)
                jsonl_line = {"gt_parse": json_data}
                gt_list.append(jsonl_line)


        os.makedirs(split_path)
        with open(split_path / "metadata.jsonl", "w") as f:
            for image, gt_parse in zip(images, gt_list):
                line = {"file_name": image.name, "ground_truth": json.dumps(gt_parse)}
                f.write(json.dumps(line) + '\n')
                shutil.copyfile(image, split_path / image.name)

        start_idx = split_len


if __name__ == '__main__':
    from_synthetic_data()
    # from_real_data()


