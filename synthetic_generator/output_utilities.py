"""
File name: output_utilities
Author: Fran Moreno
Last Updated: 10/16/2025
Version: 1.0
Description: Multiple functions to process the output from the Synthetic Data Generator:
"""
import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from typing import List

def count_from_output_records():
    """
    Will count the total number of samples through all the records that exist in the output folder.

    :return:
    """
    output_dir = Path('output')  # Update if needed!
    count = len(list(output_dir.glob('**/*.png')))
    print(f"There are {count} synthetic samples in the output folder.")


def combine_output_records():
    """
    Collects all the valid samples from the output folder into a single directory with the following format:
    combined_dir/
        000000_000/
            000000_000.png
            000000_000.json
        000001_000/
            000001_000.png
            000001_000.json
        ...

    :return:
    """
    # Define path variables
    source_dir = Path('output')
    dest_dir = Path('output_prepared')
    dataset_folder = dest_dir / 'synth_dataset_to_be_formatted'

    # If dataset generated previously, remove and execute again
    if list(dest_dir.glob('*')):
        raise Exception("Please clean up the destination folder before running the combination utility.")

    os.mkdir(dataset_folder)

    # Collect all record directories (full path)
    records: List[Path] = [source_dir / record_path for record_path in os.listdir(source_dir)]

    # Iterate over records and give a normalized name using an index
    sample_idx = 0
    for record_path in tqdm(records, total=len(records)):
        template_folders: List[Path] = [record_path / template_name for template_name in os.listdir(record_path)]
        for template_folder in template_folders:
            ims: List[Path] = list(template_folder.glob('*.png'))
            for im_path in ims:
                json_path = im_path.with_suffix('.json')
                if not json_path.exists():
                    continue  # Invalid sample, no json data

                # Valid sample, create new folder in dataset; save image, json and metadata with template name
                base_name = str(sample_idx).zfill(6) + '_000'

                folder_path = dataset_folder / base_name
                os.mkdir(folder_path)

                shutil.copy(im_path, folder_path / (base_name + '.png'))
                shutil.copy(json_path, folder_path / (base_name + '.json'))

                # create metadata: {"template_name": "template_name"}
                with open(folder_path / '.file_metadata.json', 'w') as fp:
                    json.dump({"template_name": str(im_path.with_suffix('.docx'))}, fp)

                sample_idx += 1


if __name__ == '__main__':
    # Uncomment the function you need
    pass

    # count_from_output_records()
    combine_output_records()