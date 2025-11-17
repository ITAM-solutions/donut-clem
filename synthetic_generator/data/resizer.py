"""
File name: resizer
Author: Fran Moreno
Last Updated: 11/14/2025
Version: 1.0
Description: TOFILL
"""
from PIL import Image

from pathlib import Path
import os
DATASET_PATH = Path(r'dataset/Logos')


if __name__ == '__main__':
    ims = [name for i in os.listdir(DATASET_PATH) if (name := DATASET_PATH / i).suffix in ['.jpg', '.png', '.jpeg']]

    dest_folders = [Path('images') / i / 'samples' for i in os.listdir(Path(r'images'))]

    for dest_folder in dest_folders:
        placeholder_im = Image.open(dest_folder.parent / 'placeholder.png')
        expected_aspect_ratio = placeholder_im.height / placeholder_im.width
        print(expected_aspect_ratio)

        for idx, im_path in enumerate(ims):
            im = Image.open(im_path)
            center_h = im.height / 2
            expected_h = im.height * expected_aspect_ratio
            new_im = im.crop((0, center_h - expected_h // 2, im.width, center_h + expected_h // 2))
            new_im.save(dest_folder / (str(idx).zfill(4) + im_path.suffix))


