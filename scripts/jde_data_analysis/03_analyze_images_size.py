"""
File name: retrieve_dataset_info
Author: Fran Moreno
Last Updated: 6/12/2025
Version: 1.0
Description: TOFILL
"""
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pad
import numpy as np
import torch

def normalize_image(im: Image, expected_size: list[int]):

    im_padded = pad(im, [50, 50], fill=255)

    plt.figure()
    plt.subplot(121); plt.imshow(im)
    plt.subplot(122); plt.imshow(im_padded)
    plt.show()


def get_images_sizes_in_dataset(data_path: Path) -> dict:
    """
    retrieves a dictionary containing all the image sizes that were found in the given dataset, sorted by amount
    :param data_path:
    :return:
    """
    ims = data_path.glob('**/*.png')
    ims_sizes = dict()
    for im_path in tqdm(ims):
        im = Image.open(im_path)

        # im_padded = resize_and_pad_image()

        w, h = im.size
        label = f'{w}x{h}'
        print(f'{im_path.stem}: {label}')
        if label in ims_sizes:
            ims_sizes[label] += 1
        else:
            ims_sizes[label] = 1

    return ims_sizes


def plot_sizes(sizes_dict: dict) -> None:
    print(json.dumps(sizes_dict, indent=4, sort_keys=True))

    heights = []
    widths = []
    times_each = []
    for k, v in sizes_dict.items():
        h, w = k.split('x')
        heights.append(h)
        widths.append(w)
        times_each.append(v)

    plt.figure()
    plt.subplot(121)
    plt.bar(heights, times_each)
    plt.subplot(122)
    plt.bar(widths, times_each)
    plt.show()


def resize_and_pad_image(im: Image, expected_size: list[int]):
    w, h = im.size

    print("Before: ", im.size)
    im.thumbnail(expected_size, Image.Resampling.LANCZOS)
    w_rs, h_rs = im.size
    total_pad_x = expected_size[0] - w_rs
    total_pad_y = expected_size[1] - h_rs

    pad_x = total_pad_x / 2
    pad_y = total_pad_y / 2

    pad_l = pad_x if pad_x % 1 == 0 else pad_x + 0.5
    pad_t = pad_y if pad_y % 1 == 0 else pad_y + 0.5
    pad_r = pad_x if pad_x % 1 == 0 else pad_x - 0.5
    pad_b = pad_y if pad_y % 1 == 0 else pad_y - 0.5
    padding = [int(pad_l), int(pad_t), int(pad_r), int(pad_b)]

    im_padded = pad(im, padding, fill=255)
    # return im_padded
    print("After: ", im_padded.size)
    plt.figure()
    plt.subplot(121); plt.imshow(im)
    plt.subplot(122); plt.imshow(im_padded)
    plt.show()


if __name__ == '__main__':
    # DATASET_PATH = Path(r'C:\Users\FranMoreno\datasets\donut_hr')
    # sizes_dict = get_images_sizes_in_dataset(DATASET_PATH)
    # plot_sizes(sizes_dict)

    im_path = Path(r'C:\Users\FranMoreno\datasets\donut_hr\000436\000436_000.png')
    im = Image.open(im_path)
    # normalize_image(im, [1, 1])
    resize_and_pad_image(im, [1700, 2200])