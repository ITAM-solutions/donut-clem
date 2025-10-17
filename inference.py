"""
File name: inference
Author: Fran Moreno
Last Updated: 10/17/2025
Version: 1.0
Description: TOFILL
"""
from donut import DonutModel, JSONParseEvaluator, load_json, save_json
import numpy as np
import os, shutil
from datetime import datetime
from datasets import load_dataset
import json
import torch
import pymupdf
from pathlib import Path
import os
from PIL import Image
import time


def prepare_real_document(document_path: Path, tmp_folder: Path = None):
    doc_obj = pymupdf.open(document_path)
    images = [page.get_pixmap(dpi=200) for page in doc_obj]

    if not tmp_folder:
        tmp_folder = document_path.parent / 'tmp'

    if not tmp_folder.exists():
        os.makedirs(tmp_folder)

    ims = []
    for idx, image in enumerate(images):
        im_tmp_path =  tmp_folder / f'{document_path.stem}_p{str(idx).zfill(6)}.png'
        image.save(im_tmp_path)
        ims.append(Image.open(im_tmp_path))

    return ims


def clean_tmp_folder(tmp_folder: Path):
    shutil.rmtree(tmp_folder)


def compute_score(gt: dict, pr: dict) -> float:
    """
    gt: ground-truth
    pr: prediction
    """
    for k, v in pr.items():
        if v == 'None':
            pr[k] = None

    if pr.get("products"):
        for pk, pv in pr['products'].items():
            if type(pv) != list:
                pr['products'][pk] = [pv]

    evaluator = JSONParseEvaluator()
    score = evaluator.cal_acc(pr, gt)
    return round(score * 100, 2)


def inference(model_path, data_path, document: Path, output_verbose: bool):
    task_name='dataset'

    model = DonutModel.from_pretrained(model_path)
    # model.half()
    model.eval()

    dataset = load_dataset(data_path, split="test")
    tmp_folder = document.parent / 'tmp'
    ims = prepare_real_document(document, tmp_folder=tmp_folder)

    for idx, sample in enumerate(dataset):
        ts = time.time()
    # for im in ims:
        ground_truth = json.loads(sample["ground_truth"])
        image_path = sample['image'].filename
        output = model.inference(image=sample["image"], prompt=f"<s_{task_name}>")["predictions"][0]
        # output = model.inference(image=im, prompt=f"<s_{task_name}>")["predictions"][0]
        tf = time.time() - ts
        gt = ground_truth["gt_parse"]
        if output_verbose: print("-----------------")

        # print("Image:", im.filename)
        print("Image:", image_path)

        if output_verbose:
            print("\nGround-truth:")
            print(json.dumps(gt, indent=2))
            print("\nPrediction:")
            print(json.dumps(output, indent=4))

        score = compute_score(gt, output)
        print(f"\nACCURACY: {score}%")
        print(f"TIME: {round(tf, 2)}s")

    # Clear temporary images
    # clean_tmp_folder(tmp_folder)


if __name__ == '__main__':
    model_path = r"weights/20251016_090333"
    data_path = r"C:\Users\FranMoreno\datasets\train_dataset_16Oct\real_data"

    document = Path(r"dataset/real_documents/(klaas) Chef - order doc - 2pages.pdf")

    inference(model_path, data_path, document, output_verbose=True)

