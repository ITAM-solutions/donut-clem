"""
File name: evaluation
Author: Fran Moreno
Last Updated: 10/21/2025
Version: 1.0
Description: Run this file when a new set of weights has been obtained from training to evaluate its accuracy.
Use always over the same evaluation data.
The evaluation data is expected to contain subsets (for example: normal cases, special cases, empty cases, etc). The
evaluation will be computed first for each subset, and then the results will be merged and a merged resport will be
generated.
"""
import os
import shutil
from pathlib import Path

import openpyxl

from donut import DonutModel, JSONParseEvaluator
import pandas as pd
from PIL import Image
from typing import List, Optional
import json
import numpy as np
from datetime import datetime


class Evaluator:
    def __init__(self, evaluation_dataset: Path):
        self.evaluation_dataset = evaluation_dataset

        self.evaluation_subsets = [self.evaluation_dataset / i for i in os.listdir(self.evaluation_dataset)]

        self.json_parse_evaluator = JSONParseEvaluator()


    def run(self, model_path: Path, clean_previous_results: bool = False):
        """
        Evaluates the given model, generating metrics about its accuracy and results' quality.

        :param model_path: Path to the model path ('results' folder obtained from training).
        :param clean_previous_results: if `True`, cleans previously computed evaluation results, if existing.
        :return: None (generates some files in the `model_path` folder).
        """
        model = self._load_donut(model_path)
        output_path = model_path / 'evaluation'
        logs_path = output_path / 'logs'

        results_df = pd.DataFrame(columns=[
            "subset", "tree_acc", "products_hit", "matches_count"
        ])

        self._prepare_results_folder(output_path, clean_previous_results)

        # Loop through subsets in evaluation data
        for subset_dir in self.evaluation_subsets:

            sample_ims, sample_gts = self._load_samples(subset_dir)

            tree_accs, products_hits, matches_counts = [], [], []

            # Loop through each sample to compute their individual metrics
            for sample_im_path, sample_gt_path in zip(sample_ims, sample_gts):

                sample_im = Image.open(sample_im_path)
                with open(sample_gt_path, 'r', encoding='utf-8') as fp:
                    sample_gt = json.load(fp)

                pred = self._compute_inference(model, sample_im)

                tree_acc =  self._compute_tree_acc(sample_gt, pred)
                products_hit = self._compute_products_hit_acc(sample_gt, pred)
                matches_count = self._compute_matches_count(pred, sample_gt)
                metrics = {
                    "tree_acc": tree_acc,
                    "products_hit": products_hit,
                    "matches_count": matches_count
                }

                tree_accs.append(tree_acc)
                products_hits.append(products_hit)
                matches_counts.append(matches_count)

                self._save_evaluation_log(sample_im_path, logs_path, sample_gt, pred, metrics)

            subset_results = pd.DataFrame(
                [[subset_dir.name, np.mean(tree_accs), np.mean(products_hits), np.mean(matches_counts)]],
                columns = ["subset", "tree_acc", "products_hit", "matches_count"]
            )
            results_df = pd.concat([results_df, subset_results], ignore_index=True)

        overall_results = pd.DataFrame([[
            "OVERALL",
            results_df["tree_acc"].mean(),
            results_df["products_hit"].mean(),
            results_df["matches_count"].mean(),
        ]], columns=["subset", "tree_acc", "products_hit", "matches_count"])

        results_df = pd.concat([results_df, overall_results], ignore_index=True)
        results_df.to_csv(output_path / 'metrics.csv', index=False)

    def _compute_tree_acc(self, gt: dict, pr: dict) -> float:
        """
        Score based on the JSON structure. Measures how good the model is in reproducing
        the expected data structure.

        If this score is low, then the "matches_count" score is not relevant.

        :param gt: ground-truth
        :param pr: prediction
        :return: tree accuracy score
        """
        score = self.json_parse_evaluator.cal_acc(pr, gt)
        return round(score * 100, 4)

    @staticmethod
    def _compute_products_hit_acc(gt: dict, pr: dict) -> float:
        """
        Follows this formula:

                       |len(pr.products) - len(gt.products)|
        CountAcc = 1 - -------------------------------------
                            max(1, len(gt.products))

        Which results in a normalized value (0 to 1), where 1 is the perfect result.

        Measures if the model can find all the products in the document.

        :param gt: ground-truth
        :param pr: prediction
        :return: resulting metric
        """
        gt_products = gt.get('products')
        pr_products = pr.get('products')
        num_products_gt = len(gt_products) if gt_products else 0
        num_products_pr = len(pr_products) if pr_products else 0

        length_diff = abs(num_products_pr - num_products_gt)
        max_length = max(1, num_products_gt)

        score = 1 - length_diff / max_length
        return round(score * 100, 4)

    @staticmethod
    def _compute_matches_count(gt: dict, pr: dict) -> float:
        """
        Retrieves the relation between how many values were exact matches between gt and pr,
        and the total number of values that should have been extracted.

        Measures if the model can correctly find values in the document.
        This metric strongly depends on the 'tree_acc' metric to obtain good results.

        :param gt:
        :param pr:
        :return:
        """
        total_values_count = 0
        total_matches_count = 0
        # common fields
        for key in gt:
            if key == 'products':
                continue

            if gt.get(key) == pr.get(key):
                total_matches_count += 1

            total_values_count += 1

        gt_products = gt.get('products', [])
        pr_products = pr.get('products', [])
        # Product fields
        for idx, product in enumerate(gt_products):
            if len(pr_products) < idx + 1:
                total_values_count += len(list(product.keys()))
                continue

            for product_key in product:
                if product[product_key] == pr_products[idx][product_key]:
                    total_matches_count += 1
                total_values_count += 1

        matches_ratio = total_matches_count / total_values_count
        return round(matches_ratio * 100, 4)

    def update_excel_report(self, all_results_path: Path, excel_path: Path):
        """
        # 1. Check if the given `excel_path` already exists. If so, we just need to update with the new data.
        # 2. For that, read the content of the excel (one sheet is enough), to get all the dates that are already recorded.
        # 3. Read the list of weight folders, and discard those that are included.
        # 4. Open the excel in append mode, and add those new elements.
        # 5. If it did not exist, save all the data in there.

        :return:
        """
        def as_date(folder_name: str) -> datetime:
            date_str = folder_name.split('_')[0]
            return datetime.strptime(date_str, "%Y%m%d")

        subsets = os.listdir(self.evaluation_dataset)
        # results = [as_date(folder_name) for folder_name in os.listdir(all_results_path)]
        columns = ['Date', 'Tree Accuracy', 'Product Count Accuracy', 'Exact Matches Score']

        data = {
            as_date(folder_name): pd.read_csv(csv_file) for folder_name in os.listdir(all_results_path)
            if (csv_file := all_results_path / folder_name / 'evaluation' / 'metrics.csv').exists()
        }

        if excel_path.exists():
            existing_sheet = pd.read_excel(excel_path, sheet_name='Blank')
            records = list(d.to_pydatetime() for d in existing_sheet['Date'])
            missing_records = list(set(data.keys()) - set(records))

            output_dfs = dict()
            for subset in subsets:
                combined_data = []
                for date_obj in missing_records:
                    record_df = data[date_obj]
                    combined_data.append([
                        date_obj.date(),
                        record_df[record_df['subset'] == subset]['tree_acc'].iloc[0],
                        record_df[record_df['subset'] == subset]['products_hit'].iloc[0],
                        record_df[record_df['subset'] == subset]['matches_count'].iloc[0],
                    ])

                combined_df = pd.DataFrame(combined_data, columns=columns)
                output_dfs[subset] = combined_df

            with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists="overlay") as writer:
                for subset, df in output_dfs.items():
                    df.to_excel(writer, sheet_name=subset, index=False, header=False, startrow=writer.sheets[subset].max_row)

        else:
            output_dfs = dict()
            for subset in subsets:
                combined_data = []
                for date_obj, df_csv in data.items():
                    combined_data.append([
                        date_obj.date(),
                        df_csv[df_csv['subset'] == subset]['tree_acc'].iloc[0],
                        df_csv[df_csv['subset'] == subset]['products_hit'].iloc[0],
                        df_csv[df_csv['subset'] == subset]['matches_count'].iloc[0],
                    ])

                combined_df = pd.DataFrame(combined_data, columns=columns)
                output_dfs[subset] = combined_df

            wb = openpyxl.Workbook()
            wb.save(excel_path)
            with pd.ExcelWriter(excel_path) as writer:
                for subset, df in output_dfs.items():
                    df.to_excel(writer, sheet_name=subset, index=False)

    @staticmethod
    def _prepare_results_folder(output_path: Path, clean_previous_results: bool):
        if not output_path.exists():
            os.makedirs(output_path)

        if output_path.exists() and os.listdir(output_path):
            if clean_previous_results:
                shutil.rmtree(output_path)
                os.makedirs(output_path)
            else:
                raise Exception("This model has already been evaluated. "
                                "If you want to remove them, set 'clean_previous_results = true'.")

    @staticmethod
    def _load_donut(model_path: Path):
        return DonutModel.from_pretrained(model_path)
        # return DonutModel.from_pretrained(os.getenv("HF_REPO_NAME"), use_auth_token=os.getenv("HF_TOKEN"))

    @staticmethod
    def _load_samples(subset_dir: Path) -> tuple:
        images: List[Path] = sorted(list(subset_dir.glob('*.png')))
        json_files = []

        for image in images:
            json_file = image.with_suffix('.json')

            if not json_file.exists():
                raise Exception(f"There are an invalid sample in your evaluation dataset: {image.name}")

            json_files.append(json_file)

        return images, json_files

    def _compute_inference(self, model, im):
        task_name = "dataset"
        output = model.inference(image=im, prompt=f"<s_{task_name}>")["predictions"][0]
        output = self._sanitize_output(output)
        return output

    @staticmethod
    def _sanitize_output(output: dict) -> dict:
        # Convert products to list in case it is a dictionary, to be aligned with ground-truth
        products = output.get('products')
        if products and type(products) == dict:

            # Find maximum number of products found in output
            max_num_products = 1
            for value in products.values():
                if type(value) == list and len(value) > max_num_products:
                    max_num_products = len(value)

            # Extend missing values to keep output normalized
            product_list = [dict() for _ in range(max_num_products)]
            for key, org_value in products.items():
                if type(org_value) == str:
                    values = [org_value] + [None] * (max_num_products - 1)
                else:
                    values = org_value + [None] * (max_num_products - len(org_value))

                # Update the output with the new values
                for idx, value in enumerate(values):
                    product_list[idx][key] = value

            output['products'] = product_list

        # Remove unused fields
        unused_shared_keys = ['ltype', 'ctype', 'atype', 'pub']
        unused_product_keys = ['issub']

        for k in unused_shared_keys:
            if k in output:
                del output[k]

        products = output.get('products', [])
        for k in unused_product_keys:
            for product in products:
                if k in product:
                    del product[k]

        # Rename fields that were named differently
        for product in products:
            if 'dfrom' in product:
                product['drom'] = product['dfrom']
                del product['dfrom']

        # Normalize empty values: 'None' to None
        for k, v in output.items():
            if v == 'None':
                output[k] = None

        pr_products = output.get('products')
        if pr_products:
            output['products'] = [
                {key: None if value == 'None' else value for key, value in product.items()}
                for product in pr_products
            ]

        return output

    def _save_evaluation_log(self, im_path: Path, output_path: Path, gt, pred, metrics: dict):
        if not output_path.exists():
            os.makedirs(output_path)

        sample_name = im_path.stem

        metrics_str = '\n'.join([f'{k}:\t{v}' for k, v in metrics.items()])
        comparison_str = self._parse_json_comparison(gt, pred)

        log_str = f"METRICS:\n{metrics_str}\n\n===========\n\nCOMPARISONS (Expected | Actual):\n\n{comparison_str}"

        with open(output_path / f"{sample_name}.txt", 'w') as fp:
            fp.writelines(log_str)

    def _parse_json_comparison(self, gt: dict, pred: dict):

        text_slices = []
        for k_gt, val_gt in gt.items():
            if type(val_gt) == str or val_gt is None:
                val_pr = pred.get(k_gt, "null")
                field_string = f'{k_gt:<8}:\t{val_gt} |\t{val_pr}'
                text_slices.append(field_string)

        text_slices.append('---------------------')

        # Then, products
        products_gt = gt.get('products')
        products_pr = pred.get('products')
        products_gt, products_pr = self._get_product_comparison(products_gt, products_pr)

        for idx, (product_gt, product_pr) in enumerate(zip(products_gt, products_pr)):
            product_text_slices = list()
            product_text_slices.append(f"\nproduct_{idx}:")
            for prod_k_gt, prod_v_gt in product_gt.items():
                prod_v_pr = product_pr.get(prod_k_gt, "null")
                pr_field_string = f'{prod_k_gt:<8}:\t{prod_v_gt} |\t{prod_v_pr}'
                product_text_slices.append(pr_field_string)

            # Join text to text_slices
            text_slices.extend(product_text_slices)

        return '\n'.join(text_slices)

    @staticmethod
    def _get_product_comparison(gt: Optional[List], pr: Optional[List]):
        # 1st step: make sure we operate with lists
        if gt is None: gt = []
        if pr is None: pr = []

        # 2nd step: make lists equal in length
        len_diff = len(gt) - len(pr)
        if len_diff > 0:  # len(gt) > len(pr)
            placeholder_content = {k: None for k in gt[0]} if len(gt) > 0 else dict()
            pr.extend([placeholder_content for _ in range(len_diff)])
        elif len_diff < 0:  # len(pr) > len(gt)
            placeholder_content = {k: None for k in pr[0]} if len(gt) > 0 else dict()
            gt.extend([placeholder_content for _ in range(abs(len_diff))])

        return gt, pr


if __name__ == '__main__':
    EVALUATION_DATASET = Path('dataset/evaluation/samples')
    MODEL_PATH = Path('weights/20251016_090333')
    EXCEL_PATH = Path('dataset/evaluation_excels/results.xlsx')

    evaluator = Evaluator(EVALUATION_DATASET)

    # Run evaluation on a certain model
    # evaluator.run(MODEL_PATH, clean_previous_results=True)

    # Merge results and output to excel file.
    evaluator.generate_evolution_report(Path('weights'), EXCEL_PATH)

    # gt = []
    # pr = None
    #
    # print(Evaluator._get_product_comparison(gt, pr))
