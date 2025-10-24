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
from dataclasses import dataclass, fields

import openpyxl

from donut import DonutModel, JSONParseEvaluator
import pandas as pd
from PIL import Image
from typing import List
import json
import numpy as np
from datetime import datetime

import evaluation.prediction_cleaning as pr_cleaning


@dataclass
class Metrics:
    tree_acc: float
    product_hit: float
    matches: float


class Evaluator:
    def __init__(self, evaluation_dataset: Path):
        self.evaluation_dataset = evaluation_dataset

        self.evaluation_subsets = [self.evaluation_dataset / i for i in os.listdir(self.evaluation_dataset)]

        self.json_parse_evaluator = JSONParseEvaluator()

        self.samples_with_broken_outputs: List[Path] = []  # To keep track of samples that produce a broken inference output.
        self.low_accuracy_samples: List[Path] = []  # To keep track of samples that had a low prediction accuracy for later investigation

    def run(self, model_path: Path, clean_previous_results: bool = False):
        """
        Evaluates the given model, generating metrics about its accuracy and results' quality.

        :param model_path: Path to the model path ('results' folder obtained from training).
        :param clean_previous_results: if `True`, cleans previously computed evaluation results, if existing.
        :return: None (generates some files in the `model_path` folder).
        """
        if clean_previous_results:
            ans = input(f"'clean_previous_results' is set to True, so previous logs will be permanently removed. Continue? (Y/n): ")
            if ans.lower() != 'y':
                print("Ending program.")
            else:
                print('Starting model evaluation...')

        model = self._load_donut(model_path)
        output_path = model_path / 'evaluation'
        logs_path = output_path / 'logs'

        results_df = pd.DataFrame(columns=[
            "subset", "tree_acc", "products_hit", "matches_count"
        ])

        self._prepare_results_folder(output_path, clean_previous_results)

        # Loop through subsets in evaluation data
        for subset_dir in self.evaluation_subsets:
            print(f'RUNNING ON SUBSET: {subset_dir.name}')
            sample_ims, sample_gts = self._load_samples(subset_dir)

            tree_accs, products_hits, matches_counts = [], [], []

            # Loop through each sample to compute their individual metrics
            for sample_im_path, sample_gt_path in zip(sample_ims, sample_gts):
                print(f'Sample: {subset_dir.name}/{sample_im_path.stem}')

                with open(sample_gt_path, 'r', encoding='utf-8') as fp:
                    sample_gt = json.load(fp)

                sample_gt = self._sanitize_output(sample_gt, sample_im_path)

                pred = self._compute_inference(model, sample_im_path)

                metrics = self._compute_evaluation_metrics(sample_gt, pred)
                if self._check_if_low_accuracy(metrics):
                    self.low_accuracy_samples.append(sample_im_path)

                tree_accs.append(metrics.tree_acc)
                products_hits.append(metrics.product_hit)
                matches_counts.append(metrics.matches)

                self._save_evaluation_log(sample_im_path, logs_path, sample_gt, pred, metrics)
                print("Evaluation log saved.")

            subset_results = pd.DataFrame(
                [[subset_dir.name, np.mean(tree_accs), np.mean(products_hits), np.mean(matches_counts)]],
                columns = ["subset", "tree_acc", "products_hit", "matches_count"]
            )
            results_df = pd.concat([results_df, subset_results], ignore_index=True)

        results_df.to_csv(output_path / 'metrics.csv', index=False)

        # Save a list of all the samples that generated a broken output, for later investigation
        with open(output_path / 'broken_samples.txt', 'w', encoding='utf-8') as fp:
            for sample in self.samples_with_broken_outputs:
                fp.write(f'{sample.parent}/{sample.name}\n')

        # Save samples that produced bad accuracy results for later investigation and improvement
        low_accuracy_samples_path = output_path / 'low_accuracy_samples'
        low_accuracy_samples_path.mkdir(parents=True, exist_ok=True)
        for sample in self.low_accuracy_samples:
            shutil.copy(sample, low_accuracy_samples_path / sample.name)

    def _compute_evaluation_metrics(self, gt: dict, pr: dict) -> Metrics:
        return Metrics(
            tree_acc=self._compute_tree_acc(gt, pr),
            product_hit=self._compute_products_hit_acc(gt, pr),
            matches=self._compute_matches_count(gt, pr)
        )

    @staticmethod
    def _check_if_low_accuracy(metrics: Metrics) -> bool:
        return \
            metrics.tree_acc <= 0.7 or \
            metrics.product_hit <= 0.5 or \
            metrics.matches <= 0.3



    def _compute_tree_acc(self, gt: dict, pr: dict) -> float:
        """
        Score based on the JSON structure. Measures how good the model is in reproducing
        the expected data structure.

        If this score is low, then the "matches_count" score is not relevant.

        :param gt: ground-truth
        :param pr: prediction
        :return: tree accuracy score
        """
        gt = self._prepare_none_values_for_tree_acc(gt)
        pr = self._prepare_none_values_for_tree_acc(pr)
        score = self.json_parse_evaluator.cal_acc(pr, gt)
        return score

    @staticmethod
    def _prepare_none_values_for_tree_acc(data: dict) -> dict:
        data_prepared = {k: 'None' if v is None else v for k, v in data.items()}
        data_prepared['products'] = {
            k: ['None' if v is None else v for v in values]
            for k, values in data_prepared.get('products', {}).items()
        }
        return data_prepared


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
        return score

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

        gt_products = gt.get('products', {})
        pr_products = pr.get('products', {})

        # Product fields
        for field in gt_products:
            if field not in pr_products:
                total_values_count += len(gt_products[field])
            else:
                for value in gt_products[field]:
                    if value in pr_products[field]:
                        total_matches_count += 1
                    total_values_count += 1

        matches_ratio = total_matches_count / total_values_count
        return matches_ratio

    def _sanitize_output(self, data: dict, im_path: Path) -> dict:
        try:
            data = pr_cleaning.normalize_products_structure2(data)
            data = pr_cleaning.remove_unused_fields(data)
            data = pr_cleaning.normalize_empty_values(data)
            return data
        except AttributeError:
            print(f'Warning: sample {im_path} produced a broken output. Interpreting as empty response.')
            self.samples_with_broken_outputs.append(im_path)
            return pr_cleaning.get_empty_response()
        except Exception:
            print(f'Warning: sample {im_path} produced an unexpected error. Interpreting as empty response.')
            self.samples_with_broken_outputs.append(im_path)
            return pr_cleaning.get_empty_response()

    def update_excel_report(self, all_results_path: Path, excel_path: Path):
        """

        :return:
        """
        def as_date(folder_name: str) -> datetime:
            date_str = folder_name.split('_')[0]
            return datetime.strptime(date_str, "%Y%m%d")

        subsets = os.listdir(self.evaluation_dataset)
        columns = ['Date', 'Tree Accuracy', 'Product Count Accuracy', 'Exact Matches Score', 'Num Samples']

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
                        record_df[record_df['subset'] == subset]['num_samples'].iloc[0],
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
                        df_csv[df_csv['subset'] == subset]['num_samples'].iloc[0],
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

    def _compute_inference(self, model, im_path: Path):
        im = Image.open(im_path)
        task_name = "dataset"
        output = model.inference(image=im, prompt=f"<s_{task_name}>")["predictions"][0]
        output = self._sanitize_output(output, im_path)
        return output

    def _save_evaluation_log(self, im_path: Path, output_path: Path, gt, pred, metrics: Metrics):
        output_path.mkdir(parents=True, exist_ok=True)

        sample_name = im_path.stem

        metrics_pairs = [(metric.name, getattr(metrics, metric.name)) for metric in fields(metrics)]
        metrics_str = '\n'.join([f'{k}:\t{v}' for k, v in metrics_pairs])
        comparison_str = self._parse_json_comparison(gt, pred)

        log_str = f"METRICS:\n{metrics_str}\n\n===========\n\nCOMPARISONS (Expected | Actual):\n\n{comparison_str}"

        with open(output_path / f"{sample_name}.txt", 'w', encoding='utf-8') as fp:
            fp.writelines(log_str)

    def _parse_json_comparison(self, gt: dict, pred: dict):

        text_slices = []
        for k_gt, val_gt in gt.items():
            if not isinstance(val_gt, dict):
                val_pr = pred.get(k_gt)
                text_slice = f"\n{k_gt}:\n\tExpected: {val_gt}\n\tActual:   {val_pr}"
                text_slices.append(text_slice)

        # Then, products
        products_gt = gt.get('products', {})
        products_pr = pred.get('products', {})
        products_gt, products_pr = self._get_product_comparison(products_gt, products_pr)

        for field in products_gt:
            text_slice = f"\n{field}:\n\tExpected: {products_gt[field]}\n\tActual:   {products_pr[field]}"
            text_slices.append(text_slice)

        return '\n'.join(text_slices)

    @staticmethod
    def _get_product_comparison(gt: dict, pr: dict):
        print('Comparing products:')
        print(f'gt:\n{gt}')
        print(f'pr:\n{pr}')

        for field in gt:
            if field not in pr:
                pr[field] = [None] * len(gt[field])
            else:
                len_diff = len(gt[field]) - len(pr[field])
                if len_diff > 0:  # len(gt) > len(pr)
                    pr[field].extend([None] * len_diff)
                elif len_diff < 0:  # len(gt) < len(pr)
                    gt[field].extend([None] * len_diff)
                # else: same length, everything fine

        return gt, pr


if __name__ == '__main__':
    EVALUATION_DATASET = Path('dataset/evaluation/samples')
    MODEL_PATH = Path('weights/20251016_090333')
    EXCEL_PATH = Path(r'C:\Users\FranMoreno\ITAM solutions\Innovations - Development team - Contract analysis automation\data\donut_data_evaluation\results\evaluation_results_powerbi - Copy.xlsx')

    evaluator = Evaluator(EVALUATION_DATASET)

    # Run evaluation on a certain model
    # evaluator.run(MODEL_PATH, clean_previous_results=True)

    # Merge results and output to excel file.
    evaluator.update_excel_report(Path('weights'), EXCEL_PATH)
