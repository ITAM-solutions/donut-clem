"""
File name: optimal_output_limit
Author: Fran Moreno
Last Updated: 11/13/2025
Version: 1.0
Description: Use this script to compute some metrics that will help to determine the optimal
Donut output limit based on your dataset.
This is based on ground-truth JSON files from your dataset. The process goes like follows:
1. Collect all GT JSON files.
2. Load your Donut model. Please note that it is required that it was trained at least once,
as it will include the special tokens related to your use case.
2. For each, ground-truth file, read its content, and pass through json2tokens() model method
to compute the total number of tokens it generates.
3. Compute metrics to determine the optimal output limit.

Please note that this only makes sense if your dataset is a good representation of the
real cases that Donut will handle in a Production environment.
"""
import statistics
import json
import numpy as np

from pathlib import Path

from clem.model import DonutCLEM

MODEL_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\weights\20251016_090333")
DATASET_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\samples")


def compute_tokens_from_dataset(dataset_path: Path) -> dict:
    model = DonutCLEM(MODEL_PATH)
    model_output_limit = model.model.config.max_length

    gt_files = dataset_path.glob("*/*.json")

    collected_tokens = list()
    limit_exceeded_booleans = list()
    limit_exceeded_set = list()
    for gt_file in gt_files:
        with open(gt_file, 'r') as fp:
            content = json.load(fp)
            limit_exceeded, token_count = model.exceeds_output_limit(content)
            collected_tokens.append(token_count)
            limit_exceeded_booleans.append(limit_exceeded)
            if limit_exceeded:
                limit_exceeded_set.append(token_count)

    return {
        "tokens": collected_tokens,
        "exceeded_booleans": limit_exceeded_booleans,
        "exceeded_cases": limit_exceeded_set,
        "model_output_limit": model_output_limit
    }


def compute_metrics_from_tokens(tokens_info: dict) -> dict:
    tokens_list = tokens_info['tokens']
    avg_tokens_number = statistics.mean(tokens_list)
    variance_tokens = statistics.variance(tokens_list)
    max_tokens = max(tokens_list)
    min_tokens = min(tokens_list)

    exceeded_booleans = tokens_info['exceeded_booleans']
    exceeded_count = len([i for i in exceeded_booleans if i])
    exceeded_ratio = 100 * exceeded_count / len(exceeded_booleans)
    exceeded_ratio_str = f"{exceeded_count}/{len(exceeded_booleans)} ({exceeded_ratio:.2f})"


    exceeded_cases = tokens_info["exceeded_cases"]
    exceeded_cases_avg = int(statistics.mean(exceeded_cases))
    exceeded_cases_median = int(statistics.median(exceeded_cases))
    exceeded_cases_percentile_75 = int(np.percentile(exceeded_cases, 75))

    # Find out optimal output limit using computed metrics
    new_proposed_limits = [
        ("exceeded samples average", exceeded_cases_avg),
        ("exceeded samples median", exceeded_cases_median),
        ("exceeded samples p75", exceeded_cases_percentile_75),
    ]

    proposals = dict()
    for name, new_proposed_limit in new_proposed_limits:
        new_exceeded_booleans = [num_tokens > new_proposed_limit for num_tokens in tokens_list]
        new_exceeded_count = len([i for i in new_exceeded_booleans if i])
        new_exceeded_ratio = 100 * new_exceeded_count / len(new_exceeded_booleans)
        new_exceeded_ratio_str = f"{new_exceeded_count}/{len(new_exceeded_booleans)} ({new_exceeded_ratio:.2f})"
        proposals.update({
            f"PROPOSAL - {name}: {new_proposed_limit} - EXCEEDED RATIO": new_exceeded_ratio_str
        })

    return {
        "AVERAGE": avg_tokens_number,
        "VARIANCE": variance_tokens,
        "MAX TOKENS": max_tokens,
        "MIN TOKENS": min_tokens,
        "LIMIT EXCEEDED RATIO": exceeded_ratio_str,
        "NUM SAMPLES": len(tokens_list),
        "ACTUAL MODEL LIMIT": tokens_info["model_output_limit"],
        "EXCEEDED CASES": tokens_info["exceeded_cases"],
        **proposals
    }


if __name__ == '__main__':
    tokens = compute_tokens_from_dataset(DATASET_PATH)
    metrics = compute_metrics_from_tokens(tokens)

    for k, v in metrics.items():
        print(f"{k}:\t{v}")

