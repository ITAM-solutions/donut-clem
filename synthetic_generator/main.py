"""
File name: main
Author: Fran Moreno
Last Updated: 9/24/2025
Version: 1.0
Description: TOFILL
"""
import random

from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from pipeline import SyntheticDataGenerator
from typing import Dict, List


def split_equally(total: int, length: int) -> List[int]:
    base = total // length
    mod = total % length
    res = [base for _ in range(length)]

    for i in range(length):
        if not mod:
            break
        res[i] += 1
        mod -= 1
    return res


def get_template_groups(templates: List[Path]) -> Dict[str, List[Path]]:
    groups = defaultdict(list)
    for template in templates:
        base_code = template.stem.split('_')[1]
        groups[base_code].append(template)
    return groups


def produce_synthetic_dataset(
        templates_path: Path,
        output_path: Path,
        total_samples: int,
        pct_special_samples: float,
        model_output_limit: int,
):
    num_special_samples = int(total_samples * pct_special_samples)
    num_common_samples = total_samples - num_special_samples

    common_templates = list(templates_path.glob("common/*.docx"))
    special_templates = list(templates_path.glob("special/*.docx"))

    common_template_groups = get_template_groups(common_templates)
    special_template_groups = get_template_groups(special_templates)

    num_samples_per_common_group = split_equally(num_common_samples, len(common_template_groups))
    num_samples_per_special_group = split_equally(num_special_samples, len(special_template_groups))

    # Produce common samples
    total_samples = 0
    for template_group, num_samples in tqdm(zip(common_template_groups.values(), num_samples_per_common_group)):
        for idx in range(num_samples):
            template = random.choice(template_group)
            total_samples += 1
            print(total_samples, template)
            SyntheticDataGenerator(template, output_path, model_output_limit).generate()

    # Produce special samples
    for template_group, num_samples in tqdm(zip(special_template_groups.values(), num_samples_per_special_group)):
        for _ in range(num_samples):
            template = random.choice(template_group)
            total_samples += 1
            print(total_samples, template)
            SyntheticDataGenerator(template, output_path, model_output_limit).generate()


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

    """ PARAMETERS """
    TEMPLATES_PATH = Path("templates")
    OUTPUT_PATH = Path(f"output/{timestamp}")
    NUM_SAMPLES = 200
    PCT_SPECIAL_SAMPLES = 0.1
    MODEL_MAX_OUTPUT_LIMIT = 768  # Max. number of tokens to produce in the new trained model (default is 768)
    """            """

    produce_synthetic_dataset(TEMPLATES_PATH, OUTPUT_PATH, NUM_SAMPLES, PCT_SPECIAL_SAMPLES, MODEL_MAX_OUTPUT_LIMIT)

    # from desktop_alerts import show_desktop_alert
    # show_desktop_alert("Generation Finished", f"Generated {NUM_SAMPLES} samples")




