"""
File name: main
Author: Fran Moreno
Last Updated: 9/24/2025
Version: 1.0
Description: TOFILL
"""
import os

from pathlib import Path
from datetime import datetime
from collections import defaultdict

from pipeline import SyntheticDataGenerator, GenerationException
from typing import Dict, List


def end_alert():
    import winsound
    winsound.Beep(frequency=800, duration=400)
    winsound.Beep(frequency=1200, duration=400)
    winsound.Beep(frequency=1600, duration=400)
    winsound.Beep(frequency=2000, duration=400)
    winsound.Beep(frequency=2400, duration=400)


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
        samples_per_template: int,
        model_output_limit: int,
):
    templates = [templates_path / group_name / template_name
        for group_name in os.listdir(templates_path)
        for template_name in os.listdir(templates_path / group_name)
    ]

    # Produce common samples
    total_samples = 0
    for template in templates:
        for idx in range(samples_per_template):
            try:
                SyntheticDataGenerator(template, output_path, model_output_limit).generate()
                total_samples += 1
            except GenerationException:
                continue

    return total_samples


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

    """ PARAMETERS """
    TEMPLATES_PATH = Path("templates")
    OUTPUT_PATH_BASE = Path(f"output") / timestamp
    BATCHES = 5
    SAMPLES_PER_TEMPLATE = 1
    PCT_SPECIAL_SAMPLES = 0.1
    MODEL_MAX_OUTPUT_LIMIT = 1152  # Max. number of tokens to produce in the new trained model (default is 768)
    """            """

    # Initialize output folder
    OUTPUT_PATH_BASE.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    for batch_idx in range(BATCHES):
        try:
            OUTPUT_PATH = OUTPUT_PATH_BASE / str(batch_idx).zfill(2)
            batch_samples = produce_synthetic_dataset(TEMPLATES_PATH, OUTPUT_PATH, SAMPLES_PER_TEMPLATE, MODEL_MAX_OUTPUT_LIMIT)
            total_samples += batch_samples
        except Exception as e:
            print(e)
            print(f"Something went wrong with batch '{batch_idx}'. Skipping and starting next one (if remaining).")

    try:
        print(f"OUTPUT FOLDER: {OUTPUT_PATH_BASE}")
        print(f"ASKED SAMPLES: {total_samples}")
        print(f"TOTAL GENERATED (counting subsamples): {len(list(OUTPUT_PATH_BASE.glob('**/*.png')))}")
        print(f"GENERATED {total_samples} SAMPLES IN {OUTPUT_PATH_BASE}.")
    except Exception:  # noqa
        pass
    finally:
        end_alert()




