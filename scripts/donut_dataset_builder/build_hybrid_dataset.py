import os
import shutil
from pathlib import Path
from shutil import copytree
import json


def count_dataset_samples(dataset_path: Path) -> tuple[int, int, int]:
    train_count = len(list(dataset_path.glob('train/*.png')))
    val_count = len(list(dataset_path.glob('validation/*.png')))
    test_count = len(list(dataset_path.glob('test/*.png')))
    return train_count, val_count, test_count


def get_subset(dataset_path: Path, split: str, num_samples: int = None) -> tuple[list[Path], list[str]]:
    all_imgs = list(dataset_path.glob(f'{split}/*.png'))
    json_path = dataset_path / split / 'metadata.jsonl'
    imgs = []
    json_lines = []
    if not num_samples:
        num_samples = len(all_imgs)

    with open(json_path, 'r') as fp:
        for i in range(num_samples):
            json_line = fp.readline()
            line_obj = json.loads(json_line)
            for s in all_imgs:
                if line_obj['file_name'] in str(s):
                    imgs.append(s)
                    json_lines.append(json_line)
                    break

    return imgs, json_lines


def build_hybrid_dataset(
        dataset_real_path: Path,
        dataset_synth_path: Path,
        real_data_ratio: float = None
) -> tuple:
    train_count, val_count, test_count = count_dataset_samples(dataset_real_path)

    train_count = int(train_count * real_data_ratio) if real_data_ratio else None
    val_count = int(val_count * real_data_ratio) if real_data_ratio else None
    test_count = int(test_count * real_data_ratio) if real_data_ratio else None

    train_set = get_subset(dataset_synth_path, split="train", num_samples=train_count)
    val_set = get_subset(dataset_synth_path, split="validation", num_samples=val_count)
    test_set = get_subset(dataset_synth_path, split="test", num_samples=test_count)

    return train_set, val_set, test_set


def save_to_hybrid_path(org_path, output_path: Path, data, split) -> None:
    copytree(org_path / split, output_path / split)

    last_sample_prefix = int(list((org_path / split).glob('*.png'))[-1].stem.split('_')[0])

    # Update images names and their reference in the corresponding jsonl line for synth data.
    new_json_lines = []
    for img, json_line in zip(data[0], data[1]):
        last_sample_prefix += 1

        current_name = img.stem
        new_name = str(last_sample_prefix).zfill(6) + '_000'
        new_im_path = output_path / split / (new_name + '.png')

        # We can save the image already
        shutil.copy(img, new_im_path)

        json_line_mod = json_line.replace(current_name, new_name)

        new_json_lines.append(json_line_mod)

    # Finally save new json lines
    with open(output_path / split / 'metadata.jsonl', 'a') as fp:
        fp.writelines(new_json_lines)





if __name__ == '__main__':
    DATASET_REAL = Path(r"C:\Users\FranMoreno\datasets\train_dataset_18Nov\real_data")
    DATASET_SYNTH = Path(r"C:\Users\FranMoreno\datasets\train_dataset_18Nov\synth_data")
    DATASET_HYBRID = Path(r"C:\Users\FranMoreno\datasets\train_dataset_18Nov\combined_data")

    tr, val, te = build_hybrid_dataset(DATASET_REAL, DATASET_SYNTH, real_data_ratio=None)

    if not DATASET_HYBRID.exists():
        os.makedirs(DATASET_HYBRID)

    save_to_hybrid_path(DATASET_REAL, DATASET_HYBRID, tr, "train")
    save_to_hybrid_path(DATASET_REAL, DATASET_HYBRID, val, "validation")
    save_to_hybrid_path(DATASET_REAL, DATASET_HYBRID, te, "test")

