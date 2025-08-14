"""
File name: cost_analysis
Author: Fran Moreno
Last Updated: 8/14/2025
Version: 1.0
Description: TOFILL
"""
import csv
import yaml
from pathlib import Path
import matplotlib.pyplot as plt


def fetch_from_config_file(filepath: Path) -> dict:
    with open(filepath, 'r') as f:
        config_all = yaml.safe_load(f)

    return {
        "train_batch_size": config_all['train_batch_sizes'][0],
        "validation_batch_size": config_all['val_batch_sizes'][0],
        "num_workers": config_all['num_workers'],
        "validation_steps_per_epoch": 1. / config_all['check_val_every_n_epoch'],
        "learning_rate": config_all['lr'],
        "total_epochs": config_all['max_epochs']
    }


def get_input(msg: str, numeric: bool = False):
    not_valid_input = True

    while not_valid_input:
        ans = input(f"{msg}: ")
        if numeric:
            try:
                return float(ans)
            except ValueError:
                print("Please write a numeric quantity.")
                continue
        return ans


def fetch_data_from_user() -> dict:
    gpu = get_input("GPU name")
    vram = get_input("Total VRAM available in GPU")
    pph = get_input("Price per hour ($)", numeric=True)
    return {
        "GPU": gpu,
        "VRAM": vram,
        "price_per_hour": pph
    }


def fetch_time_values_from_user() -> dict:
    time_per_epoch_min = get_input("Time per epoch (min)", numeric=True)
    time_per_val_step_min = get_input("Time per validation step (min)", numeric=True)
    return {
        "epoch_time_s": time_per_epoch_min / 60.,
        "val_step_time_s": time_per_val_step_min / 60.
    }


def compute_analysis(config_data: dict, device_data: dict, time_data: dict):
    pph = device_data['price_per_hour']
    cost_per_epoch = time_data['epoch_time_s'] * pph
    cost_per_val_step = time_data['val_step_time_s'] * pph

    total_epochs_cost = cost_per_epoch * config_data['total_epochs']
    total_val_steps_cost = cost_per_val_step * config_data['total_epochs'] * config_data['validation_steps_per_epoch']

    total_cost = total_epochs_cost + total_val_steps_cost

    return {
        "cost_per_epoch": cost_per_epoch,
        "cost_per_val_step": cost_per_val_step,
        "total_epochs_cost": total_epochs_cost,
        "total_val_steps_cost": total_val_steps_cost,
        "total_cost": total_cost
    }


def save_analysis_and_regression(config_data: dict, device_data: dict, time_data: dict , analysis_data: dict) -> None:
    txt_file_content = f"""
    Device: {device_data['GPU']} ({device_data['VRAM']} VRAM) - {device_data['price_per_hour']} $/h
    
    - Number of epochs: {config_data['total_epochs']}
    - Learning rate: {config_data['learning_rate']}
    - Number of GPU workers: {config_data['num_workers']}
    - Training batch size: {config_data['train_batch_size']}
    - Validation batch size: {config_data['validation_batch_size']}
    
    ------------
    
    - Time per epoch: {time_data['epoch_time_s']:.4f} s
    - Time per validation step: {time_data['val_step_time_s']:.4f} s
    
    ------------
    
    TOTAL COST: {analysis_data['total_cost']}
    """

    with open('results.txt', 'w') as f:
        f.write(txt_file_content)


def compute_costs(config_file: Path):
    config_data = fetch_from_config_file(config_file)
    device_data = fetch_data_from_user()
    time_data = fetch_time_values_from_user()

    analysis_data = compute_analysis(config_data, device_data, time_data)
    save_analysis_and_regression(config_data, device_data, time_data, analysis_data)


if __name__ == '__main__':
    config_file_path = Path("config/train_clem.yaml")
    compute_costs(config_file=Path(config_file_path))

