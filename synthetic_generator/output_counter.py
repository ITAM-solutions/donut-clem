"""
File name: output_counter
Author: Fran Moreno
Last Updated: 10/14/2025
Version: 1.0
Description: TOFILL
"""
from pathlib import Path


if __name__ == '__main__':
    output_dir = Path('output')
    count = len(list(output_dir.glob('**/*.png')))
    print(f"There are {count} synthetic samples in the output folder.")