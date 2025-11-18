"""
File name: export_tagged_samples
Author: Fran Moreno
Last Updated: 6/27/2025
Version: 1.0
Description: TOFILL
"""
import zipfile

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

def zip_tagged_files(source_path: Path, dest_path: Path, zip_name: str = None) -> None:
    """
    Utility function to get all tagged samples from the full Donut dataset and zip them to be used for training.

    :param zip_name:
    :param source_path:
    :param dest_path:
    :return:
    """
    tagged_samples = [folder for folder in source_path.glob('*') if (folder / '.tagged').exists()]

    if not zip_name:
        zip_name = f'donut_{len(tagged_samples)}_samples_' + datetime.now().strftime('%d_%m_%Y-%H%M%S')

    with zipfile.ZipFile(dest_path / f'{zip_name}.zip', 'w') as zf:
        for folder in tqdm(tagged_samples):
            for file in folder.glob('*'):
                if file.name not in ['.tagged']:  # Files not needed for training
                    zf.write(file, arcname=Path('dataset') / file.relative_to(file.parent.parent))

if __name__ == '__main__':
    zip_tagged_files(
        source_path=Path(r'C:\Users\FranMoreno\ITAM solutions\Innovations - Development team - Contract analysis automation\backend\donut_dataset'),
        dest_path=Path(r'C:\Users\FranMoreno\datasets\tagged_data_extractions')
    )

