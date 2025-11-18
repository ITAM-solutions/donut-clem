"""
File name: data_normalizer
Author: Fran Moreno
Last Updated: 6/2/2025
Version: 1.0
Description: TOFILL
"""
import os
import json
import pymupdf

from tqdm import tqdm
from pathlib import Path

from donut.data_preparation.output_template import OutputJSON


class DataNormalizer:

    def __init__(self, output_path: Path):

        self.output_path = output_path

        self._processed_samples_file = self.output_path / '.processed_samples'
        self._processed_samples = self._load_processed_samples()

        self.current_sample_path = None

        self.next_idx = self._get_next_idx()

        self.sample_metadata_name = '.sample_metadata'

    def _load_processed_samples(self):
        if not self._processed_samples_file.exists():
            return []

        with open(self._processed_samples_file, 'r') as f:
            return [l.replace('\n', '') for l in f.readlines()]

    def _add_processed_sample(self, sample_path: Path):
        with open(self._processed_samples_file, 'a') as f:
            f.write(str(sample_path))

    def add_sample(self, sample_path: Path):
        """
        Adds a new sample (located at `sample_path`) to the normalized dataset.

        :return:
        """
        if str(sample_path) in self._processed_samples:
            print(f'Sample from path "{str(sample_path)}" has been already formatted.')
            return

        self.current_sample_path = sample_path

        # 1. Read sample metadata (.sample_metadata)
        sample_metadata = self._load_sample_metadata()

        # 2. Get list of PDF files. Flag those with a JSON with the same name.
        files_to_process = self._get_list_of_files()

        # 3. For each file:
        for file_pdf_path, has_json in files_to_process:
            document_object = pymupdf.open(file_pdf_path)

            # 3.1. Create new folder in `output_path` using `self.current_idx + 1`
            file_output_path = self.output_path / self.next_idx
            os.makedirs(file_output_path)

            # 3.2. Extract file images.
            file_images = self._extract_file_images(document_object)
            images_paths = self._save_file_images(file_images, file_output_path)

            # 3.3. If file has JSON:
            if has_json:
                file_json_path = file_pdf_path.with_suffix('.json')
                output_template = self._get_formatted_template_from_excel_data(file_json_path)
                for image_path in images_paths:
                    output_template_path = image_path.with_suffix('.json')
                    output_template.save(output_template_path)

                    # self._save_output_template(image_path, output_template)
                # output_templates = self._get_output_template_for_images(images_paths, file_json_path)
                # self._save_output_templates(output_templates)

            # 3.4. Save file metadata using sample metadata and file info.
            self._save_file_metadata(document_object, sample_metadata, file_output_path)

            # 3.5. Update next available index.
            self.next_idx = self._get_next_idx()

        self._add_processed_sample(sample_path)

    def _get_next_idx(self) -> str:
        """
        Finds the next index to be used for a new item in the dataset located at `output_path`,
        and retrieves it in normalized string format (e.g., '000012').

        :return:
        """
        def convert_to_idx(idx_str: str) -> int:
            try:
                return int(idx_str)
            except ValueError:
                return -1

        normalized_sets = [folder for folder in os.listdir(self.output_path) if os.path.isdir(self.output_path / folder)]
        existing_indexes = [convert_to_idx(idx_str) for idx_str in normalized_sets]
        next_idx = max(existing_indexes) + 1 if existing_indexes else 0
        return str(next_idx).zfill(6)

    def _load_sample_metadata(self) -> dict:
        """
        Reads sample metadata and returns in dictionary format.

        :return:
        """
        sample_metadata = dict()

        if not self.current_sample_path:
            return sample_metadata

        sample_metadata_path = self.current_sample_path / self.sample_metadata_name

        if not sample_metadata_path.exists():
            return {}

        with open(sample_metadata_path, 'r') as f:
            lines = f.readlines()

        sample_metadata = {line[0]: line[1] for line in [x.replace('\n', '').split('=') for x in lines]}
        return sample_metadata

    def _get_list_of_files(self) -> list[tuple[Path, bool]]:
        """
        Returns a list of tuples with two elements each, containing:
         - PDF file name, without extension.
         - True if this PDF has a JSON attached to it, false otherwise.
        """
        pdf_files = list(self.current_sample_path.glob('*.pdf'))
        json_files = {f.stem for f in self.current_sample_path.glob('*.json')}

        file_pairs = [(pdf_file, pdf_file.stem in json_files) for pdf_file in pdf_files]

        return file_pairs

    @staticmethod
    def _extract_file_images(document: pymupdf.Document) -> list[pymupdf.Pixmap]:
        """
        Returns a list with an image for each page in the given PDF file.

        :param document:
        :return:
        """
        return [page.get_pixmap(dpi=50) for page in document]

    @staticmethod
    def _get_formatted_template_from_excel_data(source_data_path: Path):
        """

        :param source_data_path:
        :return:
        """
        output_json = OutputJSON()

        with open(source_data_path, 'r') as f:
            source_data = json.load(f)
            items_data = source_data["values"]

        for item_data in items_data:
            output_json.add_item(item_data)

        return output_json

    def _save_file_images(self, image_list: list[pymupdf.Pixmap], dest_path: Path) -> list[Path]:
        """
        Saves a list of images (each of them represented as a `pymupdf.Pixmap` object) to the given destiny path.

        :param image_list:
        :param dest_path:
        :return:
        """
        im_paths = []
        for i, image in enumerate(image_list):
            im_path =  dest_path / (self.next_idx + '_' + str(i).zfill(3) + '.png')
            im_paths.append(im_path)
            image.save(im_path)
        return im_paths

    @staticmethod
    def _save_output_templates(output_templates: dict[Path, dict]):
        for file_path, content in output_templates.items():
            with open(file_path, 'w') as f:
                json.dump(content, f)

    @staticmethod
    def _save_file_metadata(document: pymupdf.Document, sample_metadata: dict, file_output_path: Path):
        file_metadata = {
            'origin': {
                'name': document.name,
                'creation_date': document.metadata.get('creationDate'),
            },
            **sample_metadata
        }

        with open(file_output_path / '.file_metadata.json', 'w') as f:
            json.dump(file_metadata, f, indent=4)


def process_samples_in_folder(samples_path: Path, output_path: Path):
    samples_list = [samples_path / sample_dir for sample_dir in os.listdir(samples_path)
        if os.path.isdir(samples_path / sample_dir)]

    data_normalizer = DataNormalizer(output_path)

    for sample in tqdm(samples_list):
        data_normalizer.add_sample(sample)


if __name__ == '__main__':

    output_path = Path(r'C:\Users\FranMoreno\datasets\donut_lr_2')
    samples_path = Path(r'C:\Users\FranMoreno\datasets\c_and_o_outputs_v2')

    process_samples_in_folder(samples_path, output_path)
