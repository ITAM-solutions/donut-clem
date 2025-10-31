"""
File name: donut_clem
Author: Fran Moreno
Last Updated: 10/27/2025
Version: 2.0
Description: this file implements DonutCLEM, which is an interface built over Donut model to handle its inputs and
normalize its outputs. Use this class to abstract its inner behaviour and focus on its integration within a
Document data extraction tool.
"""
from donut import DonutModel
from pathlib import Path
from pydantic import ValidationError
from PIL import Image
from tempfile import TemporaryDirectory
from typing import List, Tuple
from enum import Enum
import pymupdf

from clem.prediction_schema import PredictionSchema, get_empty_prediction
from clem.candidates.collector import CandidateCollector
from clem.candidates.merger import CandidateSelector


class ModelState(str, Enum):
    """ Enumerator that defines the possible working states of DonutCLEM class. """
    PRODUCTION = "Production"
    EVALUATION = "Evaluation"


class DonutCLEM:
    """ Interface over Donut model implementation for easier input/output processing. """

    def __init__(self, model_path: Path, mode: ModelState = ModelState.PRODUCTION, max_im_divisions: int = 2):
        """
        Class constructor.

        :param model_path: Path to an existing pre-trained/fine-tuned Donut model.
        :param mode: working mode/state. Must be a ModelState value.
        :param max_im_divisions: Maximum number of image divisions to perform during recursive inference.
        """
        self.model_path: Path = model_path
        self.mode = mode
        self.max_im_divisions = max_im_divisions if self.mode == ModelState.PRODUCTION else 0

        self.model, self.model_task = self._load_model_from_local(model_path)
        self.prompt = f"<s_{self.model_task}>"

    def predict_document(self, document_path: Path):
        """
        1. If PDF:
            1.1. Split into pages and save them into a temporary directory.
            1.2. Execute `predict` for each image, combining their solutions into the same CandidateCollector.
            1.3. Merge everything.

        # TODO NOT FINISHED!!
        :param document_path:
        :return:
        """
        tmp_dir_obj = TemporaryDirectory()
        tmp_dir = Path(tmp_dir_obj.name)

        candidates_collector = CandidateCollector()  # Will collect value candidates for each field.

        # Let's assume by now that the document is always a PDF file.
        if document_path.suffix.lower() == '.pdf':
            # Read with PyMuPDF
            pdf_obj = pymupdf.open(document_path)
            for page in pdf_obj:
                pixmap = page.get_pixmap(dpi=200)
                im_path = tmp_dir / f"{document_path.stem}_p{page.number}.png"
                pixmap.save(im_path)

                self.predict(im_path, output=candidates_collector, page=page.number)

            # After prediction collection:
            final_prediction = CandidateSelector.merge(candidates_collector)
            print(final_prediction)
        else:
            # TODO implement for images, maybe even docx.
            return None

        tmp_dir_obj.cleanup()


    def predict(self,
            im_path: Path,
            output: CandidateCollector,
            divisions: int = 0,
            **metadata
    ) -> None:
        """
        This recursive method executes the model inference. In case that the model's prediction doesn't fit the
        expected data schema, it will keep trying to produce inference by dividing the input image in multiple
        sub-images, and concatenating their outputs in the same CandidateCollector object.

        :param im_path: Path to the image that will be used as the model's input to compute inference.
        :param divisions: number of current image divisions performed.
        :param output: CandidateCollector object that keeps track of all the previous predictions.
        :return: None (Modifies the output argument in place).
        """
        try:
            if 'sections' in metadata:
                metadata['section'] = self._compute_sections(metadata['sections'])
                # del metadata['sections']
            output.add(self._inference(im_path), metadata)
        except ValidationError:  # Pydantic raises ValidationError is schema validation failed.
            if divisions < self.max_im_divisions:
                # Save sub-images to temporary directory.
                sub_ims = self._split_image_in_half(im_path)
                for idx, sub_im in enumerate(sub_ims):
                    sections = metadata.get('sections', [])
                    sections.append(idx)
                    metadata['sections'] = sections

                    self.predict(sub_im, output=output, divisions = divisions+1, **metadata)
            else:  # Max iteration depth reached. Could not extract data. Returning empty prediction with error status.
                output.add(get_empty_prediction(raised_error=True), metadata)

    def _inference(self, im_path: Image) -> PredictionSchema:
        """
        Executed model's inference using the given image and the specified task.

        :raises ValidationError: if output doesn't fit the PredictionSchema model, a ValidationError is raised.

        :param im_path: path to the image to be used for inference.
        :return: model's output formatted as a PredictionSchema instance.
        """
        im = Image.open(im_path)
        output_raw = self.model.inference(image=im, prompt=self.prompt)["predictions"][0]
        output = PredictionSchema(**output_raw)
        return output

    @staticmethod
    def _split_image_in_half(im_path: Path) -> List[Path]:
        """
        Utility method that divides an image in half, and saves both halves at the same directory as the original image.
        Please note that the destination directory is expected to be temporary, and will be removed at some point
        during execution.

        :param im_path: Path to source image.
        :return: List containing the path to each resulting sub-image.
        """
        im = Image.open(im_path)
        height_half = im.height // 2

        ims = [
            im.crop((0, 0, im.width, height_half)),
            im.crop((0, height_half, im.width, im.height))
        ]

        ims_paths = []
        for idx, sub_im in enumerate(ims):
            sub_im_path = im_path.parent / f"{im_path.stem}_s{idx}.png"
            sub_im.save(sub_im_path)
            ims_paths.append(sub_im_path)

        return ims_paths

    @staticmethod
    def _load_model_from_local(model_path: Path) -> Tuple[DonutModel, str]:
        """
        Loads a model's weights from the given folder path, and reads the task name for which it was trained for.

        :param model_path: Path to the model's folder.
        :return: Loaded model, task name
        """
        model = DonutModel.from_pretrained(model_path)

        task_file = model_path / 'task.txt'

        if not task_file:
            raise FileNotFoundError("Please make sure that your model has its purpose task defined in a "
                                    "'task.txt' file inside the model's folder.")

        with open(task_file, 'r') as fp:
            task = fp.readline().replace('\n', '').strip()

        return model, task

    @staticmethod
    def _compute_sections(sections: list) -> int:
        """

        :param sections:
        :return:
        """
        return sum([i * (2 ** idx) for idx, i in enumerate(sections[::-1])])


if __name__ == '__main__':
    MODEL_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\weights\20251016_090333")
    # donut = DonutCLEM(MODEL_PATH, mode=ModelState.EVALUATION)
    #
    # # Test with sample that generates good output schema:
    # # IM_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\samples\Blank\abigail_05.pdf_003.png")
    # # output = donut.predict(IM_PATH)
    # # print(output)
    #
    # # Test with sample that produces hallucination
    # IM_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\samples\Multiple Tables\abigail_04.pdf_001.png")
    # output = donut.predict(IM_PATH)
    # print(output)

    # Test with full document
    DOCUMENT_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\documents\abigail_01.pdf")
    donut = DonutCLEM(MODEL_PATH)

    donut.predict_document(DOCUMENT_PATH)

