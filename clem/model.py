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
from clem.prediction_schema import PredictionSchema, get_empty_prediction
from clem.combination_logic import CandidateCollector
from enum import Enum


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

    def predict(self,
            im_path: Path,
            divisions: int = 0,
            output: CandidateCollector = None
    ) -> CandidateCollector:
        """
        This recursive method executes the model inference. In case that the model's prediction doesn't fit the
        expected data schema, it will keep trying to produce inference by dividing the input image in multiple
        sub-images, and concatenating their outputs.

        :param im_path: Path to the image that will be used as the model's input to compute inference.
        :param divisions: number of current image divisions performed.
        :param output: list of previous outputs from image divisions (if existing).
        :return: list of outputs from inference.
        """
        def _combine_outputs(outputs: CandidateCollector, recursion_depth: int) -> CandidateCollector:
            """ If located at recursion depth 0 (divisions == 0), combines the collected outputs. """
            if recursion_depth == 0:
                return outputs.merge()

        if output is None:
            output = CandidateCollector()  # Initially empty

        try:
            output.add(self._inference(im_path))
            return _combine_outputs(output, divisions)
        except ValidationError:  # Pydantic raises ValidationError is schema validation failed.
            if divisions < self.max_im_divisions:
                # Save sub-images to temporary directory.
                with TemporaryDirectory() as tmp_dir:
                    sub_ims = self._split_image_in_half(im_path, tmp_dir)
                    for sub_im in sub_ims:
                        self.predict(sub_im, divisions = divisions+1, output=output)
                    return _combine_outputs(output, divisions)
            else:  # Max iteration depth reached. Could not extract data. Returning empty prediction with error status.
                output.add(get_empty_prediction(raised_error=True))
                return _combine_outputs(output, divisions)

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
    def _split_image_in_half(im_path: Path, tmp_dir: str) -> List[Path]:
        """
        Utility method that divides an image in half, and saves both halves at the given directory.
        Please note that the destination directory is expected to be temporary, and will be removed at some point
        during execution.

        :param im_path: Path to source image.
        :param tmp_dir: Folder where image halves will be (temporarily) saved.
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
            tmp_im_path = Path(tmp_dir) / f"{im_path.stem}_{idx}.png"
            sub_im.save(tmp_im_path)
            ims_paths.append(tmp_im_path)

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


if __name__ == '__main__':
    MODEL_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\weights\20251016_090333")
    donut = DonutCLEM(MODEL_PATH, mode=ModelState.EVALUATION)

    # Test with sample that generates good output schema:
    # IM_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\samples\Blank\abigail_05.pdf_003.png")
    # output = donut.predict(IM_PATH)
    # print(output)

    # Test with sample that produces hallucination
    IM_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\samples\Multiple Tables\abigail_04.pdf_001.png")
    output = donut.predict(IM_PATH)
    print(output)
