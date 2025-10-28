"""
File name: donut_clem
Author: Fran Moreno
Last Updated: 10/27/2025
Version: 1.0
Description: TOFILL
"""
from pathlib import Path
from donut import DonutModel
from pydantic import ValidationError
from PIL import Image
import json
from tempfile import TemporaryDirectory
from typing import List
from clem.output_schema import PredictionSchema


class DonutCLEM:
    def __init__(self, model_path: Path):
        self.model_path: Path = model_path

        self.model: DonutModel = DonutModel.from_pretrained(model_path)
        self.task_name = "dataset"
        self.prompt = f"<s_{self.task_name}>"

        self.max_im_divisions: int = 2

    def predict(self, im_path: Path, divisions: int = 0, output: list = None):
        if output is None:
            output = []

        try:
            output.append(self._inference(im_path))
            return output
        except ValidationError:
            if divisions < self.max_im_divisions:
                with TemporaryDirectory() as tmp_dir:
                    sub_ims = self._split_image_in_half(im_path, tmp_dir)
                    for sub_im in sub_ims:
                        self.predict(sub_im, divisions = divisions+1, output=output)
                    return output
            else:  # Max iteration depth reached. Will bypass previous output.
                return output

    def _inference(self, im_path: Image):
        im = Image.open(im_path)
        output = self.model.inference(image=im, prompt=self.prompt)["predictions"][0]
        PredictionSchema.model_validate_json(json.dumps(output))
        return output

    @staticmethod
    def _split_image_in_half(im_path: Path, tmp_dir: str) -> List[Path]:
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


if __name__ == '__main__':
    MODEL_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\weights\20251016_090333")
    donut = DonutCLEM(MODEL_PATH)

    # Test with sample that generates good output schema:
    # IM_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\samples\Blank\abigail_05.pdf_003.png")
    # output = donut.predict(IM_PATH)
    # print(output)

    # Test with sample that produces hallucination
    IM_PATH = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\samples\Multiple Tables\abigail_04.pdf_001.png")
    output = donut.predict(IM_PATH)
