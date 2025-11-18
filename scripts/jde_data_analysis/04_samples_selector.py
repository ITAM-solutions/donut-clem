"""
File name: samples_selector
Author: Fran Moreno
Last Updated: 8/20/2025
Version: 1.0
Description: Simple tool to quickly go over samples of the real dataset and select those that have an interesting
schema/layout to be emulated with a template to build synthetic data with it.
"""
from pathlib import Path
from PIL import Image
import cv2
from screeninfo import get_monitors
import numpy as np
import shutil


class SelectionTool:
    def __init__(self, dataset_path: Path, output_path: Path):
        self.dataset_path = dataset_path
        self.output_path = output_path

        self.samples_list = self._load_samples()

    def run(self):
        for sample in self.samples_list:
            cv2_window = self._display_image(sample)

            while True:
                k = cv2.waitKey(50)
                if k & 0xFF == ord('s'):
                    self.save_im(sample)
                    break
                elif k & 0xFF == ord('q'):  # Next image
                    break

            cv2.destroyWindow(cv2_window)

    def save_im(self, im: Path):
        shutil.copyfile(im, self.output_path / im.name)

    def _load_samples(self):
        samples = self.dataset_path.glob("**/*.png")
        return list(samples)

    def _display_image(self, im_path: Path):
        image_pil = Image.open(im_path)

        im_w, im_h = image_pil.size
        im_aspect = im_w / im_h

        monitor = self.get_biggest_monitor()

        max_width = monitor.width // 2
        max_height = monitor.height - 90

        target_width = max_width
        target_height = int(target_width / im_aspect)

        if target_height > max_height:
            target_height = max_height
            target_width = int(target_height * im_aspect)

        image_pil = image_pil.resize((target_width, target_height))
        image = np.asarray(image_pil)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        window_name = im_path.stem

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, monitor.x, monitor.y)
        cv2.imshow(window_name, image)

        return window_name

    @staticmethod
    def get_biggest_monitor():
        monitors = get_monitors()
        biggest_monitor = None
        for monitor in monitors:
            if not biggest_monitor:
                biggest_monitor = monitor
            else:
                if monitor.width_mm > biggest_monitor.width_mm:
                    biggest_monitor = monitor
        return biggest_monitor

if __name__ == '__main__':
    dataset_path = Path(r"C:\Users\FranMoreno\datasets\donut")
    output_path = Path(r"generic_layout_samples")

    selection_tool = SelectionTool(dataset_path, output_path)
    selection_tool.run()


