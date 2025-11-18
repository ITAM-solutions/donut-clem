"""
File name: tagging_tool
Author: Fran Moreno
Last Updated: 6/4/2025
Version: 1.0
Description: TOFILL
"""
import os
import random
from pathlib import Path
import json
import numpy as np
import cv2
import threading
import time
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from screeninfo import get_monitors
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.text import Text
import pytesseract
from rich import print_json
from prompt_toolkit import prompt
import pygetwindow as gw
from pywinauto.application import Application


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from data_preparation.output_template import OutputJSON, readable_names


class CLI:
    def __init__(self):
        self.console = Console()

    def info(self, msg: str, as_title=False):
        text = Text(msg, justify='right')

        if as_title:
            text.stylize('bold magenta')

        self.console.print(text)

    @staticmethod
    def print_json_pretty(json_data):
        pretty_json = {readable_names[key]: value for key, value in json_data.items()}
        print_json(json.dumps(pretty_json), indent=4)

    @staticmethod
    def ask(msg: str, default=None):
        if default is None:
            msg += ' (Press Enter to leave empty)'
        answer = Prompt.ask(msg, default=default, show_default=True)
        return answer

    @staticmethod
    def ask_yes_or_no(msg: str, default=True) -> bool:
        answer = Confirm.ask(msg, default=default)
        return answer

    @staticmethod
    def choose(msg: str, options: list[str], default: str = None):
        if default is None and len(options) >= 1:
            default = options[0]

        return Prompt.ask(msg, choices=options, default=default)


class TaggingTool:
    def __init__(self, dataset_path: Path, source: str = 'documents'):
        self.dataset_path = dataset_path
        self.source = source

        self._tagged_documents_file = self.dataset_path / '.tagged_documents'
        # self._tagged_documents = self._load_tagged_documents()
        self.documents_sources = self._load_dataset()

        self.cli = CLI()

        self.prompt_thread = None
        self.thread_flag = threading.Event()
        self.select_roi_flag = threading.Event()

        self.shared_variable = None

        self.current_document_shared_values = dict()

    def run(self, shuffle=False):
        """
        CLI Tool to tag images for Donut.
        :return:
        """
        if shuffle:
            random.shuffle(self.documents_sources)

        if self.source == 'documents':
            for document_source in self.documents_sources:
                ts_doc = time.time()
                images_paths = [Path(i) for i in list(document_source.glob('*.png'))]

                for image_path in images_paths:
                    ts_page = time.time()
                    info_text = (f'\nFile {int(document_source.name)+1}/{len(self.documents_sources)} | '
                                 f'Image {int(image_path.stem.split('_')[1]) + 1}/{len(images_paths)}')
                    self.cli.info(info_text, as_title=True)

                    self.thread_flag.set()

                    self.prompt_thread = threading.Thread(target=self.tag_image, args=[image_path])
                    self.prompt_thread.start()

                    self._display_and_wait(image_path)
                    tf_page = time.time()
                    tt_page = tf_page - ts_page
                    self.cli.info(f"Time spent on page: {self._seconds_to_minutes(tt_page)}")

                self._set_as_tagged(document_source)
                tf_doc = time.time()
                tt_doc = tf_doc - ts_doc
                self.cli.info(f"Time spent on document: {self._seconds_to_minutes(tt_doc)}")
        else:
            for image_path in self.documents_sources:
                ts_page = time.time()
                self.cli.info(image_path.stem, as_title=True)

                self.thread_flag.set()
                self.prompt_thread = threading.Thread(target=self.tag_image, args=[image_path])
                self.prompt_thread.start()

                self._display_and_wait(image_path)
                tf_page = time.time()
                tt_page = tf_page - ts_page
                self.cli.info(f"Time spent on page: {self._seconds_to_minutes(tt_page)}")


    def tag_image(self, im_path: Path):
        output_prompt = OutputJSON()
        output_prompt.load_existing_json(im_path)

        self._modify_or_populate_shared_fields(output_prompt.template)
        self._modify_or_delete_current_products(output_prompt.template['products'])
        self._add_new_items(output_prompt.template['products'], output_prompt.item_template)

        self.save_output(output_prompt.template, im_path)

        self.thread_flag.clear()

    def _load_tagged_documents(self):

        if self._tagged_documents_file.exists():
            with open(self._tagged_documents_file, 'r') as fp:
                return [Path(l.replace('\n', '')) for l in fp.readlines()]

        return []

    def _set_as_tagged(self, file_path: Path):
        tagged_file = file_path / '.tagged'
        with open(tagged_file, 'w') as fp:
            fp.write('')

    def _load_dataset(self):
        if self.source == 'documents':
            documents_folders = [self.dataset_path / i for i in os.listdir(self.dataset_path)
                if os.path.isdir(self.dataset_path / i) and '.' not in (self.dataset_path / i).name
                   and not (self.dataset_path / i / '.tagged').exists()]
            return documents_folders
        else:
            images_list = [im_path for i in os.listdir(self.dataset_path) if (im_path := self.dataset_path / i).suffix == '.png']
            return images_list

    def _modify_or_populate_shared_fields(self, output_template: dict):
        """
        If the current image has a JSON attached to it, this method will let the user modify its shared values.
        If no JSON exists, the user will be allowed to populate all of them manually or with an image selector.

        :param output_template:
        :return:
        """
        for field_ in output_template:
            if field_ == 'products':
                continue

            self._interact_with_field_value(output_template, field_)

    def _modify_or_delete_current_products(self, products):
        """
        If the current image has a JSON attached to it, this method will let the user modify (or delete) the
        items listed in it. If no JSON exists, there will be no iterations in this method.

        :param products:
        :return:
        """

        # Ask to delete items
        to_delete = []
        for idx, product in enumerate(products):
            # Ask to delete item or not
            self.cli.info(f'\nItem "{product["name"]}"')
            self.cli.print_json_pretty(product)

            option = self.cli.choose(f'Keep (1), Update (2) or Delete (3)?', options=["1", "2", "3"], default="1")
            if option == "1":  # Keep product, no changes
                continue

            elif option == "2":  # Update product fields
                for item_field in product:

                    self._interact_with_field_value(product, item_field)

            else:  # Remove product
                to_delete.append(idx)

            self.cli.info(f'Finished with item "{product["name"]}"\n--------')

        self.cli.info("\nStarting to check products...")

        # Delete selected products
        items_to_keep = [i for i in range(len(products)) if i not in to_delete]
        original_products = products.copy()
        products.clear()
        products.extend([original_products[i] for i in items_to_keep])

    def _add_new_items(self, products, item_template):
        """
        Lets the user add new items to the current template.

        :param products:
        :return:
        """
        add_new_item = self.cli.ask_yes_or_no("\nAdd a new item?")
        while add_new_item:
            new_item = {k: None for k in item_template.keys()} if not products else products[-1].copy()

            for item_field in item_template:
                self._interact_with_field_value(new_item, item_field)

            products.append(new_item)

            self.cli.info(f'Finished adding item "{new_item["name"]}"\n---------')
            add_new_item = self.cli.ask_yes_or_no("\nAdd another item?")

    def _interact_with_field_value(self, item, field_):
        """
        Display options to the user to modify (or populate) a certain value in the current JSON.

        :param item:
        :param field_:
        :return:
        """
        current_value = item[field_]
        self.cli.info(f'\nField "{readable_names[field_]}". Current value: "{current_value}"')

        method = self.cli.choose("Keep (1), Edit (2), Type new (3), Select (4) from image or Leave Empty (5)?", options=["1", "2", "3", "4", "5"], default="1")

        if method == "2":
            str_value = "" if not current_value else str(current_value)
            item[field_] = prompt("Modify the extracted value: ", default=str_value)
            # item[field_] = self.cli.ask(f'Type value for {readable_names[field_]}')

        elif method == "3":
            text = self.cli.ask("Write the value")
            item[field_] = text

        elif method == "4":
            self.select_roi_flag.set()

            while self.select_roi_flag.is_set():
                time.sleep(0.1)

            value = self.shared_variable
            item[field_] = value

        elif method == "5":
            item[field_] = None

    def _display_and_wait(self, im_path: Path):
        image, window_name = self._display_image(im_path)

        while self.thread_flag.is_set():
            if cv2.waitKey(50) == 27:
                self.thread_flag.clear()
                del self.prompt_thread

            if self.select_roi_flag.is_set():
                try:
                    text = self._extract_text_from_image(window_name, image)
                except Exception:
                    self.cli.info("\nAn error happened during data extraction. Try again.\n")
                    continue
                choice = self.cli.choose(f'Extracted text: "{text}". Keep (1), Modify (2), Try again (3) or Exit selection mode? (4)', ["1", "2", "3", "4"], default="1")

                if choice == "2":
                    text = prompt("Modify the extracted value: ", default=text)
                elif choice == "3":
                    continue
                elif choice == "4":
                    text = self.cli.ask("Write the value manually")

                self.shared_variable = text
                self.select_roi_flag.clear()

        self._close_image(window_name)

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

        return image, window_name

    @staticmethod
    def _close_image(window_name):
        cv2.destroyWindow(window_name)

    @staticmethod
    def _extract_text_from_image(window_name, image):
        bbox = cv2.selectROI(window_name, image, showCrosshair=False)
        x, y, w, h = bbox
        im_section = image[y:y + h, x:x + w]
        text = pytesseract.image_to_string(im_section, lang='eng+nld')  # English or Dutch
        text = text.replace('\n', '').strip()
        windows = gw.getWindowsWithTitle('CLEM-Automation_branch2 – tagging_tool.py')
        if not windows:
            windows = gw.getWindowsWithTitle("Windows PowerShell")
        if windows:
            app = Application().connect(handle=windows[0]._hWnd)
            app.top_window().set_focus()

        return text

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

    @staticmethod
    def save_output(output_json: dict, im_path: Path):
        json_path = im_path.with_suffix('.json')
        with open(json_path, 'w') as fp:
            json.dump(output_json, fp, indent=4)

    @staticmethod
    def _seconds_to_minutes(t: float):
        t = int(t)
        return f"{t // 60}min {t % 60}s"


if __name__ == '__main__':
    DATASET_PATH = Path(r'C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\dataset\evaluation\images')

    tagging_tool = TaggingTool(DATASET_PATH, source='images')
    tagging_tool.run(shuffle=False)
