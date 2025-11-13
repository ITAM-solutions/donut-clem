"""
File name: synthetic_data_generator_improved
Author: Fran Moreno
Last Updated: 8/26/2025
Version: 1.0
Description: TOFILL
"""
import os.path
import random
import copy
import docx2pdf
import json
import pymupdf
import io
from copy import deepcopy

from typing import List, Union, Dict, Tuple
from docxtpl import DocxTemplate
from pathlib import Path

from synthetic_generator.factory import GeneratorFactory
from synthetic_generator.template_parser import JinjaCustomParser

from clem.model import DonutCLEM

IMS_PATH = Path(r'data/images')


class SyntheticDataGenerator:
    """ # TODO: fill """

    def __init__(self, template_path: Path, output_folder: Path, model_output_limit: int):
        self.template_path = template_path
        self.output_folder = output_folder
        self.model_output_limit = model_output_limit

        self.ims_path = IMS_PATH

        self.factory = GeneratorFactory(template_path)
        self._model_path = Path(r"C:\Users\FranMoreno\ITAM_software\repositories\donut-clem\weights\20251016_090333")
        self.model = DonutCLEM(self._model_path)

        self.tpl = DocxTemplate(self.template_path)
        self.jinja_vars = sorted(list(self.tpl.get_undeclared_template_variables()))

        self.context = {}  # To contain all data to be rendered into the template.
        self.root_info = {}  # To contain all raw data so it is not generated twice if already exists.
        self.gt_data = {}  # To be later converted to JSON.

    def generate(self):
        """
        # TODO: fill

        :return:
        """
        self.populate_images()
        self.populate_template()
        self.save()

    def populate_images(self):
        """
        TODO fill
        :return:
        """
        img_classes_paths = self.ims_path.iterdir()

        for img_class_path in img_classes_paths:
            placeholder_im_path = img_class_path / "placeholder.png"
            with open(placeholder_im_path, "rb") as fp:
                placeholder_im_b = io.BytesIO(fp.read())

            ims_options = list((img_class_path / "samples").glob('*.png'))
            im_path = random.choice(ims_options)
            with open(im_path, "rb") as fp:
                im_b = io.BytesIO(fp.read())

            self.tpl.replace_media(placeholder_im_b, im_b)

    def populate_template(self):
        """
        TODO fill
        :return:
        """

        self.context['cp'] = self.factory.generate_color_palette()
        # self.context['fields'] = self.factory.inv_fields

        for jinja_var in self.jinja_vars:

            if jinja_var in ['fields', 'cp']:  # Already populated when factory initialized.
                continue

            # 1. Process jinja var name to determine expected content
            parser = JinjaCustomParser(jinja_var)
            parser.get_segments()
            root_name = parser.root_name

            # 2. Choose data generator, or pick generated data if existing for current variable.
            if root_name in self.root_info:
                content = copy.deepcopy(self.root_info[root_name])
            else:
                quantity, missing = parser.get_repeat_count(parser.range)
                content = self.factory.generate(
                    parser.name,
                    quantity=quantity,
                    missing=missing,
                    mods=parser.mods,
                    gt_data=self.gt_data
                )

            if content and not self.root_info.get(root_name):
                self.root_info[root_name] = copy.deepcopy(content)

            # And now, apply modifiers and rich format to the raw content that was generated for this variable.
            # TODO: Implement modifiers.
            # content = self.factory.apply_modifiers(content, mods=parser.mods)

            try:
                self.context[jinja_var] = content['rich']
            except TypeError:
                raise TypeError(f"{jinja_var}")

        self._add_products_shared_data()
        a = 1

    def save(self):
        # Prepare output folder
        template_name = self.template_path.stem

        template_output_path = self.output_folder / template_name

        if not os.path.exists(template_output_path):
            os.makedirs(template_output_path)

        current_max_idx = self._find_max_idx_for_template(template_output_path)

        new_sample_name = template_name + '_' + str(current_max_idx).zfill(3)

        sample_base_name = template_output_path / new_sample_name

        # Save docx file
        self._escape_special_characters()
        docx_file = sample_base_name.with_suffix('.docx')
        self.tpl.render(self.context)
        self.tpl.save(sample_base_name.with_suffix('.docx'))

        # Convert to PDF and save PDF
        pdf_file = sample_base_name.with_suffix('.pdf')
        docx2pdf.convert(docx_file, pdf_file)

        # Convert to Image, save image
        with pymupdf.open(pdf_file) as pdf_obj:
            for page_idx, page in enumerate(pdf_obj):
                page_basename = sample_base_name.with_name(
                    sample_base_name.stem + f'-{str(page_idx).zfill(2)}' + '.png'
                )

                samples_from_page = self.generate_samples_from_page(page)

                for sample_idx, (pixmap, partial_gt, token_count) in enumerate(samples_from_page):
                    sub_image_name = page_basename.with_name(
                        page_basename.stem + f'-{str(sample_idx).zfill(2)}' + page_basename.suffix
                    )
                    pixmap.save(sub_image_name)

                    json_file = sub_image_name.with_suffix('.json')
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(partial_gt, f, indent=4, ensure_ascii=False)  # noqa

        # DOCX and PDF not needed anymore
        # os.remove(docx_file)
        # os.remove(pdf_file)

    def generate_samples_from_page(self, page, clip_heights: Tuple[int, int] = (0, 4), generated_samples: List = None):
        if generated_samples is None:
            generated_samples = list()

        w, h = page.rect.width, page.rect.height
        hmin, hmax = clip_heights
        clip = pymupdf.Rect(0, h * hmin // 4, w, h * hmax // 4)

        actual_gt_data = self._cleanup_gt_data(page, clip, self.factory.expected_product_fields)
        limit_exceeded, tokens_count = self.model.exceeds_output_limit(actual_gt_data, self.model_output_limit)

        if limit_exceeded and ((hmax - hmin) // 2) > 0:  # Jump to next division until we get to quarters
            l = hmax - hmin
            heights_halves = [
                (hmin, hmax - l // 2),
                (hmax - l // 2, hmax),
            ]
            for new_heights in heights_halves:
                generated_samples = self.generate_samples_from_page(page, clip_heights=new_heights, generated_samples=generated_samples)
            return generated_samples
        else:
            pixmap = page.get_pixmap(dpi=200, clip=clip)
            generated_samples.append((pixmap, actual_gt_data, tokens_count))
            return generated_samples


    """
    def foo(page, heights: List[int, int] = [0, 4]):
        w, h = page.rect.width, page.rect.height
        hmin, hmax = heights
        clip = pymupdf.Rect(0, h * (hmin // h), w, h * (hmax // h))
        text = get_text(page, clip)
        limit_exceeded = clean_gt(current_gt, text)
        if limit_exceeded and (hmax // 2) > 0: # Jump to next division if possible.
            heights_halves = [
                (hmin, hmax // 2),
                (hmax // 2, hmax),
            ]
            for new_heights in heights_halves:
                foo(page, new_heights)
        else:
            # Save the corresponding image portion with its gt json.
    """


    def _add_products_shared_data(self):
        if not self.gt_data.get('products'):
            return

        shared_product_field_generators = [
            ('prMetric', 'met'),
            ('prMetricGroup', 'metgr'),
            ('prDateFrom', 'dfrom'),
            ('prDateTo', 'dto'),
            ('prQty', 'qty'),
        ]

        for generator, key in shared_product_field_generators:
            if self.context.get(generator):
                for product in self.gt_data['products']:
                    product[key] = self.context[generator]['text']


    def _cleanup_gt_data(self, page, clip, expected_product_fields: List[str]):
        print(f"Expected product fields:\n{', '.join(expected_product_fields)}")
        text_in_page = page.get_text("text", clip=clip).replace('\n', '').replace(' ', '').lower()

        gt_data_page = deepcopy(self.gt_data)

        for gt_key, val in gt_data_page.items():
            if gt_key == "products":
                incomplete_products = []
                for idx, product in enumerate(gt_data_page['products']):
                    for field_name, field_val in product.items():
                        clean_val = field_val.replace(' ', '').lower() if field_val is not None else None
                        if not clean_val or clean_val not in text_in_page:
                            product[field_name] = None

                    # for expected_key in expected_product_fields:
                    #     val = product.get(expected_key, '').replace(' ', '').lower()
                    #     if val not in text_in_page or val == '':
                    #         incomplete_products.append(idx)

                # Remove totally empty products.
                gt_data_page['products'] = [
                    product for product in gt_data_page['products']
                    if any([product[k] for k in expected_product_fields])
                ]
                # gt_data_page['products'] = [
                #     product for idx, product in enumerate(gt_data_page['products'])
                #     if idx not in incomplete_products
                # ]
            else:
                val = val.replace(' ', '').lower()
                if val not in text_in_page or val == '':
                    gt_data_page[gt_key] = None

        if gt_data_page.get("products") and gt_data_page.get("products") == []:
            del gt_data_page['products']

        return gt_data_page

    def _escape_special_characters(self):
        for k in self.context:

            if k == 'cp':
                continue

            self.context[k] = self._escape_special_chars_in_tree(self.context[k])

    def _escape_special_chars_in_tree(self, item: Union[Dict, str, List]):
        if type(item) == str:
            item = self._escape_chars(item)
        elif type(item) == dict:
            for k in item:
                if k == 'table':
                    continue
                item[k] = self._escape_special_chars_in_tree(item[k])
        elif type(item) == list:
            for idx in range(len(item)):
                item[idx] = self._escape_special_chars_in_tree(item[idx])
        else:
            return item
        return item

    @staticmethod
    def _escape_chars(text: str):
        escape_mapping = {
            '&': '&amp;',
            '<': '%lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&apos;'
        }
        for char, escape in escape_mapping.items():
            text = text.replace(char, escape)

        return text

    @staticmethod
    def _find_max_idx_for_template(template_output_path: Path) -> int:
        existing_indexes = [int(file_.stem.split('_')[-1].split('-')[0]) for file_ in template_output_path.glob('*.png')]
        return max(existing_indexes) + 1 if existing_indexes else 0

if __name__ == '__main__':
    # file_name = 'footer_error'
    # template_path = Path(fr'../synthetic_data/templates/tests/{file_name}.docx')
    # output_folder =  Path(fr'../synthetic_data/templates/tests/outputs/{file_name}.docx')

    # template_name = "contact_info"
    # # template_path = Path(fr"templates/{template_name}.docx")
    # template_path = Path(fr'../synthetic_data/templates/tests/{template_name}.docx')
    # output_folder = Path(r"output")
    # # output_path =  Path(fr'../synthetic_data/templates/tests/outputs/{template_name}.docx')

    template_path = Path(fr"templates/multitables/template_021_001.docx")
    # template_path = Path(fr"templates/special/template_017_000.docx")

    output_folder = Path(r"output")

    synth_data_gen = SyntheticDataGenerator(template_path, output_folder, model_output_limit=768)
    synth_data_gen.generate()
