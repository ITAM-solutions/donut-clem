# """
# File name: synthetic_data_generator
# Author: Fran Moreno
# Last Updated: 7/15/2025
# Version: 1.0
# Description: TOFILL
# """
# import docx2pdf
# import json
# import os.path
# import pandas as pd
# import pymupdf
# import random
# import shutil
# import time
#
# from datetime import datetime
# from docxtpl import DocxTemplate, InlineImage, RichText
# from docx.shared import Mm
# from faker import Faker
# from pathlib import Path
# from string import ascii_uppercase, digits
# from tqdm import tqdm
#
#
# class SyntheticDataGenerator:
#     def __init__(self, templates_path: Path, output_path: Path, initial_sample_idx: int = 0):
#         self.templates_path = templates_path
#         self.output_path = output_path
#         self.logos_path = Path(r'synthetic_data\sample_images')
#
#         self.renders_path = self.output_path / 'renders'
#         self.pdfs_path = self.output_path / 'pdfs'
#         self.results_path = self.output_path / 'synth_data'
#         self._init_output_folder()
#
#         self.gt_keys = ("id", "date", "cur", "vendor", "po")
#
#         self.sample_idx = initial_sample_idx
#
#         self.vendors = self._read_data_file(Path('synthetic_data/vendors.txt'))
#         self.metrics = self._read_data_file(Path('synthetic_data/metrics.txt'))
#         self.metric_groups = self._read_data_file(Path('synthetic_data/metric_groups.txt'))
#         self.currencies = self._read_data_file(Path('synthetic_data/currencies.txt'))
#         self.fonts = self._read_data_file(Path('synthetic_data/text_fonts.txt'))
#         self.date_formats = self._read_data_file(Path('synthetic_data/date_formats.txt'))
#         self.currency = ''
#         self.font = {}
#         self.date_format = ''
#         self.price_format = 0
#
#         self.fake = Faker()
#
#     def generate(self, samples_per_template):
#         templates_list = list(self.templates_path.glob('*.docx'))
#         for template in tqdm(templates_list, desc=" Templates", position=0):
#             template_name = template.name
#             for _ in tqdm(range(samples_per_template), desc="  No. samples", position=1, leave=False):
#                 self.currency = random.choice(self.currencies)
#                 self.font = {
#                     'family': random.choice(self.fonts),
#                     'text_size': random.randint(16, 19)
#                 }
#
#                 doc = DocxTemplate(template)
#                 data = self.generate_random_data(doc, template_name)
#                 gt_obj = {k: data[k] for k in self.gt_keys}
#                 gt_obj['products'] = [p.copy() for p in data['products']]
#                 self._enrich_data(data)
#
#                 doc.render(context=data)
#                 docx_path = self.output_path / 'renders' / f'{str(self.sample_idx).zfill(6)}_000.docx'
#                 doc.save(docx_path)
#                 self.sample_idx += 1
#
#                 self.save_sample(docx_path, gt_obj, template_name)
#
#     def generate_random_data(self, doc, template_name) -> dict:
#
#         self.date_format = random.choice(self.date_formats)
#         self.price_format = random.randint(0, 5)
#
#         data = {
#             'vendor': random.choice(self.vendors),
#             'id': self._get_random_id(),
#             'date': self._get_random_date(),
#             'po': self._get_random_id(),
#             'cur': self.currency,
#             'vendor_info_prose': self._get_random_text(min_words=25),
#             'vendor_info_long': self._get_random_text(min_words=300),
#             'vendor_info_very_long': self._get_random_text(min_words=450),
#             'vendor_info_structured': self._get_random_key_value_pairs((4, 6)),
#             'extra_info': self._get_random_text(min_words=40),
#             'logo': InlineImage(doc, image_descriptor=self._get_random_img('logo_only'), width=Mm(20), height=Mm(20)),
#             'products': self._get_random_products(template_name),
#             'pairs': self._get_random_key_value_pairs(),
#             **self._get_random_additional_info(),
#             **self._get_footer_info(doc, template_name)
#         }
#
#         return data
#
#     def cleanup(self, docx: bool = True, pdf: bool = True):
#         if not self.output_path.exists():
#             return
#
#         if docx and self.renders_path.exists():
#             shutil.rmtree(self.renders_path)
#
#         if pdf and self.pdfs_path.exists():
#             shutil.rmtree(self.pdfs_path)
#
#     @staticmethod
#     def _get_random_id(length_range: tuple[int, int] = (10, 15)) -> str:
#         vocab = ascii_uppercase + digits + '-'
#         text = ''.join(random.choice(vocab) for _ in range(random.randint(length_range[0], length_range[1])))
#         return text
#
#     def _get_random_date(self, date_from: datetime = None, date_to: datetime = None) -> str:
#         def str_time_prop(start, end, time_format, prop):
#             """Get a time at a proportion of a range of two formatted times.
#
#             start and end should be strings specifying times formatted in the
#             given format (strftime-style), giving an interval [start, end].
#             prop specifies how a proportion of the interval to be taken after
#             start.  The returned time will be in the specified format.
#             """
#
#             stime = time.mktime(time.strptime(start, time_format))
#             etime = time.mktime(time.strptime(end, time_format))
#
#             ptime = stime + prop * (etime - stime)
#
#             return time.strftime(time_format, time.localtime(ptime))
#
#         if not date_from:
#             date_from = datetime(2023, 1, 1).strftime(self.date_format)
#         if not date_to:
#             date_to = datetime(2025, 12, 31).strftime(self.date_format)
#
#         date_text = str_time_prop(date_from, date_to, self.date_format, random.random())
#         return date_text
#
#     def _get_random_text(self, min_words: int) -> RichText:
#         text = ""
#         while len(text.split()) < min_words:
#             text += self.fake.paragraph()
#         return RichText(text, font=self.font['family'], size=self.font['text_size'])
#         # return text
#
#     def _get_random_key_value_pairs(self, pairs_range: tuple[int, int] = (0, 4)):
#         num_pairs = random.randint(pairs_range[0], pairs_range[1])
#         pairs_df = pd.DataFrame({
#             'k': [self._get_short_lines((1, 1), (1, 2)) for _ in range(num_pairs)],
#             'v': [self._get_short_lines((1, 1), (0, 3)) for _ in range(num_pairs)],
#         })
#         return pairs_df.to_dict(orient='records')
#
#     def _get_short_lines(self,
#             num_lines_range: tuple[int, int] = (1, 3),
#             words_per_line_range: tuple[int, int] = (3, 5),
#             font_size: int = None
#     ):
#         num_lines = random.randint(*num_lines_range)
#         sep = "\n" if num_lines > 1 else ""
#         text = f"{sep}".join(' '.join(self.fake.words(nb=random.randint(*words_per_line_range))) for _ in range(num_lines))
#         if font_size:
#             return RichText(text, font=self.font['family'], size=font_size)
#         return RichText(text, font=self.font['family'])
#
#     def _get_random_img(self, logo_type: str):
#         im_folder = self.logos_path / logo_type
#         im_path = random.choice(list(im_folder.glob('*')))
#         return str(im_path)
#
#     def _get_random_products(self, template_name: str):
#         num_products, show_currency = self._get_template_based_data(template_name)
#         currency_position = random.choice([True, False]) if show_currency else None
#
#         products_df = pd.DataFrame({
#             'sku': [self._get_random_id(length_range=(3, 5)) for _ in range(num_products)],
#             'name': [self._get_random_product_name() for _ in range(num_products)],
#             'met': [random.choice(self.metrics) for _ in range(num_products)],
#             'metgr': [random.choice(self.metric_groups) for _ in range(num_products)],
#             'qty': [str(random.randint(0, 5000)) for _ in range(num_products)],
#             'unpr': [self._get_random_price(show_currency, currency_position, max_price=99999) for _ in range(num_products)],
#             'totpr': [self._get_random_price(show_currency, currency_position) for _ in range(num_products)],
#             'drom': [self._get_random_date() for _ in range(num_products)],
#             'dto': [self._get_random_date() for _ in range(num_products)]
#         })
#         return products_df.to_dict(orient='records')
#
#     @staticmethod
#     def _get_template_based_data(template) -> tuple[int, bool]:
#         min_products = 1
#         max_products = 7
#         show_currency = True
#
#         match template:
#             case 'template_03.docx':
#                 show_currency = False
#             case 'template_05.docx':
#                 min_products = 15
#                 max_products = 20
#                 show_currency = False
#             case 'template_06.docx':
#                 max_products = 4
#
#         num_products = random.randint(min_products, max_products)
#         return num_products, show_currency
#
#     def _get_random_product_name(self) -> str:
#         product_name = self.fake.company() + ' ' + ' '.join(self.fake.words(nb=2))
#         return product_name
#
#     def _get_random_price(self, show_currency: bool=True, currency_position: bool = None, max_price: int = 9999999) -> str:
#         whole = random.randint(0, max_price)
#         decimals = random.randint(0, 99)
#         sign = random.choice(['', '', '', '', '-'])  # Gives more probability to positive values
#         price = '0,00'
#         match self.price_format:
#             case 0:  #   1234.56     |  1234567.78      | 1234567
#                 price = f'{sign}{whole}.{decimals}'
#             case 1:  #   1234,56     |  1234567,78      | 1234567
#                 price = f'{sign}{whole},{decimals}'
#             case 2:  #   1,234.56    |  1,234,567.78    | 1,234,567
#                 whole_str = f'{whole:,}'
#                 price = f'{sign}{whole_str}.{decimals}'
#             case 3:  #   1.234,56    |  1.234.567,78    | 1.234.567
#                 whole_str = f'{whole:,}'.replace(',', '.')
#                 price = f'{sign}{whole_str},{decimals}'
#             case 4:  #   1 234.56    |  1 234 567.78    | 1 234 567
#                 whole_str = f'{whole:,}'.replace(',', ' ')
#                 price = f'{sign}{whole_str},{decimals}'
#             case 5:
#                 whole_str = f'{whole:,}'.replace(',', ' ')
#                 price = f'{sign}{whole_str}.{decimals}'
#
#         if show_currency:
#             price = f'{self.currency} {price}' if currency_position else f'{price} {self.currency}'
#
#         return price
#
#     def _get_random_additional_info(self):
#         return {
#             'random_info_01': self._get_random_text(40),
#             'rand_number_01': self._get_random_price()
#         }
#
#     def _get_footer_info(self, doc, template_name):
#         return {
#             'footer_info_0': InlineImage(doc, image_descriptor=self._get_random_img('cert_logos'), width=Mm(10), height=Mm(10)),
#             'footer_info_1': self._get_short_lines(font_size=12),
#         }
#
#     def _init_output_folder(self):
#         if not os.path.exists(self.output_path):
#             os.makedirs(self.output_path)
#         if not os.path.exists(self.renders_path):
#             os.makedirs(self.renders_path)
#         if not os.path.exists(self.pdfs_path):
#             os.makedirs(self.pdfs_path)
#         if not os.path.exists(self.results_path):
#             os.makedirs(self.results_path)
#
#     def save_sample(self, docx_file: Path, gt_obj: dict, template_name: str, dpi: int = 200):
#         pdf_path = self._to_pdf(docx_file)
#         im_pixmap = self._to_img(pdf_path, dpi)
#
#         filename = docx_file.stem
#
#         sample_path = self.results_path / filename
#         if not os.path.exists(sample_path):
#             os.makedirs(sample_path)
#
#             # Save image
#             im_pixmap.save(sample_path / (filename + '.png'))
#
#             # Save ground-truth JSON
#             with open(sample_path / (filename + '.json'), 'w', encoding='utf-8') as fp:
#                 json.dump(gt_obj, fp, indent=4, ensure_ascii=False)
#
#             # Save metadata file with template name
#             with open(sample_path / '.file_metadata.json', 'w') as fp:
#                 json.dump({'template_name': template_name}, fp)
#
#     def _to_pdf(self, docx_file: Path) -> Path:
#         pdf_path = self.pdfs_path / (docx_file.stem + '.pdf')
#         docx2pdf.convert(docx_file, pdf_path)
#         return pdf_path
#
#     def _to_img(self, pdf_file: Path, dpi: int = 200):
#         pdf_obj = pymupdf.open(pdf_file)
#         pixmap = pdf_obj[0].get_pixmap(dpi=dpi)
#         return pixmap
#
#     def _enrich_data(self, data: dict) -> None:
#         data['vendor'] = RichText(data['vendor'], font=self.font['family'])
#         data['id'] = RichText(data['id'], font=self.font['family'])
#         data['po'] = RichText(data['po'], font=self.font['family'])
#
#     @staticmethod
#     def _read_data_file(file_path: Path) -> list[str]:
#         with open(file_path, 'r', encoding='utf-8') as fp:
#             lines = fp.read().splitlines()
#         return lines
#
#
# if __name__ == '__main__':
#     synthetic_data_generator = SyntheticDataGenerator(
#         templates_path = Path('synthetic_data/templates'),
#         output_path = Path('synthetic_data/dataset'),
#         initial_sample_idx=0
#     )
#     synthetic_data_generator.generate(samples_per_template=50)
#     synthetic_data_generator.cleanup()