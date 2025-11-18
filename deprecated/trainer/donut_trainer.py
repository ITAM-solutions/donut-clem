"""
File name: donut_trainer
Author: Fran Moreno
Last Updated: 6/27/2025
Version: 1.0
Description: TOFILL
"""
import configparser
import json
import numpy as np
import logging
import pytesseract
import pytorch_lightning as pl
import os
import re
import torch
import xml.etree.ElementTree as ET
import wandb

from collections import defaultdict
from datetime import datetime
from huggingface_hub import login, HfApi, hf_hub_download
from nltk import edit_distance
from pathlib import Path, PureWindowsPath
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig


class TrainerHandler:
    def __init__(self, config_file: Path = 'config.ini'):
        self.config_file = config_file

        self.config = self._read_config_file(self.config_file)
        self.logger = self._get_logger()

        self.logger.info("Logger initialized!")

        pytesseract.pytesseract.tesseract_cmd = self.config['TOOLS']['tesseract_path']

        self._login_to_hf(self.config['TOOLS']['hf_token'])
        self._login_to_wandb(self.config['TOOLS']['wandb_token'])
        self.logger.info("Logged into HF and WanDB services.")

        self.batch_size = self._cast_config(self.config['GLOBAL']['batch_size'], 'int')
        self.params = self._prepare_training_params(self.config['TRAINING'])

        self.model_name = "naver-clova-ix/donut-base"
        self.hf_repo = self.config['TOOLS']['hf_repo']

        self.start_token = "<s_cord-v2>"
        self.end_token = "</s_cord-v2>"

        self.image_size = [
            self._cast_config(self.config['MODEL']['image_size_h'], 'int'),
            self._cast_config(self.config['MODEL']['image_size_w'], 'int')
        ]
        self.max_length = self._cast_config(self.config['MODEL']['max_output_length'], "int")

        self.processor = DonutProcessor.from_pretrained(self.model_name, use_fast=True)  # use_fast allows optimization
        self.processor.image_processor.do_align_long_axis = False
        self.processor.image_processor.size = {"height": self.image_size[0], "width": self.image_size[1]}

        # Defining Model
        self.model_config = VisionEncoderDecoderConfig.from_pretrained(self.model_name)
        self.model_config.encoder.image_size = self.image_size
        self.model_config.decoder.max_length = self.max_length
        self.model_config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model_config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.start_token)

        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name, config=self.model_config)
        self.model.train()
        self.logger.info("Processor and model defined!")

        self.dataset_path = self._cast_config(self.config['GLOBAL']['dataset_path'], 'path')

        self.logger.info("Starting to read dataset...")
        excluded_keys = self._cast_config(self.config['MODEL']['keys_to_exclude'], 'list')

        self.training_dataset = DonutDataset(self.dataset_path, self.processor, self.model_config, split='train', start_token=self.start_token,
            end_token=self.end_token, excluded_keys=excluded_keys)
        self.validation_dataset = DonutDataset(self.dataset_path, self.processor, self.model_config, split='val', start_token=self.start_token,
            end_token=self.end_token, excluded_keys=excluded_keys)
        self.test_dataset = DonutDataset(self.dataset_path, self.processor, self.model_config, split='test', start_token=self.start_token,
            end_token=self.end_token, excluded_keys=excluded_keys)
        self.logger.info("Finished reading the dataset.")

        # Gather all new tokens
        self.all_new_tokens = set()
        self.all_new_tokens.update(self.training_dataset.new_special_tokens)
        self.all_new_tokens.update(self.validation_dataset.new_special_tokens)
        self.all_new_tokens.update(self.test_dataset.new_special_tokens)
        self.all_new_tokens_list = list(self.all_new_tokens)

        # Add new tokens to Donut processor and model.
        newly_added_num = self.processor.tokenizer.add_tokens(self.all_new_tokens_list)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))

        # Allow special tokens you use in output (e.g. task start/end), block others
        bad_words = [id for tok, id in self.processor.tokenizer.get_vocab().items() if
            tok in self.processor.tokenizer.all_special_tokens and tok not in [self.start_token, self.end_token]]
        bad_words_ids = [[token_id] for token_id in bad_words]

        # Show final split percentages.
        total_samples = len(self.training_dataset) + len(self.validation_dataset) + len(self.test_dataset)

        """ DATA LOADERS """
        self.train_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.val_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        self.logger.info("DataLoaders created.")

        """ CALLBACKS """
        self.wandb_logger = WandbLogger(project="Donut", name="synth_data_v2")

        self.model_checkpoint_dirpath = 'checkpoints'
        self.push_to_hub_callback = PushToHubCallback(
            self.logger,
            self.hf_repo,
            self.model_checkpoint_dirpath
        )

        self.model_checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dirpath,
            filename='{epoch}_{step}',
            save_top_k=1,
            every_n_epochs=1,

            save_weights_only=False
        )

        """ LIGHTNING MODULE CREATION """
        self.model_module = DonutModelPLModule(
            self.params,
            self.processor,
            self.model,
            self.batch_size,
            self.max_length,
            self.train_dataloader,
            self.val_dataloader
        )

        """ LIGHTNING TRAINER """
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=self.params.get("max_epochs"),
            val_check_interval=self.params.get("val_check_interval"),
            check_val_every_n_epoch=self.params.get("check_val_every_n_epoch"),
            gradient_clip_val=self.params.get("gradient_clip_val"),
            precision=self.params.get("precision"),
            num_sanity_val_steps=0,
            logger=self.wandb_logger,
            callbacks=[self.push_to_hub_callback, self.model_checkpoint_callback],
            log_every_n_steps=self.params.get('log_every_n_steps')
        )

    def train(self):
        self.logger.info('Correctly defined the trainer module.')
        try:
            ckpt_path = self._load_last_weights()
            self.trainer.fit(
                self.model_module,
                ckpt_path=ckpt_path,
            )
        except Exception as e:
            self.logger.error(e)
        finally:
            wandb.finish()
            torch.cuda.empty_cache()

    def _load_last_weights(self):
        api = HfApi()

        files = api.list_repo_files(
            repo_id=self.hf_repo,
            repo_type="model",
        )

        ckpt_files = [f for f in files if f.endswith(".ckpt")]
        if not ckpt_files:
            return None

        ckpt_filename = ckpt_files[0]

        ckpt_path = hf_hub_download(
            repo_id=self.hf_repo,
            filename=ckpt_filename,
            repo_type="model"
        )
        return ckpt_path

    @staticmethod
    def _login_to_hf(token):
        login(token)

    @staticmethod
    def _login_to_wandb(token):
        os.environ['WANDB_API_KEY'] = token

    @staticmethod
    def _read_config_file(file_path: Path = Path('config.ini')) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError('Could not find configuration file.')

        config = configparser.ConfigParser()
        config.read('config.ini')

        return {s:dict(config.items(s)) for s in config.sections()}

    @staticmethod
    def _cast_config(value, to: str):
        try:
            match to:
                case 'int':
                    return int(value)
                case 'float':
                    return float(value)
                case 'list':
                    return json.loads(value)
                case 'path':
                    return Path(value)
                case _:
                    return value
        except (TypeError, ValueError, json.decoder.JSONDecodeError):
            return value

    def _prepare_training_params(self, params: dict) -> dict:
        casts = {
            'max_epochs': 'int',
            'val_check_interval': 'float',
            'check_val_every_n_epoch': 'int',
            'gradient_clip_val': 'float',
            'lr': 'float',
            'num_nodes': 'int',
            'precision': 'string',
            'log_every_n_steps': 'int'
        }

        return {
            name: self._cast_config(params[name], type_) for name, type_ in casts.items()
        }

    @staticmethod
    def _get_logger(logs_path: Path = Path('log')):
        os.makedirs(logs_path, exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        log_file = logs_path / datetime.now().strftime("%Y%m%d_%H-%M-%S_log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        return logger


class DonutDataset(Dataset):
    def __init__(self,
            dataset_path: Path,
            processor,
            model_config,
            split: str = 'train',
            start_token: str = "<s>",
            end_token: str = "</s>",
            excluded_keys: tuple = tuple()
    ):

        self.dataset_path = dataset_path
        self.split = split
        self.start_token = start_token
        self.end_token = end_token
        self.excluded_keys = excluded_keys
        self.empty_token = '<s_no_value>'
        self.ignore_id = -100
        self.processor = processor
        self.model_config = model_config
        self.new_special_tokens = set()  # Set that will collect all new special tokens found in the given data.

        self.train_set = []
        self.val_set = []
        self.test_set = []

        # Loads data from the given dataset path
        self._get_synthetic_samples_list()
        self.raw_samples = self._load_dataset()

        # Add start and end tokens as special tokens.
        self.new_special_tokens.add(self.start_token)
        self.new_special_tokens.add(self.end_token)

    @staticmethod
    def _split_train_validation_test(dataset: list):
        """ Splits the data into training, validation and testing sets. """

        x_train_tmp, x_test = train_test_split(dataset, test_size=0.1)
        x_train, x_val = train_test_split(x_train_tmp, test_size=0.135)
        return x_train, x_val, x_test

    def _load_dataset(self) -> dict:
        """ Selects a split from the dataset and loads its images and json contents as 'raw samples'. """

        samples_paths = self.train_set if self.split == 'train' \
            else self.val_set if self.split in ['validation', 'val'] \
            else self.test_set

        raw_samples = {'ims': [], 'ids': [], 'json': []}

        for raw_sample in tqdm(samples_paths):

            im_path, json_path = raw_sample.with_suffix('.png'), raw_sample.with_suffix('.json')

            # Read image in RGB format.
            im_raw = Image.open(im_path).convert('RGB')

            # Read attached JSON content.
            with open(json_path, 'r', encoding='utf-8') as fp:
                json_content = json.load(fp)

            raw_samples['json'].append(json_content)

            json_content = self._clean_excluded_keys(json_content)
            json_content = self._normalize_empty_values(json_content)

            json_tokens = self.json2token(json_content)

            raw_samples['ims'].append(im_raw)
            raw_samples['ids'].append(json_tokens)

        return raw_samples


    def _clean_excluded_keys(self, obj: dict):
        for k in self.excluded_keys:
            if k in obj.keys():
                obj.pop(k)

        if obj.get('products'):
            for k in self.excluded_keys:
                if k in obj['products']:
                    obj['products'].pop(k)

        return obj

    def _normalize_empty_values(self, obj: dict):
        # Document-level fields
        obj = {k: v if v else self.empty_token for k, v in obj.items()}

        if obj.get('products'):
            for product_key in obj['products']:
                obj['products'][product_key] = [v if v else self.empty_token for v in obj['products'][product_key]]

        return obj

    def __len__(self):
        """
        Retrieves the number of samples in the current dataset split.
        **Method required by PyTorch DataLoaders to work**
        """

        dataset_length = len(self.train_set) if self.split == 'train' \
            else len(self.val_set) if self.split in ['validation', 'val'] \
            else len(self.test_set)
        return dataset_length

    def __getitem__(self, idx) -> dict:
        """
        Retrieves the sample in position `idx` from the dataset.
        **Method required by PyTorch DataLoaders to work**
        """

        im_raw, json_tokens = self.raw_samples['ims'][idx], self.raw_samples['ids'][idx]

        im = self._normalize_image(im_raw)
        gt_tokens, target_str_sequence = self._normalize_tagging(json_tokens)

        return {
            'pixel_values': im,
            'labels': gt_tokens,
            'target_str_seq': target_str_sequence,
        }

    def _get_samples_list(self):
        """
        Finds all the samples in the dataset and splits them into training, validation and test sets
        based on providers. The split percentages are applied to each provider, and not directly to
        the full dataset, ensuring that if a provider has enough samples, they will exist in every data
        split, making validation more reasonable.
        """

        samples_by_provider = defaultdict(list)
        metadata_files = self.dataset_path.glob('**/*.file_metadata.json')

        # Iterate over sample folders to determine original providers.
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as fp:
                json_content = json.load(fp)

            original_file = PureWindowsPath(json_content['origin']['name'])  # Paths saved in a Windows machine.
            provider = Path(*original_file.parts[:original_file.parts.index('datasets') + 3]).name

            # Obtain the list of images for the current sample document.
            samples_from_document = list(i.with_suffix('') for i in metadata_file.parent.glob('*.png'))
            samples_by_provider[provider].extend(samples_from_document)

        # Split providers into training, validation and test sets.
        for provider, files in samples_by_provider.items():
            if len(files) == 1:
                self.train_set.extend(files)
            elif len(files) == 2:
                self.train_set.append(files[0])
                self.val_set.append(files[1])
            else:
                train_set, val_set, test_set = self._split_train_validation_test(files)
                self.train_set.extend(train_set)
                self.val_set.extend(val_set)
                self.test_set.extend(test_set)

    def _get_synthetic_samples_list(self):
        """
        Finds all the samples in a dataset with synthetic data and splits them into training,
        validation and test sets based on the synthetic template used to create it. The split
        percentages are applied to each template case, and not directly to the full dataset.
        """

        samples_by_template = defaultdict(list)
        metadata_files = self.dataset_path.glob('**/*.file_metadata.json')

        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as fp:
                json_content = json.load(fp)

            template_name = json_content['template_name']
            samples_by_template[template_name].append(metadata_file.parent / metadata_file.parent.name)

        for template_name, files in samples_by_template.items():
            if len(files) == 1:
                self.train_set.extend(files)
            elif len(files) == 2:
                self.train_set.append(files[0])
                self.val_set.append(files[1])
            else:
                train_set, val_set, test_set = self._split_train_validation_test(files)
                self.train_set.extend(train_set)
                self.val_set.extend(val_set)
                self.test_set.extend(test_set)

    def _normalize_image(self, im: Image) -> torch.Tensor:
        """ Normalizes the given image to the expected tensor format for Donut. """

        im = self._fix_image_orientation(im)

        # First dimension is removed, as it refers to the batch. Batches are automatically created by DataLoader.
        im_tensor = self.processor(im, return_tensors='pt').pixel_values.squeeze()
        return im_tensor

    @staticmethod
    def _fix_image_orientation(image: Image) -> Image:
        """
        Uses pytesseract to get some image metadata, then get the page orientation, and finally rotate the image
        when needed. If pytesseract cannot get the orientation, then the image is returned with no changes.

        :return:
        """
        try:
            im_metadata = pytesseract.image_to_osd(image)
            angle = 360 - int(re.search(r'(?<=Rotate: )\d+', im_metadata).group(0))
            return image.rotate(angle, expand=1)
        except Exception:
            # Tesseract failed. Return image in its original format.
            return image

    def _normalize_tagging(self, json_tokens) -> tuple[torch.Tensor, str]:
        """ Converts the JSON content to tokens that can be fed to Donut. """
        target_sequence = json_tokens + self.processor.tokenizer.eos_token

        # Remove first dimension from tensor, as it refers to the batch. Batches are automatically created by DataLoader.
        tokenizer_response = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.model_config.decoder.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )['input_ids'].squeeze(0)

        labels = tokenizer_response.clone()

        # Replace all pad tokens by the ignore token.
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id

        return labels.squeeze(0), target_sequence

    @staticmethod
    def recover_from_tensor(t: torch.Tensor) -> Image:
        """ Utility method to get back the image from a normalized tensor. Use only for debugging. """

        arr = t.cpu().numpy()

        # Donut applies mean and std transformations to image values. Reverting.
        mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        arr = (arr * std + mean) * 255.0

        # Limit values to uint8 range, which is the expected for images (0 to 255).
        arr = arr.clip(0, 255).astype(np.uint8)

        # [layers, width, height] -> [width, height, layers] (expected format for PIL images).
        arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr)

    def json2token(self, obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence.
        Source: https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/donut/model.py#L499
        """

        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        # Add to the list of special tokens detected.
                        self.new_special_tokens.add(fr"<s_{k}>")
                        self.new_special_tokens.add(fr"</s_{k}>")
                    output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            # excluded special tokens for now
            obj = str(obj)
            if f"<{obj}/>" in self.new_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model, batch_size, max_length, train_dataloader_obj, val_dataloader_obj):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_dataloader_obj = train_dataloader_obj
        self.val_dataloader_obj = val_dataloader_obj

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, dataset_idx=0):
        pixel_values = batch['pixel_values']
        answers = [a.strip().lower() for a in batch['target_str_seq']]

        batch_size = pixel_values.size(0)

        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            device=self.device
        )

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True
        )

        decoded = self.processor.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        predictions = [d.strip().lower() for d in decoded]

        # Metrics
        metrics = self._compute_metrics(predictions, answers)

        print("Prediction vs Answer (first sample in batch):")
        print("\n".join([f"\t{k}: {v}" for k, v in metrics.items()]))
        print(f"    Prediction: {predictions[0]}")
        print(f"    Reference : {answers[0]}")

        self.validation_step_outputs.append(metrics)

        return {
            "predictions": predictions,
            "references": answers,
            **metrics
        }

    def _compute_metrics(self, predictions, answers) -> dict:
        num_degenerated_samples = 0
        matching_keys_ratios = list()
        edit_distance_values = list()

        for prediction, answer in zip(predictions, answers):
            try:
                tree_pred = self._etree_to_dict(ET.fromstring(prediction))
                tree_ans = self._etree_to_dict(ET.fromstring(answer))

                matching_keys_ratio, edit_distance = self._find_matching_keys_and_values(tree_pred, tree_ans)
                matching_keys_ratios.append(matching_keys_ratio)
                edit_distance_values.append(edit_distance)
            except ET.ParseError:
                num_degenerated_samples += 1

        avg_matching_keys_ratio = sum(matching_keys_ratios) / len(matching_keys_ratios) if len(matching_keys_ratios) > 0 else 0.0
        avg_edit_distance = sum(edit_distance_values) / len(edit_distance_values) if len(edit_distance_values) > 0 else 0.0
        return {
            "degeneration_ratio": num_degenerated_samples / len(predictions),
            "extraction_similarity_ratio": avg_edit_distance,
            "matching_keys_ratio": avg_matching_keys_ratio
        }

    def _etree_to_dict(self, t) -> dict:
        """
        Source: https://stackoverflow.com/questions/7684333/converting-xml-to-dictionary-using-elementtree
        """
        d = {t.tag: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(self._etree_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {t.tag: {k: v[0] if len(v) == 1 else v
                for k, v in dd.items()}}
        if t.attrib:
            d[t.tag].update(('@' + k, v)
                for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[t.tag]['#text'] = text
            else:
                d[t.tag] = text
        return d

    def _find_matching_keys_and_values(self, pred: dict, ans: dict):
        if 's_cord-v2' in ans:
            ans = ans['s_cord-v2']

        matching_keys = 0
        edit_distance_ratios = []
        total_keys = 0
        # Shared keys
        for expected_key in ans:
            if expected_key in pred:
                matching_keys += 1
                edit_distance_ratios.append(self._compute_similarity(pred[expected_key], ans[expected_key]))
            total_keys += 1

        # Product keys
        if isinstance(ans.get('s_products'), list) and  isinstance(pred.get('s_products'), list):
            for pred_product, ans_product in zip(pred['s_products'], ans['s_products']):
                for product_key in ans_product:
                    if product_key in pred_product:
                        matching_keys += 1
                        edit_distance_ratios.append(self._compute_similarity(pred_product[product_key], ans_product[product_key]))
                    total_keys += 1

        avg_edit_distance_ratio = sum(edit_distance_ratios) / len(edit_distance_ratios)
        matching_keys_ratio = matching_keys / total_keys

        return matching_keys_ratio, avg_edit_distance_ratio

    @staticmethod
    def _compute_similarity(pred_value, gt_value):
        return 1.0 - (edit_distance(pred_value, gt_value) / max(len(pred_value), len(gt_value)))

    def on_validation_epoch_end(self) -> None:
        all_degeneration_ratios = []
        all_similarity_ratios = []
        all_matching_keys_ratios = []

        for track in self.validation_step_outputs:
            all_degeneration_ratios.append(track['degeneration_ratio'])
            all_similarity_ratios.append(track['extraction_similarity_ratio'])
            all_matching_keys_ratios.append(track['matching_keys_ratio'])

        avg_degeneration_ratio = float(np.mean(all_degeneration_ratios))
        avg_similarity_ratio = float(np.mean(all_similarity_ratios))
        avg_matching_keys_ratios = float(np.mean(all_matching_keys_ratios))

        self.log("degeneration_ratio", avg_degeneration_ratio, sync_dist=True)
        self.log("similarity_ratio", avg_similarity_ratio, sync_dist=True)
        self.log("matching_keys_ratios", avg_matching_keys_ratios, sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # TODO: Add a learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return self.train_dataloader_obj

    def val_dataloader(self):
        return self.val_dataloader_obj


class PushToHubCallback(Callback):
    def __init__(self, logger, hf_repo, ckpt_dirpath):
        super().__init__()
        self.logger = logger
        self.hf_repo = hf_repo
        self.ckpt_dirpath = ckpt_dirpath
        self.api = HfApi()

    def _push_checkpoint_file(self):
        if os.path.exists(self.ckpt_dirpath):
            ckpt_name = self._get_last_checkpoint(self.ckpt_dirpath)
            ckpt_path = os.path.join(self.ckpt_dirpath, ckpt_name)
            print("Checkpoint selected:", ckpt_path)
            self._remove_old_checkpoints_from_hub()
            self.api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=os.path.join('checkpoints', ckpt_name),
                repo_id=self.hf_repo,
                repo_type='model',
                commit_message=f"Checkpoint saved: {ckpt_name}"
            )
        else:
            self.logger.warning("No checkpoints available.")

    def _remove_old_checkpoints_from_hub(self):
        all_files = self.api.list_repo_files(repo_id=self.hf_repo, repo_type='model')
        ckpt_files = [f for f in all_files if 'checkpoints/epoch' in f]

        for ckpt_file in ckpt_files:
            print("Removing old checkpoint:", ckpt_file)

            self.api.delete_file(
                path_in_repo=ckpt_file,
                repo_id=self.hf_repo,
                repo_type='model',
                commit_message=f'Removed old checkpoint "{ckpt_file}"',
            )

    def _get_last_checkpoint(self, ckpt_dir: str):
        ckpts = os.listdir(ckpt_dir)
        print("Checkpoints saved:", ckpts)
        if not len(ckpts):
            return None
        elif len(ckpts) == 1:
            return ckpts[0]
        else:
            epoch_step_pairs = []
            for ckpt in ckpts:
                matches = re.match(r"epoch=(\d*)_step=(\d*).ckpt", ckpt)
                if matches:
                    epoch, step = map(lambda x: int(x), matches.groups())
                    epoch_step_pairs.append([epoch, step])
            best_ckpt_values = self._find_highest_list(epoch_step_pairs)
            best_ckpt_idx = epoch_step_pairs.index(best_ckpt_values)
            return ckpts[best_ckpt_idx]

    def on_train_epoch_end(self, trainer, pl_module):
        self.logger.info(f"Pushing model to the Hub [epoch {trainer.current_epoch}]")
        self._push_checkpoint_file()

    def on_train_end(self, trainer, pl_module):
        self.logger.info(f"Pushing model to the hub after training.")
        pl_module.processor.push_to_hub(self.hf_repo, commit_message="Training Finished!")
        pl_module.model.push_to_hub(self.hf_repo, commit_message="Training Finished!")

    def _find_highest_list(self, lists: list, curr_idx=0):
        if len(lists) == 0:
            return []

        lists_lengths = [len(i) for i in lists]

        if not all(x == lists_lengths[0] for x in lists_lengths):
            raise ValueError("All lists must have the same length")

        length = lists_lengths[0]

        lists_idxs_with_max_value = []
        curr_max = -1

        for i, l in enumerate(lists):
            if l[curr_idx] > curr_max:
                curr_max = l[curr_idx]
                lists_idxs_with_max_value.clear()
                lists_idxs_with_max_value.append(i)
            elif l[curr_idx] == curr_max:
                lists_idxs_with_max_value.append(i)

        if len(lists_idxs_with_max_value) == 1:
            return lists[lists_idxs_with_max_value[0]]
        else:
            curr_idx += 1
            if curr_idx > length - 1:
                return lists[lists_idxs_with_max_value[0]]

            return self._find_highest_list([lists[j] for j in lists_idxs_with_max_value], curr_idx)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    trainer_handler = TrainerHandler(Path('config.ini'))
    trainer_handler.train()
