# """
# File name: azure_word_collector
# Author: Fran Moreno
# Last Updated: 8/28/2025
# Version: 1.0
# Description: TOFILL
# """
# import os
# from nltk.corpus import words
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.core.credentials import AzureKeyCredential
# from backend.helpers.encryption import load_dotenv_encrypted
#
# from tqdm import tqdm
#
#
# def field_collector():
#     endpoint = "https://clem-doc-ai.cognitiveservices.azure.com"
#     secrets = load_dotenv_encrypted()
#     key = secrets.get("AZURE_KEY")
#     model_id = "prebuilt-invoice"
#
#     folder_path = "../synthetic_data/dataset/generic_layout_samples"
#     output_path = "../synthetic_generator/data/corpus.txt"
#
#     interesting_fields = {
#         'VendorAddressRecipient',
#         'CustomerAddressRecipient',
#         'BillingAddressRecipient',
#         'ShippingAddressRecipient',
#         'RemittanceAddressRecipient',
#         'ServiceAddressRecipient',
#         'PaymentTerm',
#     }
#     interesting_product_fields = {
#         'Unit'
#     }
#
#     files = os.listdir(folder_path)
#     for f in tqdm(files, total=len(files)):
#         data_collector = dict()
#         filepath = os.path.join(folder_path, f)
#
#         with DocumentIntelligenceClient(endpoint, AzureKeyCredential(key)) as az_client:
#             with open(filepath, "rb") as file:
#                 analysis_operation = az_client.begin_analyze_document(
#                     model_id=model_id,
#                     body=file,
#                     content_type='application/octet-stream',
#                 )
#                 document_analysis = analysis_operation.result()
#
#                 document = document_analysis.documents[0]
#
#                 fields_info = document.fields
#
#                 for field_name, field_data in fields_info.items():
#                     if field_name == 'Items':
#                         items_info = fields_info[field_name].value_array
#                         for item_info in items_info:
#                             for item_field, item_data in item_info.value_object.items():
#                                 if item_field not in interesting_product_fields:
#                                     continue
#
#                                 text = item_data.content.replace('\n', '')
#                                 if item_field in data_collector:
#                                     data_collector[item_field].append(text)
#                                 else:
#                                     data_collector[item_field] = [text]
#                     else:
#                         if field_name not in interesting_fields:
#                             continue
#
#                         text = field_data.content.replace('\n', ' ')
#                         if field_name in data_collector:
#                             data_collector[field_name].append(text)
#                         else:
#                             data_collector[field_name] = [text]
#
#         for k, v in data_collector.items():
#             with open(os.path.join(output_path, f'{k}.txt'), 'a+') as f:
#                 f.writelines([line + '\n' for line in v])
#
#
# def text_collector():
#     endpoint = "https://clem-doc-ai.cognitiveservices.azure.com"
#     secrets = load_dotenv_encrypted()
#     key = secrets.get("AZURE_KEY")
#     model_id = "prebuilt-invoice"
#
#     folder_path = "../synthetic_data/dataset/generic_layout_samples"
#     output_path = "../synthetic_generator/data/corpus2.txt"
#
#     eng_corp = set(words.words())
#
#     corpus = set()
#     files = os.listdir(folder_path)
#     for f in tqdm(files, total=len(files)):
#         filepath = os.path.join(folder_path, f)
#
#         with DocumentIntelligenceClient(endpoint, AzureKeyCredential(key)) as az_client:
#             with open(filepath, "rb") as file:
#                 analysis_operation = az_client.begin_analyze_document(
#                     model_id=model_id,
#                     body=file,
#                     content_type='application/octet-stream',
#                 )
#                 document_analysis = analysis_operation.result()
#                 document_text = document_analysis.content
#
#                 document_words = document_text.replace('\n', ' ')
#                 document_words_norm = [word for i in document_words.split(' ') if (word := norm_word(i, eng_corp))]
#
#                 with open(output_path, 'a') as f:
#                     f.write('\n'.join(document_words_norm))
#
#
#     # Remove duplicates
#     with open(output_path, 'r') as f:
#         unique_words = set([w.replace('\n', '') for w in f.readlines()])
#
#     with open(output_path, 'w') as f:
#         f.write('\n'.join(unique_words))
#
#
# def norm_word(word: str, eng_corp: set):
#     word_norm = ''.join(c for c in word if c.isalnum()).lower()
#     return word_norm if word_norm in eng_corp else None
#
#
# if __name__ == '__main__':
#     # field_collector()
#     text_collector()
#
