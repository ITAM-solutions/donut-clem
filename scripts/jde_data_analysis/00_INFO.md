# JDE Data Analysis scripts

### Creation Date
May 2025

### Goal
Analyze the JDE invoices & documents dataset and prepare it to
be used for training a Machine Learning model for Document Data Extraction.

### Useful info

The JDE Dataset is composed by a set of folders related to different companies/publishers.
Each folder contains two subfolders: 'ITAM' and 'JDE'. And within those subfolders,
we can find excel files and pdf/docx documents.

Excel files are extractions from ITAM Solutions consultants from the original PDF documents.
PDF Documents are invoices/contract/entitlements from the named publishers.

## Scripts

- `01_excel_analysis.py`: used to find out which PDF documents can actually be used to train 
Donut, by checking the excel data extractions and collecting its info for the different
fields that we would like to extract using Donut.
- `02_json_tags_analysis.py`: To run after `excel_analysis.py`. Collects all the unique fields
found and normalizes to JSON format.
- `03_analyze_images_size.py`: plots some graphics that help understand the image size variability
in the dataset. Images are extracted from the original PDF files.
- `04_samples_selector.py`: Use over a set of images to do a selection of them using a visualization window to help
deciding which samples to save. Used to extract samples to be used as a guide for the synthetic templates.

