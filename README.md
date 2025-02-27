# Ereading the unreadable
This repository is to convert scanned images of 19th century newspapers into an indexed and classified data collection. The results of this project are discussed in the paper "Reading the unreadable: Creating a dataset of 19th century English newspapers using image-to-text language models".

The project is an evolution of [CLOCR-C](https://github.com/JonnoB/clocrc), [ScrambledText](https://github.com/JonnoB/scrambledtext_analysis), and the hackathon project [Archivstral](https://github.com/JonnoB/archivestal). It is an attempt to re-frame OCR as an Language Model tast, as well as provide, a substantial new searchable data collection of 19th century English Newspapers

## Abstract

Oscar Wilde said, "The difference between literature and journalism is that journalism is unreadable, and literature is not read." Unfortunately, The digitally archived journalism of Oscar Wilde's 19th century often has no or poor quality Optical Character Recognition (OCR), reducing the accessibility of these archives and making them unreadable both figuratively and literally. This paper helps address the issue by performing OCR on "The Nineteenth Century Serials Edition" (NCSE), an 84k-page collection of 19th-century English newspapers and periodicals, using Pixtral 12B, a pre-trained image-to-text language model. The OCR capability of Pixtral was compared to 4 other OCR approaches, achieving a median character error rate of 1%, 5x lower than the next best model. The resulting NCSE v2.0 dataset features improved article identification, high-quality OCR, and text classified into four types and seventeen topics. The dataset contains 1.4 million entries, and 321 million words. Example use cases demonstrate analysis of topic similarity, readability, and event tracking. NCSE v2.0 is freely available to encourage historical and sociological research. As a result, 21st-century readers can now share Oscar Wilde's disappointment with 19th-century journalistic standards, reading the unreadable from the comfort of their own computers. 

# Using this Code Repository

This repository can be installed in two configurations:
1. Base installation (CPU-only) - for basic functionality
2. Full installation (with GPU support) - for all features including GPU-accelerated processing

## Installation

It is recommended to use the Astral UV library for package management.

### Base Installation (CPU-only)
```bash
uv pip install -r requirements.txt
uv pip install -e .
```

### Full Installation (with GPU support)
```bash
uv pip install -r requirements_gpu.txt
uv pip install -e .
```

If you are not using UV, simply drop the 'uv' part of the command and install using pip:
```bash
pip install -r requirements.txt  # for base installation
# or
pip install -r requirements_gpu.txt  # for full installation with GPU support
pip install -e .
```

## Environment Setup

If you are using the scripts which work with the original NCSE images, ensure the path to the folders is in your `.env` file.

# Project Pipeline

- Convert NCSE files to single page PNG
- Predict bounding boxes using DocLayout-YOLO
- Post-process bounding boxes
- Batch Process Data
- Construct Text Pieces
- Classify Text

Each element of the project pipeline has a python script for execution. However, there is no single end-to-end script. 

# Folders

- function_modules: Project python library
- project_scripts: The main scripts for re-producing the project, see folder for separate README
- lightning_code: scripts and ipynb code requiring GPU and run on the lightning platform. See folder for separate README

# Key files
- classification_and_data_analysis.py: used to create the silver datasets for training the ModernBERT classifiers. Also analyses the performance of the models. Model training scripts can be found in the lightning_code folder.
- result_section.py: The code used to generate the most of results section of the paper. Produces the tables plots etc.
- send_to_pixel_streaming.py: An example of realtime sending to the Pixtral server.
- comparative_analysis.py: Example use cases for comparative analysis of the periodical data in NCSE V2.0

The scripts and ipynb to measure the performance of the alternative models can be found in the "alternative_models" folder

# To Do
- Create lightning example using non-batch and batch pipeline

# Note
- The code is written in a mixture of marimo and regular .py files. This is because although marimo is preferred for it's ease of reporducibility, the GPU acitivity was performed on lightning.ai which is focused on .ipynb based code or simple .py scripts.

# Database

The database produced by this project is available at [NCSE v2.0: A Dataset of OCR-Processed 19th Century English Newspapers](https://rdr.ucl.ac.uk/articles/dataset/NCSE_v2_0_A_Dataset_of_OCR-Processed_19th_Century_English_Newspapers/28381610) held in the UCL data repository.


# Citing this project

If you use this project or the code used to generate it, please cite
[Reading the unreadable: Creating a dataset of 19th century English newspapers using image-to-text language models](https://arxiv.org/abs/2502.14901)


_______No citation information available yet_________