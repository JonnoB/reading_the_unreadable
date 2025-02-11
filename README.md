# This project is currently underdevelopment... probably best not to use

# Ereading the unreadable
This repository is to convert scanned images of 19th century newspapers into an indexed and classified data collection.

This project is an evolution of [CLOCR-C](https://github.com/JonnoB/clocrc), [ScrambledText](https://github.com/JonnoB/scrambledtext_analysis), and the hackathon project [Archivstral](https://github.com/JonnoB/archivestal). It is an attempt to re-frame OCR as an Language Model tast, as well as provide, a substantial new searchable data collection of 19th century English Newspapers

## Abstract


xxxxxxxxxx

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
- Create simple end-to-end non-batch example on Lightning

# Note
- The code is written in a mixture of marimo and regular .py files. This is becuase although marimo is preferred for it's ease of reporducibility, the GPU acitivity was performed on lightning.ai which is focused on .ipynb based code or simple .py scripts.


# Citing this project

If you use this project or the code used to generate it, please cite


_______No citation information available yet_________