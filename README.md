# Ereading the unreadable
This repository is to convert scanned images of 19th century newspapers into an indexed and classified data collection.

This project is an evolution of [CLOCR-C](https://github.com/JonnoB/clocrc), [ScrambledText](https://github.com/JonnoB/scrambledtext_analysis), and the hackathon project [Archivstral](https://github.com/JonnoB/archivestal). It is an attempt to re-frame OCR as an Language Model tast, as well as provide, a substantial new searchable data collection of 19th century English Newspapers

## This project is currently underdevelopment... probably best not to use


# Project Pipeline

- Convert NCSE files to single page PNG
- Predict bounding boxes using DocLayout-YOLO
- Post-process bounding boxes

# To Do

- Create Articles
- Classify Articles

# Key Modules

- bbox_functions.py : Post processes bounding boxes to improve quality and consistency
- send_to_lm_functions.py : Prepares image files for sending to the LM and retrieving information from the LM

# Key files

- convert_all_ncse.py: Turns the NCSE PDF's into a single page PNG files at the specified DPI
- new_approach.py: Code used to explore and plot pages and bounding boxes
- transfer_files_to_lightning.py: Move image files to lightning.ai, this is simply for convenient use of GPU
- post_process_images.py: post process the image bounding boxes produced by DocLayout-yolo
- send_processed_issues.py: Used to send data with post processed bounding boxes to Pixtral


# Note
For package management I used the uv library from astral. It is super fast an avoids dependency issues.
