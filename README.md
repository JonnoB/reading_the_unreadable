# Ereading the unreadable
This repository is to convert scanned images of 19th century newspapers into an indexed and classified data collection.

This project is an evolution of [CLOCR-C](https://github.com/JonnoB/clocrc), [ScrambledText](https://github.com/JonnoB/scrambledtext_analysis), and the hackathon project [Archivstral](https://github.com/JonnoB/archivestal). It is an attempt to re-frame OCR as an Language Model tast, as well as provide, a substantial new searchable data collection of 19th century English Newspapers

## This project is currently underdevelopment... probably best not to use


# Project Pipeline

- Convert NCSE files to single page PNG
- Predict bounding boxes using DocLayout-YOLO
- Post-process bounding boxes
- Batch Process Data
- Construct Text Pieces
- Classify Text

Each element of the project pipeline has a python script for execution. However, there is no single end-to-end script. 

# To Do
- Create Articles
- Classify Articles
- Create simple end-to-end non-batch example on Lightning

# Key Modules

- bbox_functions.py: Post processes bounding boxes to improve quality and consistency
- send_to_lm_functions.py: Prepares image files for sending to the LM and retrieving information from the LM
- analysis_functions.py: Support functions for the analysis script

# Key files

- convert_all_ncse.py: Turns the NCSE PDF's into a single page PNG files at the specified DPI
- transfer_files_to_lightning.py: Move image files to lightning.ai, this is simply for convenient use of GPU
- post_process_images.py: post process the image bounding boxes produced by DocLayout-yolo
- send_processed_issues_to_pixtral_as_batch.py: Used to send data with post-processed bounding boxes to Pixtral as a batch job
- download_batched_results_from_pixtral.py: Retrieve the batched results using the previously generated log file
- experiment_deskew_and_ratio_batch_send.py: Send the deskew and cropping experiment to the batch server
- experiment_deskew_and_ratio_batch_download.py: Retrieve the deskew and cropping experiment to the batch server
- analysis.py: The script/marimo notebook used to analyze results for the paper. Produces the tables plots etc.
- send_to_pixel_streaming.py: An example of realtime sending to the Pixtral server

The scripts and ipynb to measure the performance of the alternative models can be found in the "alternative_models" folder




# Note
For package management I used the uv library from astral. It is super fast an avoids dependency issues.

- The deskew and square crop is in process_images, DO NOT DELETE!
- example plotting is in sketching_postprocess_and_plot.py
