# Project scripts information

This is the readme providing more detail on the scripts used to create the results in this project.

- calculate_overlap_and_coverage.pr: Used to find the quality metrics for evaluating the bounding boxes of the text, images, and tables.
- convert_all_ncse.py: Turns the NCSE PDF's into a single page PNG files at the specified DPI.
- create_cropped_ncse_test_images.py: Generates individual cropped images for each bounding box in the NCSE test set, useful for easy testing of OCR quality.
- download_batched_results_from_pixtral.py: Retrieve the batched results using the previously generated log file.
- experiment_deskew_and_ratio_batch_send.py: Send the deskew and cropping experiment to the batch server.
- experiment_deskew_and_ratio_batch_download.py: Retrieve the deskew and cropping experiment to the batch server.
- pixtral_large_download.py: Retrieve batched results from Pixtral large OCR test.
- pixtral_large_send.py: extract OCR from test sets using Pixtral Large, using batched data.
- post_process_bboxes.py: post process the image bounding boxes produced by DocLayout-yolo.
- process_downloads.py: convert downloaded json files into dataframes using the post-processing described in the paper.
- send_processed_issues_to_pixtral_as_batch.py: Used to send data with post-processed bounding boxes to Pixtral as a batch job.
- transfer_files_to_lightning.py: Move image files to lightning.ai, this is simply for convenient use of GPU.

The scripts re-set the working directory to the root of the repo for ease of accessing the data folders etc.