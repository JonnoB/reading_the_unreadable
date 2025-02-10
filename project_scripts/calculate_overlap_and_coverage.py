"""
This script simply calculates the overlap and overage of the bounding boxes for the different post processing approaches

This is used to decide which post-processing to use.
"""

import pandas as pd
import os

from function_modules.bbox_functions import calculate_coverage_and_overlap

from pathlib import Path

# Change working directory to project root
os.chdir(Path(__file__).parent.parent)

# To allow quick testing to check everything works ok
sample_mode = False

post_process_folder = "data/periodical_bboxes"
source_folders = ["post_process", "post_process_fill", "post_process_raw"]

save_folder = "data/overlap_coverage"

for source_folder in source_folders:
    print(f"\nProcessing Folder:{source_folder}")
    source_folder_full_path = os.path.join(post_process_folder, source_folder)

    save_folder_full_path = os.path.join(save_folder, source_folder)

    os.makedirs(save_folder_full_path, exist_ok=True)

    for source_file in os.listdir(source_folder_full_path):
        save_file_path = os.path.join(save_folder_full_path, source_file)

        # Skip if output file already exists
        if os.path.exists(save_file_path):
            print(f"Skipping {source_file} - already processed")
            continue

        print(f"Processing periodical: {source_file}")

        df = pd.read_parquet(os.path.join(source_folder_full_path, source_file))

        if sample_mode:
            sample_pages = df["filename"].unique()[0:100]
            df = df.loc[df["filename"].isin(sample_pages)]

        df = calculate_coverage_and_overlap(df)

        df.to_parquet(save_file_path)
