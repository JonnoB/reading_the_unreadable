"""
This post-processes all the bounding boxes from DOCLayout-Yolo so that they are cleaned up and standardised with reading order etcetera.

The script allows for ouputing data for fine-tuning, and also whether columns should be filled or not.


"""

import pandas as pd
import numpy as np
import os

from function_modules.bbox_functions import (
    postprocess_bbox,
    basic_box_data,
    reclassify_abandon_boxes,
    remove_duplicate_boxes,
)

from pathlib import Path

# Change working directory to project root
os.chdir(Path(__file__).parent.parent)
# Need to make it easier/clearer to have a mode which is for creating fine-tuning boxes vs boxes for final use


input_folder = "data/periodical_bboxes/raw"
output_folder = "data/periodical_bboxes/post_process"


# When the output will be used to fine tune the yolo model the bounding boxes require slightly different processing.
fine_tune_mode = False
fill_columns_mode = True

if fine_tune_mode:
    output_folder = output_folder + "_raw"
    width_multiplier = None
    remove_abandon = False
else:
    width_multiplier = 1.5
    remove_abandon = True
    fill_columns = False
    if fill_columns_mode:
        fill_columns = True
        output_folder = output_folder + "_fill"

os.makedirs(output_folder, exist_ok=True)

all_files = os.listdir(input_folder)

for file in all_files:
    output_file_path = os.path.join(output_folder, file)

    # Skip if output file already exists
    if os.path.exists(output_file_path):
        print(f"Skipping {file} - already processed")
        continue

    print(f"Processing {file}")
    bbox_df = pd.read_parquet(os.path.join(input_folder, file))

    bbox_df["page_id"] = bbox_df["filename"].str.replace(".png", "")

    # Renames due to DOCLayout-Yolo naming conventions, having a single word class is more convenient for when I create
    # a custom ID to send to LM... yes I could use a code, but this is the choice I have made.
    bbox_df["class"] = np.where(
        bbox_df["class"] == "plain text", "text", bbox_df["class"]
    )
    if fine_tune_mode:
        bbox_df["issue"] = bbox_df["filename"].str.split("_page_").str[0]

        bbox_df = basic_box_data(bbox_df)

        bbox_df = reclassify_abandon_boxes(bbox_df, top_fraction=0.1)

        # This re-labels everything that is not abandon, text, table, or figure as title.
        bbox_df["class"] = np.where(
            (~bbox_df["class"].isin(["figure", "table", "text", "abandon"])),
            "title",  # Value if condition is True
            bbox_df["class"],
        )  # Value if condition is False
        bbox_df = remove_duplicate_boxes(bbox_df)

    else:
        bbox_df = postprocess_bbox(
            bbox_df,
            10,
            width_multiplier=width_multiplier,
            remove_abandon=remove_abandon,
            fill_columns=fill_columns,
        )

    bbox_df.to_parquet(output_file_path)
