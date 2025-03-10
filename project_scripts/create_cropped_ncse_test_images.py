###############
#
# This script is used to break the NCSE test set up into single bounding box files
# This allows the OCR to be tested easily for all the OCR methods.
# This is only needed to replicate the comparison between methods.
#
##################


import pandas as pd
import os
from function_modules.send_to_lm_functions import crop_image_to_bbox
import cv2
from tqdm import tqdm

from pathlib import Path

# Change working directory to project root
os.chdir(Path(__file__).parent.parent)

ncse_testset_bbox_df = pd.read_csv("data/ncse_testset_bboxes.csv")

cropped_image_folder = "data/converted/cropped_images"

os.makedirs(cropped_image_folder, exist_ok=True)

total_rows = ncse_testset_bbox_df.shape[0]

for _index, _row in tqdm(
    ncse_testset_bbox_df.iterrows(), total=total_rows, desc="Processing images"
):
    cropped_file_name = f"{_row['page_id']}_{_row['box_page_id']}.png"

    _image_path = os.path.join("data/converted/ncse_bbox_test_set", _row["filename"])
    _image = cv2.imread(_image_path)

    _height, _width = _image.shape[:2]

    _cropped_image = crop_image_to_bbox(_image, _row, _height, _width)

    cv2.imwrite(os.path.join(cropped_image_folder, cropped_file_name), _cropped_image)
