""" 
This post-processes all the bounding boxes from DOCLayout-Yolo so that they are cleaned up and standardised with reading order etcetera.


"""


import pandas as pd
import numpy as np
import os

from bbox_functions import postprocess_bbox, basic_box_data, reclassify_abandon_boxes, remove_duplicate_boxes


# Need to make it easier/clearer to have a mode which is for creating fine-tuning boxes vs boxes for final use


input_folder = 'data/periodical_bboxes/raw'
output_folder = 'data/periodical_bboxes/post_process'


#When the output will be used to fine tune the yolo model the bounding boxes require slightly different processing.
fine_tune_mode = False

if fine_tune_mode:
    output_folder = output_folder+"_raw"
    width_multiplier = None
    remove_abandon = False
else:
    width_multiplier = 1.5
    remove_abandon = True

os.makedirs(output_folder, exist_ok=True)

all_files = os.listdir(input_folder)
#all_files = ['Publishers_Circular_issue_PDF_files_1040.parquet']

for file in all_files:
    output_file_path = os.path.join(output_folder, file)
    
    # Skip if output file already exists
    if os.path.exists(output_file_path):
        print(f"Skipping {file} - already processed")
        continue
        
    print(f"Processing {file}")
    bbox_df = pd.read_parquet(os.path.join(input_folder, file))
    
    bbox_df['page_id'] = bbox_df['filename'].str.replace('.png', "")

    if fine_tune_mode:

        bbox_df['issue'] = bbox_df['filename'].str.split('_page_').str[0]
    
        bbox_df = basic_box_data(bbox_df)
    
        bbox_df = reclassify_abandon_boxes(bbox_df, top_fraction=0.1)

        # This re-labels everything that is not abandon, text, table, or figure as title.
        bbox_df['class'] = np.where((~bbox_df['class'].isin(['figure', 'table', 'text', 'abandon'])), 
                                    'title',  # Value if condition is True
                                    bbox_df['class'])  # Value if condition is False
        bbox_df = remove_duplicate_boxes(bbox_df)
    
    else:
        #Renames due to DOCLayout-Yolo naming conventions, having a single word class is more convenient for when I create
        #a custom ID to send to LM... yes I could use a code, but this is the choice I have made.
        bbox_df['class'] = np.where(bbox_df['class']=='plain text', 'text', bbox_df['class']) 
        bbox_df = postprocess_bbox(bbox_df, 10, width_multiplier= width_multiplier, remove_abandon=remove_abandon)
    
    bbox_df.to_parquet(output_file_path)