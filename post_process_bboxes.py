""" 
This post-processes all the bounding boxes so that they are cleaned up and standardised with reading order etcetera.


"""


import pandas as pd
import numpy as np
import os

from bbox_functions import preprocess_bbox



input_folder = 'data/periodical_bboxes/raw'
output_folder = 'data/periodical_bboxes/post_process'

os.makedirs(output_folder, exist_ok=True)

all_files = os.listdir(input_folder)
all_files = ['Publishers_Circular_issue_PDF_files_1040.parquet']

for file in all_files:
    output_file_path = os.path.join(output_folder, file)
    
    # Skip if output file already exists
    if os.path.exists(output_file_path):
        print(f"Skipping {file} - already processed")
        continue
        
    print(f"Processing {file}")
    bbox_df = pd.read_parquet(os.path.join(input_folder, file))
    
    bbox_df['page_id'] = bbox_df['filename'].str.replace('.png', "")
    bbox_df['class'] = np.where(bbox_df['class']=='plain text', 'text', bbox_df['class'])
    bbox_df = preprocess_bbox(bbox_df, 10)
    
    bbox_df.to_parquet(output_file_path)