"""
This script prepares and sends the images to Pixtral. It uses the post-processed bounding boxes
And sends the cropped images batched by issue. 

It contains three prompt types,

plain text
figure
table

"""

import pandas as pd
import os
from mistralai import Mistral
from send_to_lm_functions import process_issues_to_jobs

# Dictionary mapping parquet files to their corresponding image folders
path_mapping = {
    'Leader_issue_PDF_files_1040.parquet': '/media/jonno/ncse/converted/all_files_png_120/Leader_issue_PDF_files',
    'English_Womans_Journal_issue_PDF_files_1040.parquet': '/media/jonno/ncse/converted/all_files_png_120/English_Womans_Journal_issue_PDF_files',
    'Monthly_Repository_issue_PDF_files_1040.parquet': '/media/jonno/ncse/converted/all_files_png_120/Monthly_Repository_issue_PDF_files',
    'Tomahawk_issue_PDF_files_1040.parquet': '/media/jonno/ncse/converted/all_files_png_120/Tomahawk_issue_PDF_files',
    'Publishers_Circular_issue_PDF_files_1040.parquet': '/media/jonno/ncse/converted/all_files_png_120/Publishers_Circular_issue_PDF_files',
    'Northern_Star_issue_PDF_files_2080.parquet': '/media/jonno/ncse/converted/all_files_png_200/Northern_Star_issue_PDF_files'
}

# API setup
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# Prompt dictionary
prompt_dict = {
    'plain text': "You are an expert at transcription. The text is from a 19th century English newspaper. Please transcribe exactly, including linebreaks, the text found in the image. Do not add any commentary. Do not use mark up please transcribe using plain text only.",
    'figure': 'Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary',
    'table': 'Please extract the table from the image taken from a 19th century English newspaper. Use markdown, do not add any commentary'
}

# Process each file
for parquet_file, image_path in path_mapping.items():
    # Construct full path to parquet file
    parquet_path = os.path.join('data/periodical_bboxes/post_process', parquet_file)
    
    # Load bbox dataframe
    bbox_df = pd.read_parquet(parquet_path)
    
    # Create output filename based on the base folder name
    base_name = os.path.basename(image_path)
    output_file = f'data/processed_jobs/{base_name}.csv'
    
    print(f"Processing {parquet_file}...")
    
    # Process the data
    process_issues_to_jobs(
        bbox_df=bbox_df,
        images_folder=image_path,
        prompt_dict=prompt_dict,
        client=client,
        output_file=output_file
    )
    
    print(f"Completed processing {parquet_file}")

print("All files processed!")