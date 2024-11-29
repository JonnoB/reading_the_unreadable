#########
##
## Send images to pixtral starting from a full page then cropping the images down.
##
##
#########


print('loading packages')

import pandas as pd
import os

from helper_functions import (process_page)
import json
from dotenv import load_dotenv
from tqdm import tqdm

import logging
logging.basicConfig(filename='data/processing_errors.log', level=logging.ERROR)

from mistralai import Mistral
load_dotenv()
image_drive = '/media/jonno/ncse'

print('loading data')

input_file = 'data/page_dict.json'
with open(input_file, 'r') as f:
    page_dict = json.load(f)


api_key = os.environ["MISTRAL_API_KEY"]
model = "pixtral-12b-2409"

client = Mistral(api_key=api_key)

dataset_df = pd.read_parquet('data/example_set_1858-1860.parquet')
dataset_df['save_name'] = 'id_' + dataset_df['id'].astype(str) + '_type_' + dataset_df['article_type_id'].astype(str)+"_"+ dataset_df['file_name'].str.replace('.pdf', '.txt')
#create data folder for returned objects

save_folder = 'data/returned_text'
os.makedirs(save_folder, exist_ok=True)





target_pages_issues = dataset_df.copy().loc[:, 
['issue_id', 'page_id', 'page_number', 'file_name', 'folder_path', 'width', 'height']].drop_duplicates().reset_index(drop=True)

print(f"Number of issues to extract {len(target_pages_issues[['issue_id']].drop_duplicates())}, number of pages {len(target_pages_issues[['page_id']].drop_duplicates())},")



print('processing files')

processing_log_path = 'data/processing_log.csv'

# Initialize the log DataFrame
if os.path.exists(processing_log_path):
    log_df = pd.read_csv(processing_log_path)
else:
    log_df = pd.DataFrame(columns=['page_id', 'status', 'processing_time', 'timestamp', 'error_message'])

#remove already process pages 
processed_page_ids = log_df['page_id'].unique()
target_pages_issues = target_pages_issues[~target_pages_issues['page_id'].isin(processed_page_ids)]

for _, row in tqdm(target_pages_issues.iterrows(), total=len(target_pages_issues)):
    log_df = process_page(row, image_drive, page_dict, client, save_folder, dataset_df, log_df)
    
    # Save the updated log after each processed page
    log_df.to_csv(processing_log_path, index=False)
print("Processing complete.")