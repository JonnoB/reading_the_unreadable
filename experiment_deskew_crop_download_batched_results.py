
import pandas as pd
import os
from mistralai import Mistral
from send_to_lm_functions import download_processed_jobs, convert_returned_json_to_dataframe, combine_article_segments, decompose_filenames
import json

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

experiment_dir = f'data/download_jobs/experiments'
json_dir = os.path.join(experiment_dir, 'json')
os.makedirs(json_dir, exist_ok=True)

dataframe_dir = os.path.join(experiment_dir, 'dataframe')
os.makedirs(dataframe_dir, exist_ok=True)


experiments = list(zip(
    [True, True, True, False, False, False],
    [1, 1.5, 2, 1, 1.5, 2]
))

for dataset in ['NCSE','BLN600']:

    for deskew, max_ratio in experiments:

        file_name = f"{dataset}_deskew_{str(deskew)}_max_ratio_{str(max_ratio)}"

        # Example usage
        results= download_processed_jobs(
            client=client,
            jobs_file=f'data/processed_jobs/{file_name}.csv',
            output_dir=json_dir,
            log_file=f'data/download_jobs/experiments/{file_name}_log.csv'
        )

        

        json_path = os.path.join(json_dir, f'{file_name}.jsonl')

        with open(json_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            content_df = convert_returned_json_to_dataframe(json_data)

        # Post processing the returned data to create sensible csv's that match the original bounding boxes
        content_df = decompose_filenames(content_df)
        content_df = combine_article_segments(content_df)
        content_df['filename'] = content_df['page_id'] + "_box_page_id_" + content_df['box_page_id'] + ".txt"
        content_df.to_csv(os.path.join(dataframe_dir,  f'{file_name}.csv'), index=False)