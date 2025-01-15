import pandas as pd
import os
from mistralai import Mistral
from send_to_lm_functions import download_processed_jobs, convert_returned_json_to_dataframe, combine_article_segments, decompose_filenames
import json
import glob

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

experiment_dir = f'data/download_jobs/experiments'
json_dir = os.path.join(experiment_dir, 'json')
os.makedirs(json_dir, exist_ok=True)

dataframe_dir = os.path.join(experiment_dir, 'dataframe')
os.makedirs(dataframe_dir, exist_ok=True)

experiments = list(zip(
    [True, True, True,True, False, False, False, False],
    [1, 1.5, 2,1000, 1, 1.5, 2,1000]
))


for dataset in ['NCSE','BLN600']:
    for deskew, max_ratio in experiments:
        file_name = f"{dataset}_deskew_{str(deskew)}_max_ratio_{str(max_ratio)}"
        
        print(f"Processing {file_name}")

        # Example usage
        results = download_processed_jobs(
            client=client,
            jobs_file=f'data/processed_jobs/{file_name}.csv',
            output_dir=json_dir,
            log_file=f'data/download_jobs/experiments/{file_name}_log.csv'
        )

        # Find all batch files for this experiment
        batch_files = glob.glob(os.path.join(json_dir, f'{file_name}_*.jsonl'))
        
        if not batch_files:
            print(f"Warning: No batch files found for {file_name}")
            continue

        try:
            # Initialize an empty list to store all JSON data
            all_json_data = []
            
            # Read each batch file
            for batch_file in sorted(batch_files):  # sorted to process in order
                print(f"Processing batch file: {os.path.basename(batch_file)}")
                with open(batch_file, 'r', encoding='utf-8') as json_file:
                    # Load the JSON array directly
                    batch_data = json.load(json_file)
                    all_json_data.extend(batch_data)
            
            # Convert all collected data to dataframe
            content_df = convert_returned_json_to_dataframe(all_json_data)
            #if dataset =='BLN600':
            #    content_df['custom_id'] = content_df['custom_id'].str.replace("_B", "_page_1_B")
            #content_df.to_csv('test.csv')
            # Post processing
            content_df = decompose_filenames(content_df)
            content_df = combine_article_segments(content_df)
            content_df['filename'] = content_df['page_id'] + "_box_page_id_" + content_df['box_page_id'] + ".txt"
            
            # Save combined results
            output_path = os.path.join(dataframe_dir, f'{file_name}.csv')
            content_df.to_csv(output_path, index=False)
            print(f"Saved combined results to {output_path}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            # Print more detailed error information
            import traceback
            print(traceback.format_exc())