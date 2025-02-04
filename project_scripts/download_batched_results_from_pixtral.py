"""
This script is for downloading the batch jobs when they are finished and getting them ready for post processing and analysis.
For each periodical the script produces, a folder containing all the jobs in json format, a csv log of the jobs, and a parquet of 
the jobs reassembeled into a dataframe where one row is one bounding box.
"""

import pandas as pd
import os
from mistralai import Mistral
from function_modules.send_to_lm_functions import (
    download_processed_jobs, 
    reassemble_issue_segments,
    process_json_files
)
from pathlib import Path
from tqdm import tqdm

# Change working directory to project root
os.chdir(Path(__file__).parent.parent)

# Setup API client
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# Setup paths
processed_jobs_folder = 'data/processed_jobs/ncse'
download_jobs_folder = 'data/download_jobs/ncse'
dataframe_folder = os.path.join(download_jobs_folder, 'dataframes')

# Create output directory
os.makedirs(dataframe_folder, exist_ok=True)

# Process each file
for file in tqdm(os.listdir(processed_jobs_folder), desc="Processing periodicals"):
    print(f"\nProcessing {file}")
    
    # Setup paths for this file
    json_folder = os.path.join(download_jobs_folder, file).replace('.csv', '')
    output_parquet = os.path.join(dataframe_folder, f"{file}.parquet")
    
    # Verify JSON folder exists and contains files
    if not os.path.exists(json_folder):
        print(f"Warning: Folder {json_folder} does not exist")
        continue
        
    json_files = [f for f in os.listdir(json_folder) if f.endswith(('.json', '.jsonl'))]
    if not json_files:
        print(f"Warning: No JSON files found in {json_folder}")
        continue
    
    print(f"Found {len(json_files)} JSON files in {json_folder}")
    
    try:
        # Download the data from each periodical
        results = download_processed_jobs(
            client=client,
            jobs_file=os.path.join(processed_jobs_folder, file),
            output_dir=json_folder,
            log_file=os.path.join(download_jobs_folder, file)
        )

        # Convert the JSONs to a dataframe using parallel processing
        print(f"Converting JSONs to dataframe for {file}")
        df = process_json_files(
            json_folder=json_folder,
            output_path=output_parquet,
            num_workers=None  # Will use CPU count - 1
        )
        
        print(f"Successfully processed {file}. DataFrame shape: {df.shape}")
        
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        continue

print("\nAll processing complete!")