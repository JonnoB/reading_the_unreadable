
"""
This script is for downloading the batch jobs when they are finished and getting them ready for post processing and analysis.
For each periodical the script produces, a folder containing all the jobs in json format, a csv log of the jobs, and a parquet of 
the jobs reassembeled into a dataframe where one row is one bounding box.

"""

import pandas as pd
import os
from mistralai import Mistral
from function_modules.send_to_lm_functions import download_processed_jobs, reassemble_issue_segments

from pathlib import Path

# Change working directory to project root
os.chdir(Path(__file__).parent.parent)

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)


processed_jobs_folder = 'data/processed_jobs/ncse'
download_jobs_folder = 'data/download_jobs/ncse'


for file in os.listdir(processed_jobs_folder):
    print(file)
    # download the data from each periodical
    json_folder = os.path.join(download_jobs_folder, file).replace('.csv', '')
    results = download_processed_jobs(
        client = client,
        jobs_file = os.path.join(processed_jobs_folder, file),
        output_dir = json_folder,
        log_file = os.path.join(download_jobs_folder, file)
    )

    # convert the json into a dataframe and save as a parquet
    df = []
    for json_file in os.listdir(json_folder):
        print("Re-building dataframe")
        df.append(reassemble_issue_segments(os.path.join(json_folder, json_file)))
                  
    df = pd.concat(df, ignore_index=True)

    df.to_parquet(json_folder+".parquet")
