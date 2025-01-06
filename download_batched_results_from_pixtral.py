
import pandas as pd
import os
from mistralai import Mistral
from send_to_lm_functions import download_processed_jobs

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)



# Example usage
results= download_processed_jobs(
    client=client,
    jobs_file='data/processed_jobs/Leader_issue_PDF_files.csv',
    output_dir='data/download_jobs/Leader_issue_PDF_files',
    log_file='data/download_jobs/Leader_issue_PDF_files.csv'
)