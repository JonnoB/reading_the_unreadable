import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        """
        Great little poem
        issue EWJ-1862-11-01 page 24 reading order 2

        I'm happy as the bee, love, 
        The winsome honey-bee, 
        In sunny hours from summer flowers 
        Drinking sweets so thirstingly.

        ISSUE EWJ-1858-09-01 PAGE 66 READING ORDER 8
        Let the luscious south wind Breathe in lover's sighs, While the lazy gallants, Bask in ladies' eyes.
        """
    )
    return


@app.cell
def _():
    from datasets import Dataset
    import random
    import os
    import glob
    from typing import List, Tuple
    import re
    import pandas as pd
    from function_modules.helper_functions_class import create_genre_prompt
    from function_modules.send_to_lm_functions import download_processed_jobs, process_json_files
    from datetime import datetime
    import json

    from mistralai import Mistral
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    return (
        Dataset,
        List,
        Mistral,
        Tuple,
        api_key,
        client,
        create_genre_prompt,
        datetime,
        download_processed_jobs,
        glob,
        json,
        os,
        pd,
        process_json_files,
        random,
        re,
    )


@app.cell
def _(os):
    dataset_folder = 'data/download_jobs/ncse/dataframes/post_processed'

    all_data_parquets = os.listdir(dataset_folder)
    return all_data_parquets, dataset_folder


@app.cell
def _(all_data_parquets, dataset_folder, os, pd):
    # Define the path for saving/loading the processed dataset
    output_file = os.path.join('data', 'text_type_training_set.parquet')

    # Check if the processed file already exists
    if os.path.exists(output_file):
        # Load the existing processed dataset
        data_set = pd.read_parquet(output_file)
    else:
        # Create the data folder if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Process the data as before
        data_set = []

        for periodical in all_data_parquets:
            df = pd.read_parquet(os.path.join(dataset_folder, periodical))
            df = df.loc[df['class'].isin(['title', 'text']) & (df['completion_tokens']>30)]
            df = df.sample(2000, random_state = 1850)
            df['bbox_uid'] = df['page_id'] + "_" + df['box_page_id']
            df = df[['bbox_uid', 'content']]

            data_set.append(df)

        data_set = pd.concat(data_set, ignore_index=True)

        # Save the processed dataset
        data_set.to_parquet(output_file)
    return data_set, df, output_file, periodical


@app.cell
def _(datetime, json, os, pd):
    def create_text_batch_job(client, base_filename, df, prompt_function, model="pixtral-12b-2409", job_type='testing', max_tokens=2000):
        """
        Create a batch job for text data from a DataFrame.

        Args:
            client: Mistral client instance
            base_filename: string of desired filename
            df: DataFrame containing 'bbox_uid' and 'content' columns
            prompt_function: function that generates the prompt for each text
            model: model to use for inference
            job_type: type of job (for metadata)
            max_tokens: maximum tokens for response

        Returns:
            tuple: (job_id, target_filename)
        """
        target_filename = f"{base_filename}.jsonl"

        # Create JSONL content
        jsonl_content = create_text_jsonl_content(df, prompt_function, max_tokens)

        # Convert string content to bytes
        content_bytes = jsonl_content.encode('utf-8')

        # Upload the file
        batch_data = client.files.upload(
            file={
                "file_name": target_filename,
                "content": content_bytes
            },
            purpose="batch"
        )

        # Create the job
        created_job = client.batch.jobs.create(
            input_files=[batch_data.id],
            model=model,
            endpoint="/v1/chat/completions",
            metadata={
                "job_type": job_type,
                "target_filename": target_filename
            }
        )

        return created_job.id, target_filename

    def create_text_jsonl_content(df, prompt_function, max_tokens=2000):
        """
        Create JSONL content for text data.

        Args:
            df: DataFrame containing 'bbox_uid' and 'content' columns
            prompt_function: function that generates the prompt for each text
            max_tokens: maximum tokens for response

        Returns:
            str: The JSONL content as a string
        """
        jsonl_lines = []

        for _, row in df.iterrows():
            prompt = prompt_function(row['content'])

            entry = {
                "custom_id": row['bbox_uid'],
                "body": {
                    "max_tokens": max_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                }
            }
            jsonl_lines.append(json.dumps(entry))

        return "\n".join(jsonl_lines)

    def check_and_create_batch_job(
        client,
        df,
        prompt_function,
        job_type,
        tracking_dir,
        tracking_filename=None,
        model="mistral-large-latest",
        base_filename_prefix=None,
        max_tokens=2000,
        additional_metadata=None
    ):
        """
        Check if job has been run before and create a new batch job if it hasn't.

        Args:
            client: Mistral client instance
            df: DataFrame containing data to process
            prompt_function: function to create prompts
            job_type: string identifying the type of job (e.g., 'genre_classification', 'sentiment_analysis')
            tracking_dir: directory path where tracking files should be stored
            tracking_filename: optional, specific name for the tracking file (default: {job_type}_job.csv)
            model: model to use for inference (default: "mistral-large-latest")
            base_filename_prefix: optional, prefix for the batch job filename 
                                (default: derived from job_type)
            max_tokens: maximum tokens for response (default: 2000)
            additional_metadata: optional dict of additional metadata to store in tracking file

        Returns:
            tuple: (job_id, filename) or (None, None) if job already exists
        """
        # Set default values based on job_type if not provided
        if tracking_filename is None:
            tracking_filename = f"{job_type}_job.csv"

        if base_filename_prefix is None:
            base_filename_prefix = job_type.replace('_', '-')

        # Create full paths
        os.makedirs(tracking_dir, exist_ok=True)
        tracking_file = os.path.join(tracking_dir, tracking_filename)

        # Check if tracking file exists and load it
        if os.path.exists(tracking_file):
            print(f"Found existing job tracking file for {job_type}.")
            return None, None

        # Create a new job
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{base_filename_prefix}_{timestamp}"

        # Create the batch job
        job_id, filename = create_text_batch_job(
            client=client,
            base_filename=base_filename,
            df=df,
            prompt_function=prompt_function,
            model=model,
            job_type=job_type,
            max_tokens=max_tokens
        )

        # Prepare job information
        job_info = {
            'job_id': [job_id],
            'filename': [filename],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'model': [model],
            'job_type': [job_type],
            'num_records': [len(df)]
        }

        # Add additional metadata if provided
        if additional_metadata:
            for key, value in additional_metadata.items():
                job_info[key] = [value]

        # Save to CSV
        job_info_df = pd.DataFrame(job_info)
        job_info_df.to_csv(tracking_file, index=False)

        print(f"Created new {job_type} job {job_id} with filename {filename}")
        print(f"Job information saved to {tracking_file}")

        return job_id, filename

    # Usage examples:
    return (
        check_and_create_batch_job,
        create_text_batch_job,
        create_text_jsonl_content,
    )


@app.cell
def _(check_and_create_batch_job, client, create_genre_prompt, data_set):
    check_and_create_batch_job(
        client,
        data_set,
        prompt_function = create_genre_prompt,
        job_type = 'text_type_classification',
        tracking_dir = 'data/classification/silver_data/processed_data',
        tracking_filename=None,
        model="mistral-large-latest",
        base_filename_prefix=None,
        max_tokens=200,
        additional_metadata=None
    )
    return


@app.cell
def _(client, download_processed_jobs, os, pd):
    # Define paths
    jobs_file = 'data/classification/silver_data/processed_data/text_type_classification_job.csv'
    output_dir = 'data/classification/silver_data/text_type'
    log_file = 'data/classification/text_type_log.csv'

    # Read the jobs file to get the filename
    if os.path.exists(jobs_file):
        job_info = pd.read_csv(jobs_file)
        expected_jsonl = os.path.join(output_dir, job_info['filename'].iloc[0])

        # Check if the output file already exists
        if os.path.exists(expected_jsonl):
            print(f"Output file {expected_jsonl} already exists. Skipping download.")
        else:
            print(f"Downloading results to {expected_jsonl}")
            download_processed_jobs(client, jobs_file=jobs_file, output_dir=output_dir, log_file=log_file)
    else:
        print(f"Jobs file {jobs_file} not found. No jobs to download.")
    return expected_jsonl, job_info, jobs_file, log_file, output_dir


@app.cell
def _(json, pd):
    def process_dictionary_responses(jsonl_path):
        """
        Process JSON responses containing dictionary classifications and return a dictionary
        where keys are custom_ids and values are the classification dictionaries.
        
        Parameters:
        -----------
        jsonl_path : str
            Path to the JSON file containing the responses
            
        Returns:
        --------
        dict
            Dictionary with custom_ids as keys and classification dictionaries as values
        """
        try:
            # Load JSON data
            with open(jsonl_path, 'r') as file:
                json_data = json.load(file)
                
            # Initialize results dictionary
            results = {}
            
            # Process each response
            for item in json_data:
                try:
                    custom_id = item['custom_id']
                    content = item['response']['body']['choices'][0]['message']['content']
                    
                    # Clean up the content string to convert it to a dictionary
                    # Remove potential markdown formatting
                    content = content.replace('```json\n', '').replace('\n```', '').strip()
                    
                    # Convert string representation of dictionary to actual dictionary
                    classification = eval(content)  # Be careful with eval - use only with trusted data
                    
                    # Store in results
                    results[custom_id] = classification
                    
                except Exception as e:
                    print(f"Error processing item with custom_id {item.get('custom_id', 'unknown')}: {str(e)}")
                    continue
                    
            return results
            
        except Exception as e:
            print(f"Error processing file {jsonl_path}: {str(e)}")
            return {}
            
    def classifications_dict_to_df(classifications_dict):
        """
        Convert a dictionary of classifications into a DataFrame.
        Handles cases where value dictionaries might be strings that need to be evaluated.
        
        Parameters:
        -----------
        classifications_dict : dict
            Dictionary where keys are custom_ids and values are classification dictionaries
            or string representations of dictionaries
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with custom_id and all classification elements as columns
        """
        rows = []
        for custom_id, class_dict in classifications_dict.items():
            # Start with the custom_id
            row = {'custom_id': custom_id}
            
            try:
                # Handle string representation of dictionary
                if isinstance(class_dict, str):
                    # Clean up the string (remove potential JSON formatting)
                    class_dict = class_dict.replace('```json\n', '').replace('\n```', '').strip()
                    # Convert string to dictionary
                    class_dict = eval(class_dict)  # Be careful with eval - use only with trusted data
                
                # Update row with classification data
                if isinstance(class_dict, dict):
                    row.update(class_dict)
                else:
                    print(f"Warning: Unexpected value type for custom_id {custom_id}: {type(class_dict)}")
                    row['value'] = class_dict
                    
            except Exception as e:
                print(f"Error processing custom_id {custom_id}: {str(e)}")
                print(f"Value: {class_dict}")
                # Add the raw value to help with debugging
                row['raw_value'] = str(class_dict)
                
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows).rename(columns = {'class':'class_code'})
        
        # Optional: Sort by custom_id
        df = df.sort_values('custom_id').reset_index(drop=True)
        
        return df

    return classifications_dict_to_df, process_dictionary_responses


@app.cell
def _(classifications_dict_to_df, data_set, process_dictionary_responses):

    classified_text_type =process_dictionary_responses( 'data/classification/silver_data/text_type/text_type_classification.jsonl')


    text_class_df = classifications_dict_to_df(classified_text_type)

    text_class_df['text_class'] = text_class_df['class_code'].map({
        0: 'news report',
        1: 'editorial',
        2: 'letter',
        3: 'advert',
        4: 'review',
        5: 'poem/song/story',
        6: 'other'
    })

    text_class_df = text_class_df.merge(data_set, left_on='custom_id', right_on = 'bbox_uid')

    text_class_df.to_parquet('data/classification/silver_data/silver_text_type.parquet')


    return classified_text_type, text_class_df


@app.cell
def _(text_class_df):
    text_class_df
    return


@app.cell
def _():
    return


@app.cell
def _(data_set):
    data_set
    return


@app.cell
def _(os, pd):
    test = pd.read_parquet(os.path.join('data/classification/silver_data/',
                                                                         'silver_text_type.parquet'))
    return (test,)


@app.cell
def _(test):
    test
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
