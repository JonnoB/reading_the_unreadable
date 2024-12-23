import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # The process for preparing images with bounding boxes for sending to Pixtral

        This notebook finds the process to take a page image and convert it into a format that can be processed by the language model. 
        By this point the page will have had bounding boxes detected using docyolo-layout, and have been post processed. 

        This notebook shows the code necessary to 

        - Load an image or series of images
        - Crop out each region of interest
        - Split up the regions of interest
        - Create the batch json file
        - Send to Pixtral

        In this process each Issue will be a batch, this will allow for batches that are neither to large or too small, and contain all relevant information.

        I will trial the process on an issue of the leader specifically `CLD-1850-04-20`
        """
    )
    return


app._unparsable_cell(
    r"""
    import pandas as pd
    import numpy as np
    import os
    import cv2
    from pathlib import Path
    from PIL import Image  # 
    from io import BytesIO
    import base64
    import json
    import wand

    from mistralai import Mistral

    api_key = os.environ[\"MISTRAL_API_KEY\"]

    client = Mistral(api_key=api_key)


    from bbox_functions import preprocess_bbox, save_plots_for_all_files
    from send_to_lm_functions import (split_image, crop_and_encode_boxes, create_jsonl_content,  process_pages_to_jobs
    save_encoded_images)
    """,
    name="__"
)


@app.cell
def __(os, pd, preprocess_bbox):
    folder_path = 'data/leader_test_issue/'

    output_folder = "data/leader_test_cropped"


    bbox_df = pd.read_parquet('data/periodical_bboxes/raw/Leader_issue_PDF_files_1056.parquet')
    # Just subsetting to test issues of the leader to make sure the process works as expected
    test_issues = os.listdir(folder_path)

    # Just the subset of the files in the test set
    bbox_df = bbox_df.loc[bbox_df['filename'].isin(test_issues)]

    bbox_df['page_id'] = bbox_df['filename'].str.replace('.png', "")

    bbox_df = preprocess_bbox(bbox_df, 10)

    # All the preprossing of the data should be done in a single step and saved as it is probably quite slow.
    return bbox_df, folder_path, output_folder, test_issues


@app.cell
def __(bbox_df):
    bbox_df.columns
    return


@app.cell
def __(bbox_df):
    bbox_df
    return


@app.cell
def __(bbox_df, client, images_folder, process_pages_to_jobs, prompt):
    process_pages_to_jobs(bbox_df, images_folder, prompt ,client, output_file='data/leader_processed_jobs.csv')
    return


@app.cell
def __(
    bbox_df,
    client,
    create_batch_job,
    crop_and_encode_boxes,
    folder_path,
):
    # Run the cropping function
    encoded_images = crop_and_encode_boxes( df = bbox_df, 
                                            images_folder = folder_path, 
                                            max_ratio = 1, 
                                            overlap_fraction= 0.2, 
                                            deskew = True )

    prompt =  "You are an expert at transcription. The text is from a 19th century English newspaper. Please transcribe exactly, including linebreaks, the text found in the image. Do not add any commentary. Do not use mark up please transcribe using plain text only."


    job_id, target_filename = create_batch_job(client, bbox_df, encoded_images, prompt)
    return encoded_images, job_id, prompt, target_filename


@app.cell
def __(job_id):
    job_id
    return


@app.cell
def __(json):
    def get_completed_job_ids(client, job_type="testing"):
        """
        Get IDs of all successfully completed batch jobs.

        Args:
            client: Mistral client instance
            job_type (str): The job type to filter by in metadata (default: "testing")

        Returns:
            list: List of job IDs
        """
        completed_jobs = client.batch.jobs.list(
            status="SUCCESS",
            metadata={"job_type": job_type}
        )

        # Extract just the IDs from the job objects
        job_ids = [job.id for job in completed_jobs.data]

        return job_ids

    def process_mistral_responses(response):
        """
        Process the raw bytes data from Mistral API responses.

        Args:
            data (bytes): Raw response data from Mistral

        Returns:
            list: List of parsed JSON objects
        """

        data = response.read() 
        # Decode bytes to string
        text_data = data.decode('utf-8')

        # Split into individual JSON objects
        json_strings = text_data.strip().split('\n')

        parsed_responses = []
        for json_str in json_strings:
            try:
                parsed_response = json.loads(json_str)
                parsed_responses.append(parsed_response)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue

        return parsed_responses
    return get_completed_job_ids, process_mistral_responses


@app.cell
def __(client, get_completed_job_ids, json, process_mistral_responses):
    completed_ids = get_completed_job_ids(client)

    # need to then check against log of retrieved and stored jobs


    # Here if the completed id has not already been downloaded and stored retrieve the job
    retrieved_job = client.batch.jobs.get(job_id=completed_ids[0])
    # use the output file id to download the file
    response = client.files.download(file_id=retrieved_job.output_file)
    parsed_data = process_mistral_responses(response)

    # I need to work out how to get the original file name here
    #Save the file with the issue code
    with open('data/output.json', 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    return completed_ids, f, parsed_data, response, retrieved_job


@app.cell
def __(completed_ids):
    completed_ids
    return


@app.cell
def __():
    return


@app.cell
def __(parsed_data):
    # Usage:
     # Your original data

    # Now you can work with the parsed data
    for item in parsed_data:
        # Access specific fields, for example:
        custom_id = item.get('custom_id')
        content = item.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content')
        print(f"Custom ID: {custom_id}")
        print(f"Content: {content}\n")

    # If you want to save to a file:
    return content, custom_id, item


@app.cell
def __(data):
    data
    return


@app.cell
def __(response):
    help(response)
    return


@app.cell
def __(encoded_images, save_encoded_images):
    save_encoded_images(encoded_images, output_folder = 'data/leader_test_cropped/')
    return


@app.cell
def __(bbox_df):
    bbox_df.columns
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
