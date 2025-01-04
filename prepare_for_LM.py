import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
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


@app.cell
def _():
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

    api_key = os.environ["MISTRAL_API_KEY"]

    client = Mistral(api_key=api_key)


    from bbox_functions import preprocess_bbox, save_plots_for_all_files
    from send_to_lm_functions import (split_image, crop_and_encode_boxes, create_jsonl_content,
    save_encoded_images, process_issues_to_jobs)
    return (
        BytesIO,
        Image,
        Mistral,
        Path,
        api_key,
        base64,
        client,
        create_jsonl_content,
        crop_and_encode_boxes,
        cv2,
        json,
        np,
        os,
        pd,
        preprocess_bbox,
        process_issues_to_jobs,
        save_encoded_images,
        save_plots_for_all_files,
        split_image,
        wand,
    )


@app.cell
def _(os, pd):
    images_folder = '/media/jonno/ncse/converted/all_files_png_120/Leader_issue_PDF_files'#'data/leader_test_issue/'

    output_folder = "data/leader_test_cropped"


    bbox_df = pd.read_parquet('data/periodical_bboxes/post_process/Leader_issue_PDF_files_1040.parquet')
    # Just subsetting to test issues of the leader to make sure the process works as expected
    test_issues = os.listdir(images_folder)

    # Just the subset of the files in the test set
    #bbox_df = bbox_df.loc[bbox_df['filename'].isin(test_issues)]

    #bbox_df['page_id'] = bbox_df['filename'].str.replace('.png', "")


    # All the preprossing of the data should be done in a single step and saved as it is probably quite slow.
    return bbox_df, images_folder, output_folder, test_issues


@app.cell
def _(bbox_df):
    bbox_df.columns
    return


@app.cell
def _(pd):
    issues_df = pd.read_parquet('data/periodicals_issue.parquet')

    print(issues_df.shape)
    issues_df.groupby('publication_id').size()
    return (issues_df,)


@app.cell
def _():
    4263/(60*24)
    return


@app.cell
def _(bbox_df):
    bbox_df
    return


@app.cell
def _(bbox_df):
    (bbox_df.groupby('class').size()/bbox_df.shape[0]).round(2)
    return


@app.cell
def _():
    prompt_dict = {'plain text':"You are an expert at transcription. The text is from a 19th century English newspaper. Please transcribe exactly, including linebreaks, the text found in the image. Do not add any commentary. Do not use mark up please transcribe using plain text only.",
                  'figure':'Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary',
                  'table':'Please extract the table from the image taken from a 19th century English newspaper. Use markdown, do not add any commentary'}
    return (prompt_dict,)


@app.cell
def _():

    """
    process_issues_to_jobs(bbox_df, images_folder, prompt_dict , client, 
                           output_file='data/processed_jobs/Leader_issue_PDF_files.csv')
    """
    return


@app.cell
def _(json):
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
def _(mo):
    mo.md(
        """
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
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _():

    """
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

    """
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
