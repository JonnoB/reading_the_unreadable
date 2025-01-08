import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Real time OCR extraction
        This note book provides a simple example, processing the test set images for the NCSE
        """
    )
    return


@app.cell
def _():
    import os
    import pandas as pd
    from send_to_lm_functions import process_image_with_api, crop_and_encode_boxes, decompose_filenames, combine_article_segments, convert_returned_streaming_to_dataframe, process_image_with_api
    from bbox_functions import plot_boxes_on_image
    from tqdm import tqdm

    total_root = "/media/jonno/ncse/converted/all_files_png_120"
    # image folder path
    EWJ = os.path.join(total_root, 'English_Womans_Journal_issue_PDF_files')
    CLD = os.path.join(total_root, 'Leader_issue_PDF_files')

    # load bbox dataframe
    bbox_df = pd.read_parquet('data/periodical_bboxes/post_process/English_Womans_Journal_issue_PDF_files_1040.parquet')


    CLD_bbox_df = pd.read_parquet('data/periodical_bboxes/post_process/Leader_issue_PDF_files_1040.parquet')
    # sub-select 1 or two issues from EWJ


    EWJ_sub = bbox_df.loc[bbox_df['issue']=='EWJ-1858-03-01'].sample(30, random_state = 1858)

    CLD_sub = CLD_bbox_df.loc[CLD_bbox_df['filename'].isin(['CLD-1850-04-20_page_10.png', 'CLD-1850-04-20_page_11.png', 'CLD-1850-04-20_page_12.png'])]

    return (
        CLD,
        CLD_bbox_df,
        CLD_sub,
        EWJ,
        EWJ_sub,
        bbox_df,
        combine_article_segments,
        convert_returned_streaming_to_dataframe,
        crop_and_encode_boxes,
        decompose_filenames,
        os,
        pd,
        plot_boxes_on_image,
        process_image_with_api,
        total_root,
        tqdm,
    )


@app.cell
def _(CLD_sub):
    CLD_sub
    return


@app.cell
def _(EWJ_sub):

    EWJ_sub
    return


@app.cell
def _():
    prompt_dict = {
        'plain text': """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only, in the case of headers use #. Do not add any commentary.""",
        'figure': 'Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary',
        'table': 'Please extract the table from the image taken from a 19th century English newspaper. Use markdown, do not add any commentary'
    }

    default_prompt = prompt_dict.get('plain text', 'Describe this text')
    return default_prompt, prompt_dict


@app.cell
def _():
    #image_dict = crop_and_encode_boxes(EWJ_sub, EWJ, max_ratio = 1)
    return


@app.cell
def _(encoded_images, prompt_dict):
    jsonl_lines = []
    default_prompt = prompt_dict.get('plain text', 'Describe this text')

    for image_id, image_data in encoded_images.items():
        # Get the appropriate prompt based on the image class
        image_class = image_data.get('class', 'plain text')
        prompt = prompt_dict.get(image_class, default_prompt)
        
    return (
        default_prompt,
        image_class,
        image_data,
        image_id,
        jsonl_lines,
        prompt,
    )


@app.cell
def _(image_dict, process_image_with_api):

    _prompt = """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Please transcribe using markdown hash to indicate headers, otherwise plain text only. Do not add any commentary. """


    target_image = image_dict[ 'EWJ-1858-03-01_page_11_B0C1R1_segment_0']

    content, tokens = process_image_with_api(target_image['image'], 
                           # prompt_dict.get(target_image['class']), 
                            _prompt,                       
                            model="mistral/pixtral-12b-2409")
    return content, target_image, tokens


@app.cell
def _(content):
    content
    return


@app.cell
def _(CLD, CLD_sub, crop_and_encode_boxes):
    CLD_image_dict = crop_and_encode_boxes(CLD_sub, CLD, max_ratio = 1)
    return (CLD_image_dict,)


@app.cell
def _(CLD_image_dict):
    CLD_image_dict.keys()
    return


@app.cell
def _(CLD_image_dict, process_image_with_api):
    #
    _prompt = """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only, in the case of headers use #. Do not add any commentary."""

    _target_image = CLD_image_dict['CLD-1850-04-20_page_11_B0C1R2_segment_0']

    CLD_content, CLD_tokens = process_image_with_api(_target_image['image'], 
                           # prompt_dict.get(target_image['class']), 
                            _prompt,                       
                            model="mistral/pixtral-12b-2409")
    return CLD_content, CLD_tokens


@app.cell
def _(CLD_content):
    CLD_content
    return


@app.cell
def _(CLD, CLD_bbox_df, os, plot_boxes_on_image):

    page_id = 'CLD-1850-04-20_page_12'

    plot_bbox = CLD_bbox_df.loc[CLD_bbox_df['page_id']==page_id]

    plot_boxes_on_image(plot_bbox, os.path.join(CLD,page_id+'.png') , figsize=(15,15), show_reading_order=True)

    return page_id, plot_bbox


@app.cell
def _(
    CLD,
    CLD_sub,
    combine_article_segments,
    crop_and_encode_boxes,
    decompose_filenames,
    default_prompt,
    pd,
    process_image_with_api,
    tqdm,
):
    _prompt_dict = {
        'plain text': """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only. Do not add any commentary.""",
        'figure': 'Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary',
        'table': 'Please extract the table from the image taken from a 19th century English newspaper. Use markdown, do not add any commentary'
    }
    # Create empty lists to store the data
    _data_list = []

    _encoded_images = crop_and_encode_boxes(CLD_sub, CLD, max_ratio = 1)

    for _image_id, _image_data in tqdm(_encoded_images.items()):
        # Get the appropriate prompt based on the image class
        _image_class = _image_data.get('class', 'plain text')
        _prompt = _prompt_dict.get(_image_class, default_prompt)

        _content, _tokens = process_image_with_api(_image_data['image'], 
                            _prompt,                       
                            model="mistral/pixtral-12b-2409")
        
        # Create a dictionary for each iteration
        _row_dict = {
            'image_id': _image_id,
            'content': _content,
            'prompt_tokens': _tokens[0],
            'completion_tokens': _tokens[1],
            'total_tokens': _tokens[2]
        }
        
        # Append the dictionary to the list
        _data_list.append(_row_dict)

    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(_data_list)

    df = decompose_filenames(df.rename(columns = {'image_id':'custom_id'}))

    df = combine_article_segments(df)
    return (df,)


@app.cell
def _(df):


    df
    return


@app.cell
def _(
    CLD_image_dict,
    convert_returned_streaming_to_dataframe,
    process_image_with_api2,
):
    _prompt = """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only, in the case of headers use #. Do not add any commentary."""


    _image_id = 'CLD-1850-04-20_page_11_B0C1R2_segment_0'

    _target_image = CLD_image_dict[_image_id]

    CLD_content2 = process_image_with_api2(_target_image['image'], 
                           # prompt_dict.get(target_image['class']), 
                            _prompt,                       
                            model="mistral/pixtral-12b-2409")

    CLD_content2 = convert_returned_streaming_to_dataframe(CLD_content2, id=_image_id, custom_id=_image_id)
    return (CLD_content2,)


@app.cell
def _(CLD_content2):
    CLD_content2

    return


@app.cell
def _():
    import requests

    def delete_all_batch_files(client, api_key, limit=100):
        """
        Delete all files associated with batch processing, handling pagination.
        
        Args:
            client: Mistral client instance
            api_key: Your Mistral API key
            limit: Number of files to fetch per page
        
        Returns:
            tuple: (int, list) - Count of deleted files and list of any failed deletions
        """
        deleted_count = 0
        failed_deletions = []
        
        try:
            # Set up headers for requests
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            # Handle pagination
            has_more = True
            after = None

            while has_more:
                # List files with pagination
                params = {'limit': limit}
                if after:
                    params['after'] = after

                # Make GET request to list files
                list_response = requests.get(
                    'https://api.mistral.ai/v1/files',
                    headers=headers,
                    params=params
                )

                if list_response.status_code != 200:
                    print(f"Error listing files: Status code {list_response.status_code}")
                    return deleted_count, failed_deletions

                files_data = list_response.json()

                # Process this page of files
                for file in files_data.get('data', []):
                    try:
                        # Only delete files with purpose "batch"
                        if file['purpose'] == "batch":
                            # Make delete request
                            delete_response = requests.delete(
                                f'https://api.mistral.ai/v1/files/{file["id"]}',
                                headers=headers
                            )
                            
                            if delete_response.status_code == 200:
                                deleted_count += 1
                                print(f"Successfully deleted file with ID: {file['id']}")
                            else:
                                failed_deletions.append({
                                    'file_id': file['id'],
                                    'error': f"Status code: {delete_response.status_code}"
                                })
                                print(f"Failed to delete file with ID: {file['id']}, Status code: {delete_response.status_code}")
                                
                    except Exception as e:
                        failed_deletions.append({
                            'file_id': file['id'],
                            'error': str(e)
                        })
                        print(f"Failed to delete file with ID: {file['id']}. Error: {str(e)}")

                # Check if there are more files to process
                has_more = files_data.get('has_more', False)
                if has_more and files_data['data']:
                    after = files_data['data'][-1]['id']
                else:
                    has_more = False

            return deleted_count, failed_deletions

        except Exception as e:
            print(f"Error in process: {str(e)}")
            return deleted_count, failed_deletions
    return delete_all_batch_files, requests


@app.cell
def _(delete_all_batch_files, os):
    from mistralai import Mistral

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    deleted_count, failures = delete_all_batch_files(client, api_key, 5000)

    # Print results
    print(f"\nSummary:")
    print(f"Successfully deleted {deleted_count} files")
    if failures:
        print(f"Failed to delete {len(failures)} files:")
        for failure in failures:
            print(f"- File ID: {failure['file_id']}, Error: {failure['error']}")
    return Mistral, api_key, client, deleted_count, failure, failures


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
