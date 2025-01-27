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
    from function_modules.send_to_lm_functions import process_image_with_api, crop_and_encode_boxes, decompose_filenames, combine_article_segments, convert_returned_streaming_to_dataframe, delete_all_batch_files
    from function_modules.bbox_functions import plot_boxes_on_image
    from tqdm import tqdm
    import numpy as np

    from function_modules.bbox_functions import assign_columns, basic_box_data, create_reading_order
    image_folder = os.environ['image_folder']
    total_root = os.path.join(image_folder, "converted/all_files_png_120")
    # image folder path
    EWJ = os.path.join(total_root, 'English_Womans_Journal_issue_PDF_files')
    CLD = os.path.join(total_root, 'Leader_issue_PDF_files')

    CLD_bbox_df = pd.read_parquet('data/periodical_bboxes/post_process/Leader_issue_PDF_files_1040.parquet')
    # sub-select 1 or two issues from EWJ


    CLD_sub = CLD_bbox_df.loc[CLD_bbox_df['filename'].isin(['CLD-1850-04-20_page_10.png', 'CLD-1850-04-20_page_11.png'])]
    return (
        CLD,
        CLD_bbox_df,
        CLD_sub,
        EWJ,
        assign_columns,
        basic_box_data,
        combine_article_segments,
        convert_returned_streaming_to_dataframe,
        create_reading_order,
        crop_and_encode_boxes,
        decompose_filenames,
        delete_all_batch_files,
        np,
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
def _():
    prompt_dict = {
        'text': """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only, in the case of headers use #. Do not add any commentary.""",
        'figure': 'Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary',
        'table': 'Please extract the table from the image taken from a 19th century English newspaper. Use markdown, do not add any commentary'
    }

    default_prompt = prompt_dict.get('text', 'Describe this text')
    return default_prompt, prompt_dict


@app.cell
def _(CLD, CLD_sub, crop_and_encode_boxes):
    CLD_image_dict = crop_and_encode_boxes(CLD_sub, CLD, max_ratio = 1)
    return (CLD_image_dict,)


@app.cell
def _(CLD_image_dict):
    CLD_image_dict.keys()
    return


@app.cell
def _(CLD, CLD_bbox_df, os, plot_boxes_on_image):
    page_id = 'CLD-1850-04-20_page_12'

    plot_bbox = CLD_bbox_df.loc[CLD_bbox_df['page_id']==page_id]

    plot_boxes_on_image(plot_bbox, os.path.join(CLD,page_id+'.png') , figsize=(15,15), show_reading_order=True)
    return page_id, plot_bbox


@app.cell
def _(CLD_image_dict):
    CLD_image_dict.keys()
    return


@app.cell
def _(
    CLD_image_dict,
    convert_returned_streaming_to_dataframe,
    process_image_with_api,
):
    _prompt = """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only, in the case of headers use #. Do not add any commentary."""


    _image_id = 'CLD-1850-04-20_page_10_B0C2R4_segment_0'

    _target_image = CLD_image_dict[_image_id]

    CLD_content2 = process_image_with_api(_target_image['image'], 
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
def _(mo):
    mo.md(
        """
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
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""# Format the NCSE testset bounding boxes""")
    return


@app.cell
def _(assign_columns, basic_box_data, create_reading_order, np, pd):
    import json


    # Initialize lists to store the data
    data_list = []

    # Read the NDJSON file
    with open('data/converted/Export  project - NCSE_bbox_testset - 1_11_2025.ndjson', 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                json_obj = json.loads(line)

                # Extract base information that will be repeated for each box
                base_info = {           
                    'filename': json_obj['data_row']['external_id'],
                    'height': json_obj['media_attributes']['height'],
                    'width': json_obj['media_attributes']['width'],
                }

                # Extract bounding box information
                project_key = list(json_obj['projects'].keys())[0]
                boxes = json_obj['projects'][project_key]['labels'][0]['annotations']['objects']

                # Create a new row for each box
                for box in boxes:
                    row = base_info.copy()  # Create a copy of base info
                    # Calculate x1,y1,x2,y2
                    x1 = box['bounding_box']['left']
                    y1 = box['bounding_box']['top']
                    x2 = x1 + box['bounding_box']['width']
                    y2 = y1 + box['bounding_box']['height']

                    row.update({
                        'class': box['name'],
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    })
                    data_list.append(row)

    # Create DataFrame
    df2 = pd.DataFrame(data_list)
    df2['class'] = np.where(df2['class']=='text', 'text', df2['class'])
    df2['confidence'] = 1
    df2['page_id'] = df2['filename'].str.replace(".png", "")

    df2 = basic_box_data(df2)

    # Display the DataFrame
    df2 = assign_columns(df2)

    df2= create_reading_order(df2)

    df2['box_page_id'] = "B" + df2['page_block'].astype(str) + "C"+df2['column_number'].astype(str)  + "R" + df2['reading_order'].astype(str) 
    df2['ratio'] = df2['height']/df2['width']  
    df2

    df2.to_csv('data/ncse_testset_bboxes.csv')
    return (
        base_info,
        box,
        boxes,
        data_list,
        df2,
        file,
        json,
        json_obj,
        line,
        project_key,
        row,
        x1,
        x2,
        y1,
        y2,
    )


@app.cell
def _(df2):
    df2
    return


@app.cell
def _(mo):
    mo.md("""The below block only runs if the files have not already been created as it takes about 30 mins and costs money""")
    return


@app.cell
def _(
    combine_article_segments,
    convert_returned_streaming_to_dataframe,
    crop_and_encode_boxes,
    decompose_filenames,
    default_prompt,
    df2,
    os,
    pd,
    process_image_with_api,
    tqdm,
):
    _prompt_dict = {
        'text': """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only. Do not add any commentary.""",
        'figure': 'Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary',
        'table': 'Please extract the table from the image taken from a 19th century English newspaper. Use markdown, do not add any commentary'
    }


    _save_path = 'data/converted/pixtral_test_resutls.csv'

    if not os.path.exists(_save_path):

        # Create empty lists to store the data
        _data_list = []

        test_set_folder = 'data/converted/ncse_bbox_test_set'

        _encoded_images = crop_and_encode_boxes(df2, test_set_folder, max_ratio = 1)

        for _image_id, _image_data in tqdm(_encoded_images.items()):
            # Get the appropriate prompt based on the image class
            _image_class = _image_data.get('class', 'text')
            _prompt = _prompt_dict.get(_image_class, default_prompt)

            _response = process_image_with_api(_image_data['image'], 
                                _prompt,                       
                                model="mistral/pixtral-12b-2409")

            # Create a dictionary for each iteration
            _extracted_data = convert_returned_streaming_to_dataframe(_response, id=_image_id, custom_id=_image_id)

            # Append the dictionary to the list
            _data_list.append(_extracted_data)

        # Create DataFrame from the list of dictionaries
        _df = pd.concat(_data_list, ignore_index=True)

        _df = decompose_filenames(_df)

        test_results_df = combine_article_segments(_df)

        test_results_df.to_csv(_save_path)

    else: 
        test_results_df = pd.read_csv(_save_path)
    return test_results_df, test_set_folder


@app.cell
def _(test_results_df):
    test_results_df
    return


@app.cell
def _(df2):
    df2
    return


@app.cell
def _(process_image_with_api):


    from function_modules.send_to_lm_functions import deskew_image, encode_pil_image
    from PIL import Image


    _prompt = """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only. Do not add any commentary."""

    image_class = 'text'

    example_image = Image.open('data/example_for_paper/NS2-1843-04-01_page_4_excerpt.png')

    example_image = deskew_image(example_image)

    _key, _encoded_example_dict = encode_pil_image(example_image, "NS2-1843-04-01_page_4", "B0C1R0", 0, image_class)

    returned_example = process_image_with_api(_encoded_example_dict['image'], _prompt,  model="mistral/pixtral-12b-2409")
    return (
        Image,
        deskew_image,
        encode_pil_image,
        example_image,
        image_class,
        returned_example,
    )


@app.cell
def _(returned_example):
    returned_example
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
