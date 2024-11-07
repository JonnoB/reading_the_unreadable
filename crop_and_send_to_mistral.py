import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Introduction

        Early test code. Can probably be deleted 05-11-24
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import os

    from helper_functions import (create_page_dict, scale_bbox, crop_and_encode_images, 
    knit_strings, process_image_with_api, knit_string_list)
    import io
    from pdf2image import convert_from_path
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import json
    from dotenv import load_dotenv

    from mistralai import Mistral
    load_dotenv()
    image_drive = '/media/jonno/ncse'

    input_file = 'data/page_dict.json'
    with open(input_file, 'r') as f:
        page_dict = json.load(f)


    api_key = os.environ["MISTRAL_API_KEY"]
    model = "pixtral-12b-2409"

    client = Mistral(api_key=api_key)

    dataset_df = pd.read_parquet('data/example_set_1858-1860.parquet')
    dataset_df['save_name'] = 'id_' + dataset_df['id'].astype(str) + '_type_' + dataset_df['article_type_id'].astype(str)+"_"+ dataset_df['file_name'].str.replace('.pdf', '.txt')
    #create data folder for returned objects

    save_folder = 'data/returned_text'
    os.makedirs(save_folder, exist_ok=True)
    return (
        Image,
        ImageDraw,
        ImageFont,
        Mistral,
        api_key,
        client,
        convert_from_path,
        create_page_dict,
        crop_and_encode_images,
        dataset_df,
        f,
        image_drive,
        input_file,
        io,
        json,
        knit_string_list,
        knit_strings,
        load_dotenv,
        mo,
        model,
        os,
        page_dict,
        pd,
        plt,
        process_image_with_api,
        save_folder,
        scale_bbox,
    )


@app.cell
def __(dataset_df):
    target_pages_issues = dataset_df.copy().loc[:, 
    ['issue_id', 'page_id', 'page_number', 'file_name', 'folder_path', 'width', 'height']].drop_duplicates().reset_index(drop=True)

    print(f"Number of issues to extract {len(target_pages_issues[['issue_id']].drop_duplicates())}, number of pages {len(target_pages_issues[['page_id']].drop_duplicates())},")
    return (target_pages_issues,)


@app.cell
def __(dataset_df):
    dataset_df
    return


@app.cell
def __():
    #load issue
    return


@app.cell
def __(convert_from_path, image_drive, os, target_pages_issues):
    #select row
    check_row_df = target_pages_issues.loc[300, :]

    _file = os.path.join(image_drive, check_row_df['folder_path'], check_row_df['file_name'])

    #load all pages from an issue
    #relatively slow
    all_pages = convert_from_path(_file, dpi = 300)
    return all_pages, check_row_df


@app.cell
def __(mo):
    mo.md(r"""## Plot 1 page from the issue""")
    return


@app.cell
def __(all_pages, check_row_df):
    all_pages[check_row_df['page_number']-1]
    return


@app.cell
def __(all_pages, check_row_df, crop_and_encode_images, page_dict):
    # Use the function
    _page = all_pages[check_row_df['page_number']-1].copy()
    _bounding_boxes = page_dict[str(check_row_df['page_id'])]

    cropped_images = crop_and_encode_images(
        _page,
        _bounding_boxes,
        (check_row_df['width'], check_row_df['height']),
        _page.size
    )

    # Now 'cropped_images' is a list of PIL Image objects, each representing a cropped region
    return (cropped_images,)


@app.cell
def __(cropped_images):
    cropped_images
    return


@app.cell
def __(
    client,
    cropped_images,
    dataset_df,
    knit_string_list,
    process_image_with_api,
):
    article_id = '522845'

    image_string = cropped_images[article_id]

    content_list = []
    usage_list = []
    for image in image_string:

        _content, _usage = process_image_with_api(image, client, model="pixtral-12b-2409")
        content_list.append(_content)
        usage_list.append(_usage)

    full_string = knit_string_list(content_list)

    _file_name = dataset_df.loc[dataset_df['id']==int(article_id), 'save_name'].iloc[0]

    #with open(os.path.join(save_folder, _file_name), "w") as file:
    #    file.write(full_string)
    return (
        article_id,
        content_list,
        full_string,
        image,
        image_string,
        usage_list,
    )


@app.cell
def __(usage_list):
    usage_list
    return


@app.cell
def __():
    (8000+1500)*97000/1e6
    return


@app.cell
def __():
    921*0.15
    return


if __name__ == "__main__":
    app.run()
