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
    from send_to_lm_functions import process_image_with_api, crop_and_encode_boxes
    from bbox_functions import plot_boxes_on_image


    total_root = "/media/jonno/ncse/converted/all_files_png_120"
    # image folder path
    EWJ = os.path.join(total_root, 'English_Womans_Journal_issue_PDF_files')
    CLD = os.path.join(total_root, 'Leader_issue_PDF_files')

    # load bbox dataframe
    bbox_df = pd.read_parquet('data/periodical_bboxes/post_process/English_Womans_Journal_issue_PDF_files_1040.parquet')


    CLD_bbox_df = pd.read_parquet('data/periodical_bboxes/post_process/Leader_issue_PDF_files_1040.parquet')
    # sub-select 1 or two issues from EWJ


    EWJ_sub = bbox_df.loc[bbox_df['issue']=='EWJ-1858-03-01'].sample(30, random_state = 1858)

    CLD_sub = CLD_bbox_df.loc[CLD_bbox_df['filename']=='CLD-1850-04-20_page_11.png']

    return (
        CLD,
        CLD_bbox_df,
        CLD_sub,
        EWJ,
        EWJ_sub,
        bbox_df,
        crop_and_encode_boxes,
        os,
        pd,
        plot_boxes_on_image,
        process_image_with_api,
        total_root,
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
        'plain text': "You are an expert at transcription. The text is from a 19th century English newspaper. Please transcribe exactly, including linebreaks, the text found in the image. Do not add any commentary. Do not use mark up please transcribe using plain text only.",
        'figure': 'Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary',
        'table': 'Please extract the table from the image taken from a 19th century English newspaper. Use markdown, do not add any commentary'
    }

    default_prompt = prompt_dict.get('plain text', 'Describe this text')
    return default_prompt, prompt_dict


@app.cell
def _(EWJ, EWJ_sub, crop_and_encode_boxes):
    image_dict = crop_and_encode_boxes(EWJ_sub, EWJ, max_ratio = 1)
    return (image_dict,)


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
    _prompt = """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Please transcribe using a markdown # to indicate headers, otherwise plain text only. Do not add any commentary."""

    _target_image = CLD_image_dict['CLD-1850-04-20_page_11_B0C1R1_segment_0']

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

    plot_boxes_on_image(plot_bbox, os.path.join(CLD,page_id+'.png') , figsize=(15,15), show_reading_order=False)

    return page_id, plot_bbox


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
