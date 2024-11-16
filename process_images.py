import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import os
    import time
    from datetime import datetime
    import base64
    from helper_functions import (create_page_dict, scale_bbox, crop_and_encode_images, 
    knit_strings, knit_string_list, process_page, process_image_with_api, split_image, initialize_log_file, load_image, process_image_segments, process_segment, save_text_output, update_log, process_jpeg_folder,
    compute_metric, load_txt_files_to_dataframe, load_and_join_texts_as_dataframe)
    import io
    from pdf2image import convert_from_path
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import json
    from dotenv import load_dotenv
    import shutil
    from io import BytesIO
    from tqdm import tqdm

    import seaborn as sns

    import psutil
    import traceback
    import re

    from mistralai import Mistral
    load_dotenv()



    api_key = os.environ["MISTRAL_API_KEY"]
    model = "pixtral-12b-2409"

    client = Mistral(api_key=api_key)

    data_folder = 'data'
    BLN_folder = 'data/BLN_results'
    save_folder = os.path.join(data_folder,'BLN600_mistral')
    os.makedirs(save_folder, exist_ok=True)
    return (
        BLN_folder,
        BytesIO,
        Image,
        ImageDraw,
        ImageFont,
        Mistral,
        api_key,
        base64,
        client,
        compute_metric,
        convert_from_path,
        create_page_dict,
        crop_and_encode_images,
        data_folder,
        datetime,
        initialize_log_file,
        io,
        json,
        knit_string_list,
        knit_strings,
        load_and_join_texts_as_dataframe,
        load_dotenv,
        load_image,
        load_txt_files_to_dataframe,
        mo,
        model,
        np,
        os,
        pd,
        plt,
        process_image_segments,
        process_image_with_api,
        process_jpeg_folder,
        process_page,
        process_segment,
        psutil,
        re,
        save_folder,
        save_text_output,
        scale_bbox,
        shutil,
        sns,
        split_image,
        time,
        tqdm,
        traceback,
        update_log,
    )


@app.cell
def __(data_folder, os):
    import wand.image
    from wand.display import display


    with wand.image.Image(filename=os.path.join(data_folder,'repeating.jpeg')) as img:
        img.deskew(0.4*img.quantum_range)
        img.save(filename=os.path.join(data_folder,'repeating_deskew.jpeg'))
        display(img)
    return display, img, wand


@app.cell
def __():
    transcriber_prompt = "You are an expert at transcription. The text is from a 19th century news article. Please transcribe exactly, including linebreaks, the text found in the image. Do not add any commentary. Do not use mark up please transcribe using plain text only."
    return (transcriber_prompt,)


@app.cell
def __(mo):
    mo.md(r"""## With deskew""")
    return


@app.cell
def __(BLN_folder, np, os, process_jpeg_folder, transcriber_prompt):
    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder = os.path.join(BLN_folder,'BLN600_deskew'),
                        prompt = transcriber_prompt, 
                         max_ratio=np.inf, overlap_fraction=0.2)

    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder = os.path.join(BLN_folder,'BLN600_deskew_ratio_15'),
                        prompt = transcriber_prompt, 
                         max_ratio=1.5, overlap_fraction=0.2)

    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder =  os.path.join(BLN_folder,'BLN600_deskew_ratio_10'),
                        prompt = transcriber_prompt, 
                         max_ratio=1.0, overlap_fraction=0.2)
    return


@app.cell
def __(mo):
    mo.md(r"""## Without Deskew""")
    return


@app.cell
def __(BLN_folder, np, os, process_jpeg_folder, transcriber_prompt):
    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder =  os.path.join(BLN_folder,'BLN600_ratio_1000'),
                        prompt = transcriber_prompt, 
                         max_ratio=np.inf, overlap_fraction=0.2, deskew =False)

    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder =  os.path.join(BLN_folder,'BLN600_ratio_15'),
                        prompt = transcriber_prompt, 
                         max_ratio=1.5, overlap_fraction=0.2, deskew = False)

    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder =  os.path.join(BLN_folder,'BLN600_ratio_10'),
                        prompt = transcriber_prompt, 
                         max_ratio=1.0, overlap_fraction=0.2, deskew = False)
    return


@app.cell
def __():
    import evaluate

    metric_cer = evaluate.load("cer")
    metric_wer = evaluate.load("wer")
    return evaluate, metric_cer, metric_wer


@app.cell
def __(BLN_folder, data_folder, load_and_join_texts_as_dataframe, os):
    _list_of_folders = [os.path.join(BLN_folder, folder) for folder in ['BLN600_deskew', 'BLN600_deskew_ratio_15', 'BLN600_ratio_1000', 'BLN600_ratio_15', 'BLN600_ratio_10', 'BLN600_deskew_ratio_10', 'american_stories_txt','BLN600_GOT', 'docling'
                                                                       ]]


    _list_of_folders = _list_of_folders + [os.path.join(data_folder, folder) for folder in ['BLN600/Ground Truth', 
                                                                                            'BLN600/OCR Text']]

    df = load_and_join_texts_as_dataframe(_list_of_folders)
    return (df,)


@app.cell
def __(compute_metric, df, metric_cer, metric_wer):
    df['cer_ocr'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='OCR Text', reference_col='Ground Truth')
    df['wer_ocr'] = df.apply(compute_metric, axis=1, metric =metric_wer, prediction_col='OCR Text', reference_col='Ground Truth')

    df['cer_deskew_1000'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_deskew', reference_col='Ground Truth')


    df['cer_deskew_15'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_deskew_ratio_15', reference_col='Ground Truth')

    df['cer_deskew_10'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_deskew_ratio_10', reference_col='Ground Truth')

    df['cer_nodeskew_1000'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_ratio_1000', reference_col='Ground Truth')

    df['cer_nodeskew_15'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_ratio_15', reference_col='Ground Truth')

    df['cer_nodeskew_10'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_ratio_10', reference_col='Ground Truth')

    df['am_stories'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='american_stories_txt', reference_col='Ground Truth')

    df['GOT'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_GOT', reference_col='Ground Truth')

    df['docling'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='docling', reference_col='Ground Truth')
    return


@app.cell
def __(df):
    df[['file_name','cer_ocr', 'cer_deskew_1000', 'cer_nodeskew_1000','cer_nodeskew_10', 
        'cer_deskew_10', 'am_stories', 'GOT', 'docling'
       ]].describe()
    return


@app.cell
def __(df, np, pd, plt, sns):
    _plot_df = df[['file_name',#'cer_ocr', 
                   'cer_deskew_1000', 'cer_nodeskew_1000', 
                   'cer_deskew_10', 'cer_nodeskew_10'#,'cer_nodeskew_15', 'cer_deskew_15',
                  ]].melt(id_vars = 'file_name')

    _plot_df['deskew'] = np.where(_plot_df['variable'].str.contains('no'),'no deskew', 'deskew' ) 

    mapping = {
        'cer_ocr': 'ocr',
        'cer_deskew_1000': 'Whole image',
        'cer_deskew_15': '1.5',
        'cer_deskew_10': 'Square Crop',
        'cer_nodeskew_1000': 'Whole image',
        'cer_nodeskew_15': '1.5',
        'cer_nodeskew_10': 'Square Crop',
    }


    # Create a new column with the mapped values
    _plot_df['mapped_category'] = _plot_df['variable'].map(mapping)

    # Function to calculate bootstrapped statistics
    def bootstrap_stats(data, n_bootstrap=1000, statistic='mean'):
        stats_func = np.mean if statistic == 'mean' else np.median
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(stats_func(sample))
        return bootstrap_stats

    # Calculate bootstrapped statistics for each group
    bootstrap_results = []
    for (var, deskew), group in _plot_df.groupby(['variable', 'deskew']):
        bootstrap_values = bootstrap_stats(group['value'].values, n_bootstrap=1000, statistic='mean')  # or 'median'
        bootstrap_results.extend([{
            'variable': var,
            'deskew': deskew,
            'bootstrapped_value': val
        } for val in bootstrap_values])

    # Create new dataframe with bootstrapped results
    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_df['mapped_category'] = bootstrap_df['variable'].map(mapping)

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=bootstrap_df, 
                x='mapped_category', 
                y='bootstrapped_value', 
                hue='deskew')

    plt.title('Analysing the impact of deskewing and cropping images')
    plt.xlabel('Image Crop')
    plt.ylabel('Bootstrapped Mean Value')
    plt.show()
    return (
        bootstrap_df,
        bootstrap_results,
        bootstrap_stats,
        bootstrap_values,
        deskew,
        group,
        mapping,
        var,
    )


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        """
        df2 = df.copy()
        df2['cer_diff_deskew'] = df2['cer_nodeskew_1000'] - df2['cer_deskew_1000']

        df2['cer_diff_ratio'] = df2['cer_deskew_1000'] - df2['cer_deskew_15']

        df2[['file_name', 'cer_diff_deskew', 'cer_diff_ratio' ]]
        """
    )
    return


if __name__ == "__main__":
    app.run()
