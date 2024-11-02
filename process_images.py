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
    knit_strings, knit_string_list, process_page, process_image_with_api)
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
    save_folder = os.path.join(data_folder,'BLN600_mistral')
    os.makedirs(save_folder, exist_ok=True)
    return (
        BytesIO,
        Image,
        ImageDraw,
        ImageFont,
        Mistral,
        api_key,
        base64,
        client,
        convert_from_path,
        create_page_dict,
        crop_and_encode_images,
        data_folder,
        datetime,
        io,
        json,
        knit_string_list,
        knit_strings,
        load_dotenv,
        mo,
        model,
        np,
        os,
        pd,
        plt,
        process_image_with_api,
        process_page,
        psutil,
        re,
        save_folder,
        scale_bbox,
        shutil,
        sns,
        time,
        tqdm,
        traceback,
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
def __(
    BytesIO,
    Image,
    base64,
    datetime,
    knit_string_list,
    os,
    pd,
    process_image_with_api,
    time,
    tqdm,
    wand,
):
    def split_image(image, max_ratio=1.5, overlap_fraction=0.2, max_segments=10):
        """
        Split an image into segments based on a maximum aspect ratio.
        The final segment will maintain the same ratio as other segments.

        Args:
            image: PIL Image object
            max_ratio (float): Maximum width-to-height ratio before splitting
            overlap_fraction (float): Fraction of overlap between segments
            max_segments (int): Maximum number of segments to create

        Returns:
            list: List of PIL Image objects (segments)
        """
        width, height = image.size
        current_ratio = height / width

        # If the image ratio is already acceptable, return the original image
        if current_ratio <= max_ratio:
            return [image]

        # Calculate the ideal height for each segment
        segment_height = int(width * max_ratio)
        overlap_pixels = int(segment_height * overlap_fraction)

        segments = []
        y_start = 0

        while y_start < height and len(segments) < max_segments:
            y_end = min(y_start + segment_height, height)

            # Check if this would be the final segment
            remaining_height = height - y_start
            if remaining_height < segment_height and remaining_height / width < max_ratio:
                # Adjust y_start backwards to maintain the desired ratio for the final segment
                y_start = height - segment_height
                y_end = height

                # If this isn't the first segment, add it
                if segments:
                    segment = image.crop((0, y_start, width, y_end))
                    segments.append(segment)
                break

            # Create the segment
            segment = image.crop((0, y_start, width, y_end))
            segments.append(segment)

            # If we've reached the bottom of the image, break
            if y_end >= height:
                break

            # Calculate the start position for the next segment
            y_start = y_end - overlap_pixels

        return segments



    # Core process functions

    def initialize_log_file(output_folder):
        log_file_path = os.path.join(output_folder, 'processing_log.csv')
        if os.path.exists(log_file_path):
            log_df = pd.read_csv(log_file_path)
        else:
            log_df = pd.DataFrame(columns=['file_name', 'processing_time', 'input_tokens', 'output_tokens', 'total_tokens', 'sub_images', 'status', 'timestamp'])
        return log_file_path, log_df

    def load_image(file_path, deskew, output_folder):
        if deskew:
            with wand.image.Image(filename=file_path) as wand_img:
                wand_img.deskew(0.4 * wand_img.quantum_range)
                temp_path = os.path.join(output_folder, f"temp_deskewed_{os.path.basename(file_path)}")
                wand_img.save(filename=temp_path)
                img = Image.open(temp_path)
                os.remove(temp_path)
        else:
            img = Image.open(file_path)
        return img

    def process_image_segments(segments, prompt):
        content_list = []
        total_input_tokens = total_output_tokens = total_tokens = 0

        for i, segment in enumerate(segments):
            segment_content, usage = process_segment(segment, prompt)
            if segment_content and usage:
                content_list.append(segment_content)
                input_tokens, output_tokens, segment_total_tokens = usage
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_tokens += segment_total_tokens

        return content_list, total_input_tokens, total_output_tokens, total_tokens, len(segments)


    def process_segment(segment, prompt):
        buffered = BytesIO()
        segment.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        try:
            segment_content, usage = process_image_with_api(image_base64, prompt=prompt)
        except Exception as e:
            print(f"Error in process_image_with_api for segment: {str(e)}")
            segment_content, usage = None, None
        return segment_content, usage

    def save_text_output(output_folder, filename, content_list):
        combined_content = knit_string_list(content_list)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_content)

    def update_log(log_df, filename, processing_time, total_input_tokens, total_output_tokens, total_tokens, sub_images, status):
        log_entry = pd.DataFrame({
            'file_name': [filename],
            'processing_time': [processing_time],
            'input_tokens': [total_input_tokens],
            'output_tokens': [total_output_tokens],
            'total_tokens': [total_tokens],
            'sub_images': [sub_images],
            'status': [status],
            'timestamp': [datetime.now()]
        })
        return pd.concat([log_df[log_df['file_name'] != filename], log_entry], ignore_index=True)



    def process_jpeg_folder(folder_path, output_folder, prompt, max_ratio=1.5, overlap_fraction=0.1, deskew=True):
        os.makedirs(output_folder, exist_ok=True)
        log_file_path, log_df = initialize_log_file(output_folder)

        for filename in tqdm(os.listdir(folder_path)):
            if filename.lower().endswith(('.jpg', '.jpeg')) and filename not in log_df[log_df['status'] == 'Success']['file_name'].values:
                file_path = os.path.join(folder_path, filename)
                start_time = time.time()

                try:
                    img = load_image(file_path, deskew, output_folder)
                    segments = split_image(img, max_ratio, overlap_fraction)
                    content_list, total_input_tokens, total_output_tokens, total_tokens, sub_images = process_image_segments(segments, prompt)
                    save_text_output(output_folder, filename, content_list)
                    status = 'Success'
                except Exception as e:
                    print(f"Processing failed for {filename}: {str(e)}")
                    total_input_tokens = total_output_tokens = total_tokens = sub_images = 0
                    status = 'Failed'
                finally:
                    if 'img' in locals():
                        img.close()

                processing_time = time.time() - start_time
                log_df = update_log(log_df, filename, processing_time, total_input_tokens, total_output_tokens, total_tokens, sub_images, status)
                log_df.to_csv(log_file_path, index=False)

        return log_df
    return (
        initialize_log_file,
        load_image,
        process_image_segments,
        process_jpeg_folder,
        process_segment,
        save_text_output,
        split_image,
        update_log,
    )


@app.cell
def __():
    transcriber_prompt = "You are an expert at transcription. The text is from a 19th century news article. Please transcribe exactly, including linebreaks, the text found in the image. Do not add any commentary. Do not use mark up please transcribe using plain text only."
    return (transcriber_prompt,)


@app.cell
def __(mo):
    mo.md(r"""## With deskew""")
    return


@app.cell
def __(np, process_jpeg_folder, transcriber_prompt):
    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder = 'data/BLN600_deskew',
                        prompt = transcriber_prompt, 
                         max_ratio=np.inf, overlap_fraction=0.2)

    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder = 'data/BLN600_deskew_ratio_15',
                        prompt = transcriber_prompt, 
                         max_ratio=1.5, overlap_fraction=0.2)

    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder = 'data/BLN600_deskew_ratio_10',
                        prompt = transcriber_prompt, 
                         max_ratio=1.0, overlap_fraction=0.2)
    return


@app.cell
def __(mo):
    mo.md(r"""## Without Deskew""")
    return


@app.cell
def __(np, process_jpeg_folder, transcriber_prompt):
    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder = 'data/BLN600_ratio_1000',
                        prompt = transcriber_prompt, 
                         max_ratio=np.inf, overlap_fraction=0.2, deskew =False)

    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder = 'data/BLN600_ratio_15',
                        prompt = transcriber_prompt, 
                         max_ratio=1.5, overlap_fraction=0.2, deskew = False)

    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                        output_folder = 'data/BLN600_ratio_10',
                        prompt = transcriber_prompt, 
                         max_ratio=10, overlap_fraction=0.2, deskew = False)
    return


@app.cell
def __(os, pd, re):
    import evaluate

    metric_cer = evaluate.load("cer")
    metric_wer = evaluate.load("wer")

    from markdown_it import MarkdownIt
    from mdit_plain.renderer import RendererPlain

    parser = MarkdownIt(renderer_cls=RendererPlain)


    def compute_metric(row, metric, prediction_col, reference_col):
        try:
            # Preprocess the text: lowercasing and replacing line breaks with spaces
            prediction = re.sub(r'\s+', ' ', row[prediction_col].lower().strip())
            #prediction = parser.render(prediction)
            reference = re.sub(r'\s+', ' ', row[reference_col].lower().strip())

            # Ensure the inputs to metric.compute are lists of strings
            predictions = [prediction]
            references = [reference]
            return metric.compute(predictions=predictions, references=references)
        except KeyError as e:
           #print(f"KeyError: {e} in row: {row}")
            return None
        except Exception as e:
            #print(f"Error: {e} in row: {row}")
            return None

    def load_txt_files_to_dataframe(folder_path, text_column_name):
        #Get list of .txt files
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        # Initialize lists to store data
        file_names = []
        file_contents = []

        # Read each file
        for file in txt_files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Append data to lists
            file_names.append(os.path.splitext(file)[0])  # Remove .txt extension
            file_contents.append(content)

        # Create DataFrame
        df = pd.DataFrame({
            'file_name': file_names,
            text_column_name: file_contents
        })

        return df

    def load_and_join_texts_as_dataframe(folder_list):
        result_df = None

        for folder_path in folder_list:
            # Extract the folder name from the path
            folder_name = os.path.basename(folder_path)

            # Load the text files from this folder
            df = load_txt_files_to_dataframe(folder_path, folder_name)
            df[folder_name] = df[folder_name].apply(lambda x: parser.render(x))
            # Rename the 'content' column to the folder name
            #df = df.rename(columns={'content': folder_name})

            if result_df is None:
                result_df = df
            else:
                # Perform a left join with the existing result
                result_df = result_df.merge(df, on='file_name', how='left')

        return result_df
    return (
        MarkdownIt,
        RendererPlain,
        compute_metric,
        evaluate,
        load_and_join_texts_as_dataframe,
        load_txt_files_to_dataframe,
        metric_cer,
        metric_wer,
        parser,
    )


@app.cell
def __(data_folder, load_and_join_texts_as_dataframe, os):
    df = load_and_join_texts_as_dataframe([os.path.join(data_folder, 'BLN600', 'Ground Truth'),
                                               os.path.join(data_folder, 'BLN600', 'OCR Text'),
                                              os.path.join(data_folder, 'BLN600_deskew'),
                                              os.path.join(data_folder, 'BLN600_deskew_ratio_15'),
                                              os.path.join(data_folder, 'BLN600_ratio_1000'),
                                            os.path.join(data_folder, 'BLN600_ratio_15')

                                              ])
    return (df,)


@app.cell
def __(compute_metric, df, metric_cer, metric_wer):
    df['cer_ocr'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='OCR Text', reference_col='Ground Truth')
    df['wer_ocr'] = df.apply(compute_metric, axis=1, metric =metric_wer, prediction_col='OCR Text', reference_col='Ground Truth')

    df['cer_deskew_1000'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_deskew', reference_col='Ground Truth')


    df['cer_deskew_15'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_deskew_ratio_15', reference_col='Ground Truth')


    df['cer_nodeskew_1000'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_ratio_1000', reference_col='Ground Truth')

    df['cer_nodeskew_15'] = df.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='BLN600_ratio_15', reference_col='Ground Truth')
    return


@app.cell
def __(df):
    df[['file_name','cer_ocr', 'cer_deskew_1000', 'cer_deskew_15','cer_nodeskew_1000','cer_nodeskew_15',
       ]].describe()
    return


@app.cell
def __(df, np, plt, sns):
    _plot_df = df[['file_name','cer_ocr', 'cer_deskew_1000', 'cer_deskew_15','cer_nodeskew_1000','cer_nodeskew_15']].melt(id_vars = 'file_name')

    _plot_df['deskew'] = np.where(_plot_df['variable'].str.contains('no'),'no deskew', 'deskew' ) 

    _mapping = {
        'cer_ocr': 'none',
        'cer_deskew_1000': 'none',
        'cer_deskew_15': '1.5',
        'cer_nodeskew_1000': 'none',
        'cer_nodeskew_15': '1.5'
    }

    # Create a new column with the mapped values
    _plot_df['mapped_category'] = _plot_df['variable'].map(_mapping)

    # Create the histogram
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=_plot_df, x='mapped_category', y='value', hue = 'deskew')
    plt.title('Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.show()
    return


@app.cell
def __(df, np, pd, plt, sns):
    _plot_df = df[['file_name','cer_ocr','cer_deskew_15','cer_nodeskew_15']].melt(id_vars = 'file_name')

    _plot_df['deskew'] = np.where(_plot_df['variable'].str.contains('no'),'no deskew', 'deskew' ) 

    mapping = {
        'cer_ocr': 'OCR',
        'cer_deskew_1000': 'none',
        'cer_deskew_15': '1.5',
        'cer_nodeskew_1000': 'none',
        'cer_nodeskew_15': '1.5'
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

    plt.title('Distribution of Bootstrapped Means by Category')
    plt.xlabel('Category')
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


@app.cell
def __(mo):
    mo.md("""sns.scatterplot(data = df2, x = 'cer_diff_deskew', y = 'cer_diff_ratio')""")
    return


@app.cell
def __():
    first_text = s1= """Mrs. Elizabeth Elson, the servant alluded to, said: she found Mrs. Rawlings at a house in the Waterloo-road, and on telling her for what purpose she had come to town, she replied that unless she got her money, as she was in the habit of doing, from her husband, she should poison the children and herself. This threat she repeated, and was in a violent passion.

    Joseph Coster, one of the summonsing officers, said: that a warrant had been placed in his hands to apprehend Mrs. Rawlings, and he found her walking in the Waterloo-road. He told her he had a warrant for her apprehension, on a charge of threatening to murder her children. She replied that she did not consider what she said in the nature of a threat. What she said was, that if she did not get her money she should take she repeated that unless she got her money, she was in the habit of doing; from her husband, she should poison the children and herself. This threat she repeated, and was in a violent passion."""


    second_text= s2 = """Joseph Coster, one of the summonsing officers, said that a warrant had been placed in his hands to apprehend Mrs. Rawlings, and he found her walking in the Waterloo-road. He told her he had a warrant for her apprehension, on a charge of threatening to murder her children. She replied that she did not consider what she said in the nature of a threat. What she said was, that if she did not get her money she should take her children to the top of the house and fling them out of the window, and then jump out herself. She then asked him to go to the house where her children were. He did so, and that house was a common brothel.

    Mrs. Rawlings, whose tone, manner, and language, stamped her as a person of superior education, but whose countenance, as evidently once handsome, now showed the bloated and sodden appearance produced by excessive drink, said that she was not aware of the character of the house when she took lodgings there, as it was not in every place that a mother with three children could get a lodging. She admitted that she had got into a passion on hearing a threat from the old servant of the family to take away her children, leaving her destitute and unprovided for; and with that feeling she had used expressions which she did not mean. She declared she had no intention of leaving her children.
    """
    return first_text, s1, s2, second_text


@app.cell
def __(first_text, knit_strings, second_text):
    knit_strings(first_text, second_text)
    return


@app.cell
def __():
    import difflib

    def knit_strings2(s1: str, s2: str) -> str:
        """
        Knit two strings together based on their longest common substring.

        This function finds the longest common substring between s1 and s2,
        then combines them by keeping the text up to but excluding the matching
        substring from s1, and adding the remaining text from s2.

        Args:
            s1 (str): The first string.
            s2 (str): The second string.

        Returns:
            str: A new string combining s1 and s2, with proper merging at the common substring.
        """
        # Create a SequenceMatcher object to compare the two strings
        matcher = difflib.SequenceMatcher(None, s1, s2, autojunk=False)

        # Find the longest matching substring
        match = matcher.find_longest_match(0, len(s1), 0, len(s2))

        # If no match is found, simply concatenate the strings
        if match.size == 0:
            return s1 + s2

        # Take s1 up to but not including the match
        result = s1[:match.a]

        # Add everything from s2 that starts from the match
        result += s2[match.b:]

        return result
    return difflib, knit_strings2


@app.cell
def __(first_text, knit_strings2, second_text):
    knit_strings2(first_text, second_text)
    return


@app.cell
def __(difflib, s1, s2):
    matcher = difflib.SequenceMatcher(None, s1, s2, autojunk=False)

    # Find the longest matching substring
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    return match, matcher


@app.cell
def __(match):
    match
    return


@app.cell
def __(match, s1):
    s1[:match.a]
    return


@app.cell
def __(difflib):
    s12 = """Mrs. Elizabeth Elson, the servant alluded to, said: she found Mrs. Rawlings at a house in the Waterloo-road, and on telling her for what purpose she had come to town, she replied that unless she got her money, as she was in the habit of doing, from her husband, she should poison the children and herself. This threat she repeated, and was in a violent passion.

    Joseph Coster, one of the summonsing officers, said: that a warrant had been placed in his hands to apprehend Mrs. Rawlings, and he found her walking in the Waterloo-road. He told her he had a warrant for her apprehension, on a charge of threatening to murder her children. She replied that she did not consider what she said in the nature of a threat. What she said was, that if she did not get her money she should take she repeated that unless she got her money, she was in the habit of doing; from her husband, she should poison the children and herself. This threat she repeated, and was in a violent passion."""


    s22 = """Joseph Coster, one of the summonsing officers, said that a warrant had been placed in his hands to apprehend Mrs. Rawlings, and he found her walking in the Waterloo-road. He told her he had a warrant for her apprehension, on a charge of threatening to murder her children. She replied that she did not consider what she said in the nature of a threat. What she said was, that if she did not get her money she should take her children to the top of the house and fling them out of the window, and then jump out herself. She then asked him to go to the house where her children were. He did so, and that house was a common brothel."""

    matcher2 = difflib.SequenceMatcher(None, s12, s22)
    return matcher2, s12, s22


@app.cell
def __(matcher2, s12, s22):
    matcher2.find_longest_match(0, len(s12), 0, len(s22))
    return


@app.cell
def __(s12):
    s12[0:5]
    return


if __name__ == "__main__":
    app.run()
