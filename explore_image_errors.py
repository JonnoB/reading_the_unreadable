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
        time,
        tqdm,
        traceback,
    )


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
def __(Image, os, process_segment, split_image):

    # Define paths and parameters
    image_path = "data/BLN600/Images_jpg/3200810905.jpg"       # Path to the image file
    output_folder = "data/test_segment"           # Folder to save segment results
    max_ratio = 1                             # Maximum segment aspect ratio
    overlap_fraction = 0.1                      # Overlap fraction for splitting


    prompt = "You are an expert at transcription. The text is from a 19th century news article. Please transcribe exactly the text found in the image. Do not add any commentary. Do not use mark up please transcribe using plain text only."
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load the image
    img = Image.open(image_path)

    # Step 2: Split the image into segments
    segments = split_image(img, max_ratio, overlap_fraction)

    # Step 3: Process each segment and save results
    for i, segment in enumerate(segments):
        # Save the image segment for comparison
        segment_image_file = os.path.join(output_folder, f"segment_{i+1}.jpg")
        segment.save(segment_image_file, format="JPEG")
        print(f"Saved segment image {i+1} to {segment_image_file}")
        
        # Process the segment and retrieve content and token usage
        segment_content, usage = process_segment(segment, prompt)
        
        # If content was successfully processed, save it to a text file
        if segment_content:
            segment_text_file = os.path.join(output_folder, f"segment_{i+1}.txt")
            with open(segment_text_file, 'w', encoding='utf-8') as f:
                f.write(segment_content)
            print(f"Saved segment text {i+1} to {segment_text_file}")
        else:
            print(f"Segment {i+1} processing failed or returned empty content.")
    return (
        f,
        i,
        image_path,
        img,
        max_ratio,
        output_folder,
        overlap_fraction,
        prompt,
        segment,
        segment_content,
        segment_image_file,
        segment_text_file,
        segments,
        usage,
    )


@app.cell
def __(segments):
    len(segments)
    return


if __name__ == "__main__":
    app.run()
