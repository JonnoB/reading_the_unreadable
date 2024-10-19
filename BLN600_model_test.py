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


    from mistralai import Mistral
    load_dotenv()



    api_key = os.environ["MISTRAL_API_KEY"]
    model = "pixtral-12b-2409"

    client = Mistral(api_key=api_key)


    save_folder = 'data/BLN600_mistral'
    os.makedirs(save_folder, exist_ok=True)
    return (
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
        save_folder,
        scale_bbox,
        shutil,
        time,
    )


@app.cell
def __(mo):
    mo.md(
        """
        def convert_and_copy_images(source_folder, destination_folder):
            # Create the destination folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # Iterate through all files in the source folder
            for filename in os.listdir(source_folder):
                source_path = os.path.join(source_folder, filename)

                # Check if the file is an image
                if os.path.isfile(source_path):
                    # Get the file extension
                    _, extension = os.path.splitext(filename)

                    if extension.lower() == '.jpg' or extension.lower() == '.jpeg':
                        # If it's a JPG, simply copy it to the destination folder
                        destination_path = os.path.join(destination_folder, filename)
                        shutil.copy2(source_path, destination_path)

                    elif extension.lower() == '.tiff' or extension.lower() == '.tif':
                        # If it's a TIFF, convert it to JPG and save in the destination folder
                        try:
                            with Image.open(source_path) as img:
                                rgb_img = img.convert('RGB')
                                jpg_filename = os.path.splitext(filename)[0] + '.jpg'
                                destination_path = os.path.join(destination_folder, jpg_filename)
                                rgb_img.save(destination_path, 'JPEG')
                        except Exception as e:
                            print(f"Error converting {filename}: {str(e)}")

        # Usage
        source_folder = 'data/BLN600/Images'
        destination_folder = 'data/BLN600/Images_jpg'

        convert_and_copy_images(source_folder, destination_folder)
        """
    )
    return


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
):
    def split_image(image, desired_ratio=1.5, overlap_fraction=0.1):
        """
        Split an image into segments based on a desired aspect ratio.
        
        Args:
            image: PIL Image object
            desired_ratio (float): Desired width-to-height ratio
            overlap_fraction (float): Fraction of overlap between segments
        
        Returns:
            list: List of PIL Image objects (segments)
        """
        width, height = image.size
        current_ratio = width / height
        
        if current_ratio >= desired_ratio:
            return [image]  # No need to split
        
        segment_height = int(width / desired_ratio)
        overlap_height = int(segment_height * overlap_fraction)
        
        segments = []
        y = 0
        while y < height:
            bottom = min(y + segment_height, height)
            segment = image.crop((0, y, width, bottom))
            segments.append(segment)
            y = bottom - overlap_height
        
        return segments
    
    def split_image(image, max_ratio=1.5, overlap_fraction=0.1):
        """
        Split an image into segments based on a maximum aspect ratio.
        
        Args:
            image: PIL Image object
            max_ratio (float): Maximum width-to-height ratio before splitting
            overlap_fraction (float): Fraction of overlap between segments
        
        Returns:
            list: List of PIL Image objects (segments)
        """
        width, height = image.size
        current_ratio = width / height

        if current_ratio <= max_ratio:
            #print("No need to split the image.")
            return [image]
        
        segment_height = int(width / max_ratio)
        overlap_height = int(segment_height * overlap_fraction)

        
        segments = []
        y = 0
        while y < height:
            bottom = min(y + segment_height, height)
            segment = image.crop((0, y, width, bottom))
            segments.append(segment)
            y = bottom - overlap_height
            #print(f"Created segment: {segment.size}")
        
        return segments



    def process_jpeg_folder(folder_path, client, output_folder, max_ratio=1.5, overlap_fraction=0.1):
        os.makedirs(output_folder, exist_ok=True)

        log_file_path = os.path.join(output_folder, 'processing_log.csv')
        if os.path.exists(log_file_path):
            log_df = pd.read_csv(log_file_path)
        else:
            log_df = pd.DataFrame(columns=['file_name', 'processing_time', 'input_tokens', 'output_tokens', 'total_tokens', 'sub_images', 'status', 'timestamp'])

        jpeg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        for filename in tqdm(jpeg_files, desc="Processing images"):
            if filename in log_df[log_df['status'] == 'Success']['file_name'].values:
                continue

            file_path = os.path.join(folder_path, filename)
            start_time = time.time()

            try:
                img = Image.open(file_path)
                segments = split_image(img, max_ratio, overlap_fraction)

                content_list = []
                total_input_tokens = total_output_tokens = total_tokens = 0
                sub_images = len(segments)

                for segment in segments:
                    buffered = BytesIO()
                    segment.save(buffered, format="JPEG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    segment_content, usage = process_image_with_api(image_base64, client)
                    content_list.append(segment_content)

                    input_tokens, output_tokens, segment_total_tokens = usage
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_tokens += segment_total_tokens

                combined_content = knit_string_list(content_list)

                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(combined_content)

                processing_time = time.time() - start_time
                status = 'Success'

            except Exception as e:
                processing_time = time.time() - start_time
                status = 'Failed'
                total_input_tokens = total_output_tokens = total_tokens = sub_images = 0
                tqdm.write(f"Error processing {filename}: {str(e)}")

            finally:
                if 'img' in locals():
                    img.close()

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

            log_df = pd.concat([log_df[log_df['file_name'] != filename], log_entry], ignore_index=True)
            log_df.to_csv(log_file_path, index=False)

        return log_df
    
    import psutil
    import traceback

    def split_image(image, max_ratio=1.5, overlap_fraction=0.1, max_segments=10):
        """
        Split an image into segments based on a maximum aspect ratio.
        
        Args:
            image: PIL Image object
            max_ratio (float): Maximum width-to-height ratio before splitting
            overlap_fraction (float): Fraction of overlap between segments
            max_segments (int): Maximum number of segments to create
        
        Returns:
            list: List of PIL Image objects (segments)
        """
        width, height = image.size
        current_ratio = width / height
        
        print(f"Image dimensions: {width}x{height}")
        print(f"Current aspect ratio: {current_ratio:.2f}")
        print(f"Max ratio: {max_ratio}")
        
        if current_ratio <= max_ratio:
            print("No need to split the image.")
            return [image]
        
        segment_height = int(width / max_ratio)
        overlap_height = int(segment_height * overlap_fraction)
        
        print(f"Segment height: {segment_height}")
        print(f"Overlap height: {overlap_height}")
        
        segments = []
        y = 0
        while y < height and len(segments) < max_segments:
            bottom = min(y + segment_height, height)
            segment = image.crop((0, y, width, bottom))
            segments.append(segment)
            y = bottom - overlap_height
        
        print(f"Number of segments created: {len(segments)}")
        
        return segments
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB

    def process_jpeg_folder(folder_path, output_folder, max_ratio=1.5, overlap_fraction=0.1):
        os.makedirs(output_folder, exist_ok=True)

        log_file_path = os.path.join(output_folder, 'processing_log.csv')
        if os.path.exists(log_file_path):
            log_df = pd.read_csv(log_file_path)
        else:
            log_df = pd.DataFrame(columns=['file_name', 'processing_time', 'input_tokens', 'output_tokens', 'total_tokens', 'sub_images', 'status', 'timestamp'])

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                if filename in log_df[log_df['status'] == 'Success']['file_name'].values:
                    print(f"Skipping {filename} as it has already been processed successfully.")
                    continue

                file_path = os.path.join(folder_path, filename)
                start_time = time.time()

                print(f"\nStarting to process {filename}")
                print(f"Initial memory usage: {get_memory_usage():.2f} MB")

                try:
                    img = Image.open(file_path)
                    print(f"Memory usage after opening image: {get_memory_usage():.2f} MB")
                    
                    segments = split_image(img, max_ratio, overlap_fraction)
                    print(f"Memory usage after splitting image: {get_memory_usage():.2f} MB")

                    content_list = []
                    total_input_tokens = total_output_tokens = total_tokens = 0
                    sub_images = len(segments)

                    for i, segment in enumerate(segments):
                        print(f"\nProcessing segment {i+1}/{sub_images}")
                        print(f"Segment dimensions: {segment.size}")
                        buffered = BytesIO()
                        segment.save(buffered, format="JPEG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        print(f"Memory usage after encoding segment {i+1}: {get_memory_usage():.2f} MB")

                        try:
                            segment_content, usage = process_image_with_api(image_base64)
                            if segment_content is not None and usage is not None:
                                content_list.append(segment_content)
                                input_tokens, output_tokens, segment_total_tokens = usage
                                total_input_tokens += input_tokens
                                total_output_tokens += output_tokens
                                total_tokens += segment_total_tokens
                            else:
                                print(f"Skipping segment {i+1} due to API error")
                            print(f"Memory usage after API processing of segment {i+1}: {get_memory_usage():.2f} MB")
                        except Exception as e:
                            print(f"Error in process_image_with_api for segment {i+1}: {str(e)}")
                            print(f"Skipping segment {i+1}")

                    print("\nCombining content from all segments")
                    combined_content = knit_string_list(content_list)
                    print(f"Memory usage after combining content: {get_memory_usage():.2f} MB")

                    output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(combined_content)

                    processing_time = time.time() - start_time
                    status = 'Success'

                except Exception as e:
                    processing_time = time.time() - start_time
                    status = 'Failed'
                    total_input_tokens = total_output_tokens = total_tokens = sub_images = 0
                    print(f"Error processing {filename}: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error traceback: {traceback.format_exc()}")

                finally:
                    # Ensure the image is closed
                    if 'img' in locals():
                        img.close()

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

                log_df = pd.concat([log_df[log_df['file_name'] != filename], log_entry], ignore_index=True)
                log_df.to_csv(log_file_path, index=False)

                print(f"\nFinished processing {filename}")
                print(f"Final memory usage: {get_memory_usage():.2f} MB")
                print("--------------------")

        return log_df
    return process_jpeg_folder, split_image


@app.cell
def __(client, np, process_jpeg_folder):
    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                       # client = client, 
                        output_folder = 'data/BLN600_mistral_whole',
                         max_ratio=np.inf, overlap_fraction=0.1)
    return


@app.cell
def __(client, process_jpeg_folder):
    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                       # client = client, 
                        max_ratio=1.5,
                        output_folder = 'data/BLN600_mistral_ratio_15')
    return


@app.cell
def __(client, process_jpeg_folder):
    process_jpeg_folder(folder_path = 'data/BLN600/Images_jpg', 
                      #  client = client, 
                        max_ratio=1,
                        output_folder = 'data/BLN600_mistral_ratio_1')
    return


if __name__ == "__main__":
    app.run()
