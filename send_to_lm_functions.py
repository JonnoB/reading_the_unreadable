""" 
This module contains the functions which take bounding box information and the images and sends them to the LLM
This module contains a mixture of image manipulation and LLM api functions.
"""
import os
import pandas as pd
import numpy as np
import datetime 
import time 
from tqdm import tqdm

import base64
import difflib
import time
import os
from datetime import datetime
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

import wand.image
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import re
from numba import jit

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_image_with_api(image_base64, prompt,  model="mistral/pixtral-12b-2409"):
    """
    Process an image using an AI model API to extract information from it.

    This function sends a base64-encoded image to an AI model API, which analyzes
    the image (assumed to be a scan of a 19th-century English newspaper) and extracts
    relevant information from it, including text, lists, tables, and image descriptions.

    Args:
        image_base64 (str): A base64-encoded string representation of the image.
        model (str, optional): The name of the AI model to use. Defaults to "pixtral-12b-2409".

    Returns:
        tuple: A tuple containing two elements:
            - content (str): The extracted information from the image as text.
            - usage (tuple): A tuple of (prompt_tokens, completion_tokens, total_tokens)
              representing the API usage statistics.

    Raises:
        Exception: If an error occurs during the API call or processing after all retries.
    """
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ]
        )

        content = response.choices[0].message.content
        usage = response.usage
        return content, (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)

    except Exception as e:
        print(f"An error occurred in process_image_with_api: {str(e)}")
        raise  # Re-raise the exception to trigger a retry


def knit_strings(s1: str, s2: str) -> str:
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
    # match.a: start index of match in s1
    # match.b: start index of match in s2
    # match.size: length of the match
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    
    # If no match is found (match.size == 0), simply concatenate the strings
    if match.size == 0:
        return s1 + s2

    # Take s1 up to but not including the match
    result = s1[:match.a]

    # Add everything from s2 that starts from the match
    result += s2[match.b:]
    
    return result


def knit_string_list(content_list: list) -> str:
    """
    Knit a list of strings together based on their longest common substrings.

    This function iteratively applies the knit_strings function to a list of strings,
    combining them based on their longest common substrings.

    Args:
        content_list (list): A list of strings to be knitted together.

    Returns:
        str: A new string combining all strings in the list.

    Example:
        >>> knit_string_list(["Hello world", "world of Python", "Python is great"])
        'Hello world of Python is great'
    """
    if not content_list:
        return ""
    
    result = content_list[0]
    for i in range(1, len(content_list)):
        result = knit_strings(result, content_list[i])
    
    return result


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

def initialize_log_file(output_folder):
    """
    Initialize or load an existing log file for tracking image processing results.

    Args:
        output_folder (str): Path to the folder where the log file will be stored.

    Returns:
        tuple: A tuple containing:
            - str: Path to the log file
            - pandas.DataFrame: DataFrame containing the log entries
    """
    log_file_path = os.path.join(output_folder, 'processing_log.csv')
    if os.path.exists(log_file_path):
        log_df = pd.read_csv(log_file_path)
    else:
        log_df = pd.DataFrame(columns=['file_name', 'processing_time', 'input_tokens', 'output_tokens', 'total_tokens', 'sub_images', 'status', 'timestamp'])
    return log_file_path, log_df

def load_image(file_path, deskew, output_folder):
    """
    Load and optionally deskew an image file.
    This function was originally created to handle images that were a 
    single image only.

    Args:
        file_path (str): Path to the image file to be loaded.
        deskew (bool): Whether to apply deskewing to the image.
        output_folder (str): Path to store temporary files during deskewing.

    Returns:
        PIL.Image: Loaded (and potentially deskewed) image object.
    """
    if deskew:
        with wand.image.Image(filename=file_path) as wand_img:

            #The save and load is annoying but necessary as it isn't possible to convert between wand and PIL
            #Maybe can be re-done using the deskew library?
            wand_img.deskew(0.4 * wand_img.quantum_range)
            temp_path = os.path.join(output_folder, f"temp_deskewed_{os.path.basename(file_path)}")
            wand_img.save(filename=temp_path)
            img = Image.open(temp_path)
            os.remove(temp_path)
    else:
        img = Image.open(file_path)
    return img

def process_image_segments(segments, prompt):
    """
    Process multiple image segments and aggregate their results.

    Args:
        segments (list): List of image segments to process.
        prompt (str): The prompt to use for image processing.

    Returns:
        tuple: A tuple containing:
            - list: Processed content for each segment
            - int: Total input tokens used
            - int: Total output tokens used
            - int: Total tokens used
            - int: Number of segments processed
    """
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
    """
    Process a single image segment using the API.

    Args:
        segment (PIL.Image): Image segment to process.
        prompt (str): The prompt to use for image processing.

    Returns:
        tuple: A tuple containing:
            - str: Processed content from the segment
            - tuple or None: Token usage statistics (input_tokens, output_tokens, total_tokens)
    """
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
    """
    Save processed text content to a file.

    Args:
        output_folder (str): Path to the output directory.
        filename (str): Name of the original image file.
        content_list (list): List of processed text content to save.
    """
    combined_content = knit_string_list(content_list)
    output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_content)


def update_log(log_df, filename, processing_time, total_input_tokens, total_output_tokens, total_tokens, sub_images, status):
    """
    Update the processing log with new entry information.

    Args:
        log_df (pandas.DataFrame): Existing log DataFrame.
        filename (str): Name of the processed file.
        processing_time (float): Time taken to process the image.
        total_input_tokens (int): Number of input tokens used.
        total_output_tokens (int): Number of output tokens used.
        total_tokens (int): Total number of tokens used.
        sub_images (int): Number of image segments processed.
        status (str): Processing status ('Success' or 'Failed').

    Returns:
        pandas.DataFrame: Updated log DataFrame.
    """
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
    """
    This functions images one at a time. It needs to be replaced with a function that can send to the batch API.
    It is only here to be replaced

    Process all JPEG images in a folder and generate text output. Assumes that the articles have already 
    been clipped. Sends each image to the server one at a time.

    Args:
        folder_path (str): Path to the folder containing JPEG images.
        output_folder (str): Path to store processed outputs and logs.
        prompt (str): The prompt to use for image processing.
        max_ratio (float, optional): Maximum aspect ratio for image segments. Defaults to 1.5.
        overlap_fraction (float, optional): Fraction of overlap between segments. Defaults to 0.1.
        deskew (bool, optional): Whether to apply deskewing to images. Defaults to True.

    Returns:
        pandas.DataFrame: Complete log of all processed files.
    """
    os.makedirs(output_folder, exist_ok=True)
    log_file_path, log_df = initialize_log_file(output_folder)

    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename not in log_df[log_df['status'] == 'Success']['file_name'].values:
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