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
import cv2

import base64
import difflib
import os
from datetime import datetime
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

import wand.image
import json
from PIL import Image
from io import BytesIO
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





def save_encoded_images(encoded_images, output_folder):
    """
    Save base64-encoded images to files.

    Parameters:
    -----------
    encoded_images : dict
        Dictionary with keys as image identifiers and values as base64-encoded PNG strings
    output_folder : str
        Path to folder where images should be saved

    Returns:
    --------
    list
        List of saved file paths
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    saved_paths = []

    for key, encoded_string in encoded_images.items():
        try:
            # Decode base64 string
            img_data = base64.b64decode(encoded_string)
            
            # Create file path
            file_path = os.path.join(output_folder, f"{key}.png")
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(img_data)
            
            saved_paths.append(file_path)
            
        except Exception as e:
            print(f"Error saving image {key}: {str(e)}")
            continue
    
    print(f"Saved {len(saved_paths)} images to {output_folder}")
    return saved_paths



def convert_to_pil(cv2_image):
    """Convert CV2 image to PIL image."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def deskew_image(pil_image):
    """Deskew a PIL image using Wand."""
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    with wand.image.Image(blob=buffer.getvalue()) as wand_img:
        wand_img.deskew(0.4 * wand_img.quantum_range)
        deskewed_buffer = BytesIO(wand_img.make_blob('PNG'))
        return Image.open(deskewed_buffer)

def pad_wide_image(cv2_image):
    """Pad wide images to make them square."""
    height, width = cv2_image.shape[:2]
    diff = width - height
    top_padding = diff // 2
    bottom_padding = diff - top_padding
    
    return cv2.copyMakeBorder(
        cv2_image,
        top_padding,
        bottom_padding,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

def encode_pil_image(pil_image, page_id, box_page_id, segment_num, image_class):
    """Encode PIL image to base64 and create dictionary entry."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    
    key = f"{page_id}_{box_page_id}_segment_{segment_num}"
    return key, {
        'image': base64.b64encode(buffered.getvalue()).decode('utf-8'),
        'class': image_class
    }

def process_single_box(image, row, height, width, max_ratio, overlap_fraction, deskew):
    """Process a single bounding box from an image."""
    try:
        # Extract coordinates
        x1 = max(0, int(row['x1']))
        y1 = max(0, int(row['y1']))
        x2 = min(width, int(row['x2']))
        y2 = min(height, int(row['y2']))

        if x2 <= x1 or y2 <= y1:
            print(f"Invalid coordinates for box {row['box_page_id']}")
            return {}

        # Crop and convert
        # Crop image to bounding box dimensions
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            print(f"Empty crop for box {row['box_page_id']}")
            return {}

        cropped_pil = convert_to_pil(cropped)
        is_figure = row.get('class') == 'figure'
        image_class = row.get('class', 'unknown')

        # Apply deskewing if needed
        if deskew and not is_figure:
            cropped_pil = deskew_image(cropped_pil)

        if is_figure:
            key, value = encode_pil_image(cropped_pil, row['page_id'], row['box_page_id'], 0, image_class)
            return {key: value}

        # Process based on ratio
        crop_height, crop_width = cropped.shape[:2]
        ratio = crop_height / crop_width
        
        if ratio > max_ratio:
            # Split tall images
            segments = split_image(cropped_pil, max_ratio, overlap_fraction)
            return {
                k: v for i, segment in enumerate(segments)
                for k, v in [encode_pil_image(segment, row['page_id'], row['box_page_id'], i, image_class)]
            }
        
        elif ratio < 1:
            # Pad wide images
            padded = pad_wide_image(cropped)
            padded_pil = convert_to_pil(padded)
            key, value = encode_pil_image(padded_pil, row['page_id'], row['box_page_id'], 0, image_class)
            return {key: value}
        
        else:
            # Process normal ratio images
            key, value = encode_pil_image(cropped_pil, row['page_id'], row['box_page_id'], 0, image_class)
            return {key: value}

    except Exception as e:
        print(f"Error processing box {row['box_page_id']}: {str(e)}")
        return {}

def crop_and_encode_boxes(df, images_folder, max_ratio=1.5, overlap_fraction=0.2, deskew=True):
    """
    Process and encode image regions defined by bounding boxes in the input DataFrame.

    This function handles three types of image regions:
    1. Normal regions (1 ≤ height/width ratio ≤ max_ratio): Processed as-is
    2. Tall regions (ratio > max_ratio): Split into overlapping segments
    3. Wide regions (ratio < 1): Padded with white borders to make square

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the following required columns:
        - filename: Name of the image file
        - page_id: Unique identifier for the page
        - box_page_id: Unique identifier for the box within the page
        - x1, y1, x2, y2: Bounding box coordinates
        - class: (optional) Classification of the image region

    images_folder : str
        Path to the directory containing the source images

    max_ratio : float, optional (default=1.5)
        Maximum allowed height-to-width ratio before splitting the image

    overlap_fraction : float, optional (default=0.2)
        Fraction of overlap between segments when splitting tall images

    deskew : bool, optional (default=True)
        Whether to apply deskewing correction to non-figure images

    Returns
    -------
    dict
        Dictionary where:
        - Keys: '{page_id}_{box_page_id}_segment_{i}'
          (i=0 for unsplit images, i=0,1,2,... for split images)
        - Values: Dict containing:
          - 'image': Base64-encoded PNG string
          - 'class': Classification label (if provided in input)

    Notes
    -----
    - Images classified as 'figure' are processed without modifications
    - All processing is done in memory without saving to disk
    - Invalid or empty crops are skipped with warning messages
    - All images are encoded in PNG format for quality preservation
    """
    encoded_images = {}

    for filename, group in df.groupby('filename'):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        height, width = image.shape[:2]

        for _, row in group.iterrows():
            encoded_images.update(
                process_single_box(image, row, height, width, max_ratio, overlap_fraction, deskew)
            )

    return encoded_images


    
def create_jsonl_content(encoded_images, prompt_dict):
    """
    Create JSONL content as a string for file upload.

    Parameters:
    encoded_images: dict with keys as image IDs and values as dicts containing 'image' (base64-encoded string)
                   and 'class' (string)
    prompt_dict: dict, mapping image classes to their specific prompts
                e.g., {'table': 'Describe this table', 'figure': 'Describe this figure'}
                If a class is not found in prompt_dict, uses default 'plain text' prompt

    Returns:
    str: The JSONL content as a string
    """
    jsonl_lines = []
    default_prompt = prompt_dict.get('plain text', 'Describe this text')

    for image_id, image_data in encoded_images.items():
        # Get the appropriate prompt based on the image class
        image_class = image_data.get('class', 'plain text')
        prompt = prompt_dict.get(image_class, default_prompt)

        entry = {
            "custom_id": image_id,
            "body": {
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{image_data['image']}"
                            }
                        ]
                    }
                ]
            }
        }
        jsonl_lines.append(json.dumps(entry))
    
    return "\n".join(jsonl_lines)


def create_batch_job(client, base_filename, encoded_images, prompt_dict, job_type = 'testing'):
    """
    Create a batch job and return job details including original filename information.
    Assumes only a single issue is being batched

    Args:
        client: Mistral client instance
        base_filename: string of desired filename, typically the issue_id
        encoded_images: Encoded image data
        prompt_dict: dict, mapping image classes to their specific prompts
                    e.g., {'table': 'Describe this table', 'figure': 'Describe this figure'}
                    If a class is not found in prompt_dict, uses default 'plain text' prompt

    Returns:
        tuple: (job_id, original_filename)
    """
    # Get the base filename from the issue
    temp_file_path = f"temp_{base_filename}.jsonl"
    target_filename = f"{base_filename}.jsonl"  # The final filename we want

    try:
        # Write the JSONL content to the temporary file
        jsonl_content = create_jsonl_content(encoded_images, prompt_dict)
        with open(temp_file_path, 'w') as f:
            f.write(jsonl_content)

        # Upload the file
        with open(temp_file_path, 'rb') as f:
            batch_data = client.files.upload(
                file={
                    "file_name": f"{base_filename}.jsonl",
                    "content": f
                },
                purpose="batch"
            )

        # Create the job with metadata including the target filename
        created_job = client.batch.jobs.create(
            input_files=[batch_data.id],
            model="pixtral-12b-2409",
            endpoint="/v1/chat/completions",
            metadata={
                "job_type": job_type,
                "target_filename": target_filename  # Store the target filename in metadata
            }
        )

        return created_job.id, target_filename

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def process_issues_to_jobs(bbox_df, images_folder, prompt_dict , client, output_file='data/processed_jobs.csv'):
    """
    Process periodical issues one at a time, creating batch jobs for OCR processing
    
    Parameters:
    bbox_df (DataFrame): DataFrame containing bounding box information
    images_folder (str): Path to folder containing images
    prompt_dict: dict, mapping image classes to their specific prompts
            e.g., {'table': 'Describe this table', 'figure': 'Describe this figure'}
            If a class is not found in prompt_dict, uses default 'plain text' prompt
    client: API client instance
    output_file (str): Path to CSV file storing job information
    Returns:
    DataFrame: DataFrame containing job_ids and target filenames
    """

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    
    if not isinstance(prompt_dict, dict) or 'plain text' not in prompt_dict:
        raise ValueError("prompt_dict must be a dictionary containing 'plain text' key")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check if output file exists and load it
    try:
        existing_jobs_df = pd.read_csv(output_file)
        print(f"Loaded existing jobs file with {len(existing_jobs_df)} records")
    except FileNotFoundError:
        existing_jobs_df = pd.DataFrame(columns=['issue', 'job_id', 'target_filename'])
        print("Creating new jobs tracking file")
    
    # Get unique issue_ids
    unique_issues = bbox_df['issue'].unique()
    
    # Counter for saving frequency
    processed_since_save = 0
    
    for issue_id in tqdm(unique_issues, desc="Processing issues"):
        # Filter bbox_df for current issue
        issue_df = bbox_df[bbox_df['issue'] == issue_id]
        
        # Check if this issue has already been processed
        if issue_id in existing_jobs_df['issue'].values:
            #print(f"Skipping issue {issue_id} - already processed")
            continue
        
        try:
            # Encode images for this issue
            encoded_images = crop_and_encode_boxes(
                df=issue_df,
                images_folder=images_folder,
                max_ratio=1,
                overlap_fraction=0.2,
                deskew=True
            )
            
            # Create batch job
            job_id, target_filename = create_batch_job(
                client, 
                issue_id, 
                encoded_images, 
                prompt_dict
            )
            
            # Add results to DataFrame
            new_row = pd.DataFrame({
                'issue': [issue_id],
                'job_id': [job_id],
                'target_filename': [target_filename]
            })
            existing_jobs_df = pd.concat([existing_jobs_df, new_row], ignore_index=True)
            
            # Save to file every 5 processed issues
            processed_since_save += 1
            if processed_since_save >= 5:
                existing_jobs_df.to_csv(output_file, index=False)
                processed_since_save = 0
                #print(f"Saved progress to {output_file}")
            
            #print(f"Successfully processed page_id {issue_id}")
            
        except Exception as e:
            print(f"Error processing page_id {issue_id}: {str(e)}")
            # Save progress in case of error
            existing_jobs_df.to_csv(output_file, index=False)
            continue
    
    # Final save
    existing_jobs_df.to_csv(output_file, index=False)
    print(f"Processing complete. Final results saved to {output_file}")
    
    return existing_jobs_df