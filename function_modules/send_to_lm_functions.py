""" 
This module contains the functions which take bounding box information and the images and sends them to the LLM, and retrives the data
This module contains a mixture of image manipulation and LLM api functions.
"""
import os
import pandas as pd
import datetime 
import time 
from tqdm import tqdm
import cv2

import base64
import difflib
from datetime import datetime
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

import wand.image
import json
from PIL import Image
from io import BytesIO
import requests


def convert_returned_streaming_to_dataframe(response, id=None, custom_id=None):
    """
    Process a single API response and return it in DataFrame format
    """
    extracted_data = [{
        'id': id,
        'custom_id': custom_id,
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens,
        'finish_reason': response.choices[0].finish_reason,
        'content': response.choices[0].message.content
    }]

    return pd.DataFrame(extracted_data)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_image_with_api(image_base64, prompt,  model="mistral/pixtral-12b-2409"):
    """
    Process an image using an AI model API to extract information based on a given prompt.

    This function sends a base64-encoded image along with a text prompt to an AI model API,
    which analyzes the image according to the provided prompt and returns a response.

    Args:
        image_base64 (str): A base64-encoded string representation of the image.
        prompt (str): The text prompt instructing the model how to analyze the image.
        model (str, optional): The name of the AI model to use. Defaults to "mistral/pixtral-12b-2409".

    Returns:
        litellm.ModelResponse: The complete response object from the API containing:
            - choices: List of completion choices
            - model: The model used
            - usage: Token usage statistics
            - system_fingerprint: Model system identifier

    Raises:
        Exception: If an error occurs during the API call or processing after all retries.
        The function will retry up to 3 times with exponential backoff (min=4s, max=10s).

    Note:
        This function is decorated with @retry to automatically retry failed attempts
        with exponential backoff between attempts.
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

        return response

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


def split_image(image, max_ratio=1.5, overlap_fraction=0.2, max_segments=20):
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



def combine_article_segments(df):
    """
    Process DataFrame by grouping and knitting text within groups, including segment counts and token information.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'issue_id', 'page_number', 
                             'block', 'column_number', 'reading_order', 'segment', 'text',
                             'prompt_tokens', 'completion_tokens', 'total_tokens'
    
    Returns:
        pandas.DataFrame: Processed DataFrame with knitted text and aggregated metrics
    """
    # Create a copy of the input DataFrame
    df = df.copy()
    
    # Sort the DataFrame by segment within groups
    df = df.sort_values(['issue_id', 'page_number', 'block', 'column', 
                        'reading_order', 'segment'])
    
    # Function to process each group
    def knit_group_texts(group):
        text_list = group['content'].tolist()
        knitted_text = knit_string_list(text_list)
        
        return pd.Series({
            'content': knitted_text,
            'segment_count': len(group),
            'prompt_tokens': group['prompt_tokens'].sum(),
            'completion_tokens': group['completion_tokens'].sum(),
            'total_tokens': group['total_tokens'].sum()
        })
    
    # Group and apply the knitting function
    result_df = (df.groupby(['issue_id', 'page_number', 'block', 'column', 'class',
                            'reading_order'])
                 .apply(knit_group_texts)
                 .reset_index())
    
    result_df['box_page_id'] = "B" + result_df['block'].astype(str) + "C"+result_df['column'].astype(str)  + "R" + result_df['reading_order'].astype(str) 
    result_df['page_id'] = result_df['issue_id'] + "_page_" + result_df['page_number'].astype(str)
    
    return result_df



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

def crop_image_to_bbox(image, bbox_dict, height, width):
    """
    Crops an image according to the bounding box coordinates provided in the input dictionary.

    Args:
        bbox_dict (dict): A dictionary containing the following keys:
            - 'x1': Left coordinate of the bounding box
            - 'y1': Top coordinate of the bounding box
            - 'x2': Right coordinate of the bounding box
            - 'y2': Bottom coordinate of the bounding box
            - 'box_page_id': Identifier for the bounding box

    Returns:
        dict: An empty dictionary if the crop operation fails (invalid coordinates or empty crop),
              or the cropped image data if successful.

    Note:
        -Typically the dictionar is a single row of a dataframe
        - The function assumes the existence of global variables 'image', 'width', and 'height'
        - Coordinates are clamped to image boundaries (0 to width/height)
        - Coordinates are converted to integers
        - Invalid coordinates (x2 ≤ x1 or y2 ≤ y1) result in an empty dictionary
    """
        # Extract coordinates
    x1 = max(0, int(bbox_dict['x1']))
    y1 = max(0, int(bbox_dict['y1']))
    x2 = min(width, int(bbox_dict['x2']))
    y2 = min(height, int(bbox_dict['y2']))

    if x2 <= x1 or y2 <= y1:
        print(f"Invalid coordinates for box {bbox_dict['box_page_id']}")
        return {}

    # Crop and convert
    # Crop image to bounding box dimensions
    cropped_image = image[y1:y2, x1:x2]
    if cropped_image.size == 0:
        print(f"Empty crop for box {bbox_dict['box_page_id']}")
        return {}

    return cropped_image

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
    """
    Encodes a PIL image to base64 format and creates a dictionary entry with image metadata.

    Args:
        pil_image (PIL.Image): The PIL Image object to be encoded.
        page_id (str): Identifier for the page containing the image.
        box_page_id (str): Identifier for the box/region within the page format B C R.
        segment_num (int): Segment number of the image referencing the cropped bounding box.
        image_class (str): Classification or category of the image.

    Returns:
        tuple: A tuple containing:
            - str: A unique key formatted as "{page_id}_{image_class}_{box_page_id}_segment_{segment_num}"
            - dict: A dictionary containing:
                - 'image': Base64 encoded string of the PNG image
                - 'class': The image classification
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    
    key = f"{page_id}_{image_class}_{box_page_id}_segment_{segment_num}"
    return key, {
        'image': base64.b64encode(buffered.getvalue()).decode('utf-8'),
        'class': image_class
    }

def process_single_box(image, row, height, width, max_ratio, overlap_fraction, deskew, crop_image = True):
    """
    Process a single bounding box from an image by cropping, deskewing, and encoding it based on specific criteria.

    Args:
        image (numpy.ndarray): The source image to process.
        row (dict): Dictionary containing box information with keys:
            - 'x1', 'y1', 'x2', 'y2': Coordinates of the bounding box
            - 'box_page_id': Unique identifier for the box
            - 'page_id': Page identifier
            - 'class': Image classification (optional)
        height (int): Height of the source image
        width (int): Width of the source image
        max_ratio (float): Maximum allowed height-to-width ratio before splitting
        overlap_fraction (float): Fraction of overlap between split segments
        deskew (bool): Whether to apply deskew correction to non-figure images
        crop_image (bool): Whether to crop the image, defaults to true

    Returns:
        dict: Dictionary mapping encoded image keys to their corresponding values.
              Returns empty dict if processing fails.
              Format: {
                  'key': 'encoded_image_value'
              }

    Processing Steps:
        1. Extracts and validates coordinates from the row
        2. Crops image to bounding box dimensions
        3. Converts to PIL image
        4. Applies deskewing if specified (except for figures)
        5. Processes image based on its aspect ratio:
           - Splits tall images (ratio > max_ratio)
           - Pads wide images (ratio < 1)
           - Processes normal ratio images directly

    Raises:
        No exceptions are raised; errors are caught and logged, returning empty dict

    Notes:
        if the image is not going to be cropped then the row data only needs to include box_page_id, 'page_id', 'class'.
    """
    try:
        # Crop image
        if crop_image:
            cropped = crop_image_to_bbox(image, row, height, width)
        else:
            cropped = image

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

def crop_and_encode_boxes(df, images_folder, max_ratio=1, overlap_fraction=0.2, deskew=True, crop_image = True):
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
                process_single_box(image, row, height, width, max_ratio, overlap_fraction, deskew, crop_image= crop_image)
            )

    return encoded_images


    
def create_jsonl_content(encoded_images, prompt_dict, max_tokens = 2000):
    """
    Create JSONL content as a string for file upload.

    Parameters:
    encoded_images: dict with keys as image IDs and values as dicts containing 'image' (base64-encoded string)
                   and 'class' (string)
    prompt_dict: dict, mapping image classes to their specific prompts
                e.g., {'table': 'Describe this table', 'figure': 'Describe this figure'}
                If a class is not found in prompt_dict, uses default 'text' prompt

    Returns:
    str: The JSONL content as a string
    """
    jsonl_lines = []
    default_prompt = prompt_dict.get('text', 'Describe this text')

    for image_id, image_data in encoded_images.items():
        # Get the appropriate prompt based on the image class
        image_class = image_data.get('class', 'text')
        prompt = prompt_dict.get(image_class, default_prompt)

        entry = {
            "custom_id": image_id,
            "body": {
                "max_tokens": max_tokens,
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

def create_batch_job(client, base_filename, encoded_images, prompt_dict, job_type='testing'):
    """
    Create a batch job and return job details including original filename information.
    Assumes only a single issue is being batched

    Args:
        client: Mistral client instance
        base_filename: string of desired filename, typically the issue_id
        encoded_images: Encoded image data
        prompt_dict: dict, mapping image classes to their specific prompts
                    e.g., {'table': 'Describe this table', 'figure': 'Describe this figure'}
                    If a class is not found in prompt_dict, uses default 'text' prompt

    Returns:
        tuple: (job_id, original_filename)
    """
    target_filename = f"{base_filename}.jsonl"  # The final filename we want

    # Create JSONL content
    jsonl_content = create_jsonl_content(encoded_images, prompt_dict)
    
    # Convert string content to bytes
    content_bytes = jsonl_content.encode('utf-8')
    
    # Upload the file using the buffer
    batch_data = client.files.upload(
        file={
            "file_name": target_filename,
            "content": content_bytes
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
            "target_filename": target_filename
        }
    )

    return created_job.id, target_filename


def process_issues_to_jobs(bbox_df, images_folder, prompt_dict , client, output_file='data/processed_jobs.csv', 
                           deskew = True, crop_image= True, max_ratio = 1):
    """
    Process periodical issues one at a time, creating batch jobs for OCR processing
    
    Parameters:
    bbox_df (DataFrame): DataFrame containing bounding box information
    images_folder (str): Path to folder containing images
    prompt_dict: dict, mapping image classes to their specific prompts
            e.g., {'table': 'Describe this table', 'figure': 'Describe this figure'}
            If a class is not found in prompt_dict, uses default 'text' prompt
    client: API client instance
    output_file (str): Path to CSV file storing job information
    Returns:
    DataFrame: DataFrame containing job_ids and target filenames
    """

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    
    if not isinstance(prompt_dict, dict) or 'text' not in prompt_dict:
        raise ValueError("prompt_dict must be a dictionary containing 'text' key")
    
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
                images_folder = images_folder,
                max_ratio = max_ratio,
                overlap_fraction=0.2,
                deskew = deskew,
                crop_image = crop_image
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


def process_mistral_responses(response):
    """
    Process the raw bytes data from Mistral API responses.

    Args:
        data (bytes): Raw response data from Mistral

    Returns:
        list: List of parsed JSON objects
    """

    data = response.read() 
    # Decode bytes to string
    text_data = data.decode('utf-8')

    # Split into individual JSON objects
    json_strings = text_data.strip().split('\n')

    parsed_responses = []
    for json_str in json_strings:
        try:
            parsed_response = json.loads(json_str)
            parsed_responses.append(parsed_response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            continue

    return parsed_responses

def download_processed_jobs(client, jobs_file='data/processed_jobs.csv', 
                          output_dir='data/downloaded_results', 
                          log_file='data/download_log.csv'):
    """
    Download results for all processed jobs listed in the jobs file with detailed logging.
    Downloads each job's output as a JSONL file using the target filename from the jobs file.
    
    Parameters:
    -----------
    client : object
        The API client instance
    jobs_file : str
        Path to the CSV file containing job information
    output_dir : str
        Directory where downloaded JSONL files will be stored
    log_file : str
        Path to the CSV file for logging download attempts and results
    
    Returns:
    --------
    dict
        Summary of download results
    """
    # Create output and log directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Initialize or load log file
    try:
        log_df = pd.read_csv(log_file)
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=[
            'issue', 'job_id', 'target_filename', 'status', 
            'download_time', 'attempt_count', 'error_message', 'timestamp'
        ])
    
    # Load jobs file
    try:
        jobs_df = pd.read_csv(jobs_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Jobs file not found: {jobs_file}")
    
    # Initialize counters
    results = {
        'total_jobs': len(jobs_df),
        'successful_downloads': 0,
        'failed_downloads': 0,
        'failed_jobs': []
    }
    
    # Process each job
    for _, row in tqdm(jobs_df.iterrows(), total=len(jobs_df), desc="Downloading job results"):
        issue = row['issue']
        job_id = row['job_id']
        target_filename = row['target_filename']
        output_path = os.path.join(output_dir, target_filename)
        
        # Check if file already exists
        if os.path.exists(output_path):
            results['successful_downloads'] += 1
            continue
        
        # Check if job was already successfully processed
        if len(log_df[(log_df['job_id'] == job_id) & (log_df['status'] == 'success')]) > 0:
            results['successful_downloads'] += 1
            continue
        
        start_time = time.time()
        try:
            # Get job status
            retrieved_job = client.batch.jobs.get(job_id=job_id)
            
            # Check if job is completed
            if retrieved_job.status == 'SUCCESS':
                try:
                    # Download and save the file
                    output_file = client.files.download(file_id=retrieved_job.output_file)
                    parsed_data = process_mistral_responses(output_file)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
                    
                    download_time = time.time() - start_time
                    results['successful_downloads'] += 1
                    
                    # Update log
                    new_log_entry = pd.DataFrame({
                        'issue': [issue],
                        'job_id': [job_id],
                        'target_filename': [target_filename],
                        'status': ['success'],
                        'download_time': [download_time],
                        'attempt_count': [1],
                        'error_message': [''],
                        'timestamp': [datetime.now()]
                    })
                    
                except Exception as e:
                    error_msg = f"Download failed: {str(e)}"
                    download_time = time.time() - start_time
                    results['failed_downloads'] += 1
                    results['failed_jobs'].append(job_id)
                    
                    new_log_entry = pd.DataFrame({
                        'issue': [issue],
                        'job_id': [job_id],
                        'target_filename': [target_filename],
                        'status': ['failed'],
                        'download_time': [download_time],
                        'attempt_count': [1],
                        'error_message': [error_msg],
                        'timestamp': [datetime.now()]
                    })
                    
            else:
                error_msg = f"Job status: {retrieved_job.status}"
                results['failed_downloads'] += 1
                results['failed_jobs'].append(job_id)
                
                new_log_entry = pd.DataFrame({
                    'issue': [issue],
                    'job_id': [job_id],
                    'target_filename': [target_filename],
                    'status': ['pending'],
                    'download_time': [0],
                    'attempt_count': [1],
                    'error_message': [error_msg],
                    'timestamp': [datetime.now()]
                })
            
        except Exception as e:
            error_msg = f"Job retrieval failed: {str(e)}"
            download_time = time.time() - start_time
            results['failed_downloads'] += 1
            results['failed_jobs'].append(job_id)
            
            new_log_entry = pd.DataFrame({
                'issue': [issue],
                'job_id': [job_id],
                'target_filename': [target_filename],
                'status': ['error'],
                'download_time': [download_time],
                'attempt_count': [1],
                'error_message': [error_msg],
                'timestamp': [datetime.now()]
            })
        
        # Update log file
        log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
        log_df.to_csv(log_file, index=False)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Print summary
    print("\nDownload Summary:")
    print(f"Total jobs: {results['total_jobs']}")
    print(f"Successfully downloaded: {results['successful_downloads']}")
    print(f"Failed downloads: {results['failed_downloads']}")
    print(f"Downloaded files saved in: {output_dir}")
    
    return results


def convert_returned_json_to_dataframe(json_data):

    """
    Turns the json of the returned data into a dataframe
    
    """
    extracted_data = []

    for item in json_data:
        row = {
            'id': item['id'],
            'custom_id': item['custom_id'],
            'prompt_tokens': item['response']['body']['usage']['prompt_tokens'],
            'completion_tokens': item['response']['body']['usage']['completion_tokens'],
            'total_tokens': item['response']['body']['usage']['total_tokens'],
            'finish_reason': item['response']['body']['choices'][0]['finish_reason'],
            'content': item['response']['body']['choices'][0]['message']['content']
        }
        extracted_data.append(row)

    return pd.DataFrame(extracted_data)


def parse_filename(filename):
    # Split by '_page_' first to get issue_id
    issue_id, rest = filename.split('_page_')

    # Split the rest by '_' to separate components
    parts = rest.split('_')

    # Get page number
    page_number = int(parts[0])

    class_info = parts[1]

    # Get position info (B0C1R2)
    position_info = parts[2]

    # Extract B, C, R components
    block = int(position_info.split('B')[1].split('C')[0])
    column = int(position_info.split('C')[1].split('R')[0])
    reading_order = int(position_info.split('R')[1])

    # Get segment number
    segment = int(parts[4])

    return {
        'issue_id': issue_id,
        'page_number': page_number,
        'class':class_info,
        'block': block,
        'column': column,
        'reading_order': reading_order,
        'segment': segment
    }

# Assuming your dataframe has a column named 'filename'
def decompose_filenames(df):

    """ 
    Separates the custom id back into issue, page number, block, column, reading_order, and segment.
    """
    # Apply the parsing function to each filename
    parsed = df['custom_id'].apply(parse_filename)

    # Convert the series of dictionaries to a dataframe
    parsed_df = pd.DataFrame(parsed.tolist())

    # Combine with original dataframe
    return pd.concat([df, parsed_df], axis=1)


def reassemble_issue_segments(jsonl_path):
    """ 
    This function takes the path to a json file returned from the mistral server and reconstructs it into 
    a meaningful dataframe. It does this by converting the json itself into a dataframe,
    then extracting the bounding box and segment information for each element of the json
    then merging all the segments into a single peice of text.
    """

    with open(jsonl_path, 'r') as file:
        json_data = json.load(file)  # Use load instead of loads

    # Create the DataFrame
    df = convert_returned_json_to_dataframe(json_data)
    df = decompose_filenames(df)
    df = combine_article_segments(df)

    return df


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