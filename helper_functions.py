import pandas as pd
import io
import base64
import math
import difflib
from pdf2image import convert_from_path
import ast
import time
import os
import logging
from datetime import datetime
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

import wand.image
from PIL import Image
from io import BytesIO
import logging
from tqdm import tqdm
import re
from markdown_it import MarkdownIt
from mdit_plain.renderer import RendererPlain

parser = MarkdownIt(renderer_cls=RendererPlain)


##
## The functions need an overhaul. for example the function 'process_page' takes a pdf converts it to jpg crops it and sends to api to extract text
## In addition convert_pdf_to_image converts a pdf to a jpg and stops there. These two processes are not compatible. 'process page' and the older
## Functions need to be rebuild or dropped to make a clear a modular process
##
## The processes should all start on the basis that the image is a jpg or PNG.

def create_page_dict(df):
    """
    Create a nested dictionary from a DataFrame containing page and article information.

    This function processes a DataFrame with columns 'page_id', 'id' (article_id), and 'bounding_box'.
    It creates a nested dictionary where the outer keys are page_ids and the inner keys are article_ids,
    with the corresponding bounding box information as values.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing columns 'page_id', 'id', and 'bounding_box'.
                           The 'bounding_box' column can contain either string representations of dictionaries
                           or actual dictionary objects.

    Returns:
    dict: A nested dictionary with the following structure:
          {
              page_id: {
                  article_id: bounding_box,
                  ...
              },
              ...
          }
          Where bounding_box is a dictionary containing the bounding box information for each article.

    Note:
    - The function uses ast.literal_eval to convert string representations of dictionaries to actual dictionaries
      if necessary.
    - If a page_id is encountered multiple times, all its articles will be grouped under the same page_id key.
    """
    # Initialize an empty dictionary to store the results
    page_dict = {}

    # Iterate through the DataFrame rows
    for _, row in df.iterrows():
        page_id = row['page_id']
        article_id = row['id']

        # Convert the bounding_box string to a dictionary if it's not already
        if isinstance(row['bounding_box'], str):
            bounding_box = ast.literal_eval(row['bounding_box'])
        else:
            bounding_box = row['bounding_box']

        # If the page_id is not in the dictionary, add it
        if page_id not in page_dict:
            page_dict[page_id] = {}

        # Add the article_id and bounding_box to the page's dictionary
        page_dict[page_id][article_id] = bounding_box

    return page_dict

def scale_bbox(bbox, original_size, new_size):
    '''
    Scale the bounding box from the original image size to the new image size.

    :param bbox: List of [x1, y1, x2, y2] coordinates of the bounding box
    :param original_size: Tuple of (width, height) of the original image
    :param new_size: Tuple of (width, height) of the new image
    :return: Scaled bounding box coordinates
    '''
    original_width, original_height = original_size
    new_width, new_height = new_size

    # Calculate scale factors for width and height
    width_scale = new_width / original_width
    height_scale = new_height / original_height

    # Scale the bounding box coordinates
    x1, y1, x2, y2 = bbox
    new_x1 = int(x1 * width_scale)
    new_y1 = int(y1 * height_scale)
    new_x2 = int(x2 * width_scale)
    new_y2 = int(y2 * height_scale)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_and_encode_image(page, x0, y0, x1, y1):
    """
    Crop an image and encode it as a base64 string.

    This function takes an image, crops it according to the specified coordinates,
    and then encodes the cropped image as a base64 string.

    Parameters:
    page (PIL.Image.Image): The source image to be cropped.
    x0 (int): The x-coordinate of the top-left corner of the crop area.
    y0 (int): The y-coordinate of the top-left corner of the crop area.
    x1 (int): The x-coordinate of the bottom-right corner of the crop area.
    y1 (int): The y-coordinate of the bottom-right corner of the crop area.

    Returns:
    str: A base64-encoded string representation of the cropped image in PNG format.

    Note:
    - The function assumes that the necessary modules (io, base64) are imported.
    - The cropped image is saved in PNG format before encoding.
    """
    cropped_image = page.crop((x0, y0, x1, y1))
    buffered = io.BytesIO()
    cropped_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def split_tall_box(page, x0, y0, x1, y1, max_height, overlap):
    """

    This function should be deleted when 'process_bounding_box' is deleted

    Split a tall box from a page into multiple smaller boxes of a specified maximum height.

    This function takes a tall box defined by its coordinates and splits it into
    multiple smaller boxes, each with a maximum height. The resulting boxes can
    overlap to ensure no content is lost at the splits.

    Args:
        page: The page object containing the image to be split.
        x0 (float): The left x-coordinate of the box.
        y0 (float): The bottom y-coordinate of the box.
        x1 (float): The right x-coordinate of the box.
        y1 (float): The top y-coordinate of the box.
        max_height (float): The maximum height for each split box.
        overlap (float): The amount of vertical overlap between adjacent split boxes.

    Returns:
        list: A list of encoded image data for each split box.

    Note:
        This function assumes the existence of a `crop_and_encode_image` function
        that takes the page and box coordinates as arguments and returns the
        encoded image data.
    """
    height = y1 - y0
    num_boxes = math.ceil((height - overlap) / (max_height - overlap))
    
    split_images = []
    for i in range(num_boxes):
        box_y0 = y0 + i * (max_height - overlap)
        box_y1 = min(y1, box_y0 + max_height)
        split_images.append(crop_and_encode_image(page, x0, box_y0, x1, box_y1))
    
    return split_images

def process_bounding_box(page, key, coords, original_size, page_size):

    """
    Process a bounding box on a page, scaling coordinates and handling tall boxes.

    This function takes a bounding box, scales its coordinates to match the current page size,
    ensures the coordinates are within the page bounds, and handles tall boxes by splitting them
    if necessary.

    Args:
        page (object): The page object containing the image.
        key (str): The identifier for the bounding box.
        coords (dict): A dictionary containing the original coordinates (x0, y0, x1, y1).
        original_size (tuple): The original size of the page (width, height).
        page_size (tuple): The current size of the page (width, height).

    Returns:
        dict: A dictionary with the key and a list of processed images.
              If the box is tall, it will contain multiple split images.
              Otherwise, it will contain a single cropped image.

    Note:
        This function assumes the existence of helper functions:
        - scale_bbox: for scaling the bounding box coordinates
        - split_tall_box: for splitting tall boxes into smaller parts
        - crop_and_encode_image: for cropping and encoding the image
    """
    x0, y0, x1, y1 = scale_bbox([coords["x0"], coords["y0"], coords["x1"], coords["y1"]],
                                original_size, page_size)
    
    # Ensure the coordinates are within the image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(page.width, x1), min(page.height, y1)
    
    width = x1 - x0
    height = y1 - y0
    
    if height > 1.5 * width:
        max_height = int(1.5 * width)
        overlap = int(0.1 * width)
        images = split_tall_box(page, x0, y0, x1, y1, max_height, overlap)
        return {key: images}
    else:
        image = crop_and_encode_image(page, x0, y0, x1, y1)
        return {key: [image]}

def crop_and_encode_images(page, bounding_boxes, original_size, page_size):
    """
    Crop and encode multiple images from a page based on given bounding boxes.

    This function processes multiple bounding boxes on a single page, cropping
    and encoding each specified area. It handles scaling of coordinates and
    special processing for tall boxes.

    Args:
        page (PIL.Image.Image): The source image to be cropped.
        bounding_boxes (dict): A dictionary where keys are identifiers and values
                               are dictionaries containing bounding box coordinates
                               (x0, y0, x1, y1).
        original_size (tuple): The original size of the page as (width, height).
        page_size (tuple): The current size of the page as (width, height).

    Returns:
        dict: A dictionary where keys are the original identifiers and values are
              lists of base64-encoded strings representing the cropped images.
              For tall boxes, the list may contain multiple images.

    Note:
        This function relies on the `process_bounding_box` helper function to
        handle individual bounding boxes, including scaling, boundary checks,
        and tall box splitting.
    """
    result = {}
    for key, coords in bounding_boxes.items():
        result.update(process_bounding_box(page, key, coords, original_size, page_size))
    return result




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




def process_page(row, image_drive, page_dict, client, save_folder, dataset_df, log_df):
    """
    Process a single page of a newspaper, extracting and saving article content.

    This function handles the entire process of converting a PDF page to an image,
    cropping articles based on bounding boxes, extracting text from these images
    using an API, and saving the processed content. It also logs the processing
    status and time.

    Parameters:
    row (pandas.Series): A row from the dataset containing page information.
    image_drive (str): Path to the directory containing the PDF files.
    page_dict (dict): A dictionary mapping page IDs to article bounding boxes.
    client: The API client used for text extraction.
    save_folder (str): Path to the directory where processed articles will be saved.
    dataset_df (pandas.DataFrame): The dataset containing article information.
    log_df (pandas.DataFrame): A DataFrame to log processing status and time.

    Returns:
    pandas.DataFrame: The updated log DataFrame with new entries for the processed page.

    Notes:
    - The function handles exceptions and logs errors if they occur during processing.
    - It uses helper functions like `convert_from_path`, `crop_and_encode_images`,
      and `process_articles` to perform specific tasks.
    - The processing time for each page is recorded and logged.
    """
    start_time = time.time()
    
    try:
        file_path = os.path.join(image_drive, row['folder_path'], row['file_name'])
        all_pages = convert_from_path(file_path, dpi=300)
        
        page = all_pages[row['page_number'] - 1].copy()
        bounding_boxes = page_dict[str(row['page_id'])]
        
        cropped_images = crop_and_encode_images(
            page,
            bounding_boxes,
            (row['width'], row['height']),
            page.size
        )
        
        process_articles(cropped_images, client, save_folder, dataset_df)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Log successful processing
        new_row = pd.DataFrame({
            'page_id': [row['page_id']],
            'status': ['Success'],
            'processing_time': [processing_time],
            'timestamp': [pd.Timestamp.now()]
        })
        log_df = pd.concat([log_df, new_row], ignore_index=True)
        
        print(f"Successfully processed page {row['page_id']} in {processing_time:.2f} seconds.")
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        logging.error(f"Error processing page {row['page_id']}: {str(e)}")
        print(f"Error processing page {row['page_id']}. See log for details.")
        
        # Log failed processing
        new_row = pd.DataFrame({
            'page_id': [row['page_id']],
            'status': ['Failed'],
            'processing_time': [processing_time],
            'timestamp': [pd.Timestamp.now()],
            'error_message': [str(e)]
        })
        log_df = pd.concat([log_df, new_row], ignore_index=True)
    
    return log_df


def process_articles(cropped_images, client, save_folder, dataset_df):
    """
    Process a set of cropped images representing articles, extract text from them, and save the results.

    This function iterates through a dictionary of cropped images, where each key is an article ID
    and the value is a list of image strings. For each article, it extracts text from the images,
    combines the extracted text, and saves it to a file.

    Args:
        cropped_images (dict): A dictionary where keys are article IDs and values are lists of
                               base64-encoded image strings.
        client: An API client object used for text extraction from images.
        save_folder (str): The path to the folder where processed article texts will be saved.
        dataset_df (pandas.DataFrame): A DataFrame containing metadata about the articles,
                                       including the 'id' and 'save_name' columns.

    Returns:
        None

    Side effects:
        - Saves extracted text for each article to a file in the specified save_folder.
        - Prints processing status messages to the console.
        - Logs any errors encountered during processing.

    Note:
        This function relies on helper functions:
        - extract_text_from_images: to process image strings and extract text
        - knit_string_list: to combine extracted text from multiple images
        - save_article_text: to save the final text to a file
    """
    for article_id, image_string in cropped_images.items():
        try:
            content_list, usage_list = extract_text_from_images(image_string, client)
            full_string = knit_string_list(content_list)
            save_article_text(article_id, full_string, save_folder, dataset_df)
            print(f"Processed article {article_id}")
        except Exception as e:
            logging.error(f"Error processing article {article_id}: {str(e)}")
            print(f"Error processing article {article_id}. See log for details.")

def extract_text_from_images(image_string, client):
    """
    Extract text content from a list of base64-encoded images using an API client.

    This function processes each image in the provided list, sending it to an API
    for text extraction. It collects the extracted content and usage information
    for each successful API call.

    Args:
        image_string (list): A list of base64-encoded image strings.
        client: An API client object used to make requests to the text extraction service.

    Returns:
        tuple: A tuple containing two lists:
            - content_list (list): A list of extracted text content from each image.
            - usage_list (list): A list of API usage information for each successful call.

    Raises:
        Exception: Logs any errors that occur during the processing of individual images.

    Note:
        - This function uses the `process_image_with_api` helper function to handle
          individual API calls.
        - Any errors during processing of individual images are logged but do not
          stop the processing of subsequent images.
    """
    content_list = []
    usage_list = []
    for i, image in enumerate(image_string):
        try:
            content, usage = process_image_with_api(image, client, model="pixtral-12b-2409")
            if content is not None:
                content_list.append(content)
                usage_list.append(usage)
        except Exception as e:
            logging.error(f"Error processing image {i}: {str(e)}")
    return content_list, usage_list

def save_article_text(article_id, full_string, save_folder, dataset_df):
    """
    Save the extracted text of an article to a file.

    This function takes the processed text of an article and saves it to a file
    in the specified save folder. The file name is determined from the dataset
    DataFrame based on the article ID.

    Parameters:
    article_id (str or int): The unique identifier of the article.
    full_string (str): The complete extracted text of the article.
    save_folder (str): The path to the folder where the article text will be saved.
    dataset_df (pandas.DataFrame): A DataFrame containing article metadata,
                                   including the 'id' and 'save_name' columns.

    Raises:
    Exception: If there's an error during the file saving process, which is
               logged using the logging module.

    Note:
    - The function assumes that the 'id' column in dataset_df contains integer values.
    - Any errors during the saving process are logged but not raised, allowing
      the program to continue processing other articles.
    """
    try:
        file_name = dataset_df.loc[dataset_df['id'] == int(article_id), 'save_name'].iloc[0]
        with open(os.path.join(save_folder, file_name), "w") as file:
            file.write(full_string)
    except Exception as e:
        logging.error(f"Error saving article {article_id}: {str(e)}")



def convert_pdf_to_image(pdf_path, output_folder='output_images', dpi=300, image_format='JPEG', use_greyscale=True, quality=85):
    """
    Converts each page of a PDF file into an image and saves the images to an output folder.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to be converted.
    output_folder : str, optional
        Directory where the output images will be saved. Defaults to 'output_images'.
    dpi : int, optional
        Resolution for the converted images, in dots per inch. Higher values increase image quality.
    image_format : str, optional
        Image format for the output files. Supported formats are 'PNG' and 'JPEG'. Defaults to 'PNG'.
    use_greyscale:
        Whether to save as a 1 channel greyscale image, this reduces file size by about 66%. Default is True

    Raises
    ------
    ValueError
        If an unsupported image format is specified.

    Notes
    ------
    The function names each image file with the original PDF filename, followed by '_page_X', 
    where X is the page number.

    Examples
    --------
    >>> convert_pdf_to_image('example.pdf', output_folder='images', dpi=200, image_format='JPEG')
    """
    os.makedirs(output_folder, exist_ok=True)
    original_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, dpi=dpi)

    format_map = {'PNG': 'png', 'JPEG': 'jpg'}
    file_extension = format_map.get(image_format.upper())
    if not file_extension:
        raise ValueError("Unsupported format. Use 'PNG' or 'JPEG'.")

    for i, image in enumerate(images):
        output_file = os.path.join(output_folder, f'{original_filename}_page_{i + 1}.{file_extension}')
        
        if use_greyscale:
            # Convert to grayscale
            image = image.convert('L')
            
            # For black and white text documents, you can often binarize the image
            # This converts the image to pure black and white
            threshold = 200  # Adjust this value based on your images
            image = image.point(lambda x: 0 if x < threshold else 255, '1')

            
        if image_format.upper() == 'JPEG':
            # For JPEG, we can use quality parameter
            image.save(output_file, image_format.upper(), quality=quality, optimize=True)
        else:
            # For PNG, use optimize
            image.save(output_file, image_format.upper(), optimize=True, compression = 9)


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

    Args:
        file_path (str): Path to the image file to be loaded.
        deskew (bool): Whether to apply deskewing to the image.
        output_folder (str): Path to store temporary files during deskewing.

    Returns:
        PIL.Image: Loaded (and potentially deskewed) image object.
    """
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
    Process all JPEG images in a folder and generate text output.

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

def compute_metric(row, metric, prediction_col, reference_col):
    """
    Computes evaluation metrics between prediction and reference texts in a dataframe row.

    Args:
        row (pandas.Series): A row from a DataFrame containing the texts to compare
        metric (evaluate.EvaluationModule): The evaluation metric to compute (e.g., BLEU, ROUGE)
        prediction_col (str): Name of the column containing the prediction text
        reference_col (str): Name of the column containing the reference text

    Returns:
        dict or None: Dictionary containing computed metric scores, or None if computation fails

    Raises:
        KeyError: If specified columns are not found in the row
        Exception: For other computation errors
    """
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

##
## These are for testing the results however they clash with previous
##
def load_txt_files_to_dataframe(folder_path, text_column_name):
    """
    Loads all .txt files from a specified folder into a pandas DataFrame.
    This is an internal function used by `load_and_join_texts_as_dataframe` which is used in
    calculating the cer for various models, and `files_to_df_func`, which is used as part of 
    the general organising of data... they could possibly be merged at some point.

    Args:
        folder_path (str): Path to the folder containing .txt files
        text_column_name (str): Name to be used for the column containing file contents

    Returns:
        pandas.DataFrame: DataFrame with columns:
            - file_name (str): Name of the file without extension
            - text_column_name (str): Contents of the text file

    Note:
        Files are read using UTF-8 encoding
    """
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
    """
    Loads text files from multiple folders and joins them into a single DataFrame.

    Args:
        folder_list (list): List of folder paths containing text files to process

    Returns:
        pandas.DataFrame: Combined DataFrame where:
            - Each folder's contents are in a separate column named after the folder
            - Files are matched across folders using their names
            - The 'file_name' column serves as the joining key
            - Text contents are processed using parser.render()

    Note:
        Performs left joins, keeping all files from the first folder in the list
    """
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

def files_to_df_func(folder_path, text_column_name = 'content'):

    """
    Convert text files from a folder into a structured pandas DataFrame.

    Parameters:
        folder_path (str): Path to the directory containing text files
        text_column_name (str, optional): Name of the column that will store file contents. 
            Defaults to 'content'.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - file_name: Original filename
            - content: Text content of the file
            - artid: Article identifier extracted from filename
            - periodical: Publication name extracted from filename
            - page_number: Page number extracted from filename (as integer)
            - issue: Issue number extracted from filename

    Notes:
        - Expects files to follow a specific naming convention with elements separated by underscores
        - File names should include article ID, periodical name, issue number, and page number
        - Example filename format: "prefix_artid_something_periodical_issue_X_page_Y.txt"

    Example:
        >>> df = files_to_df_func("path/to/files")
        >>> print(df.columns)
        ['file_name', 'content', 'artid', 'periodical', 'page_number', 'issue']
    """

    # Create a DataFrame from the data list
    df =  load_txt_files_to_dataframe(folder_path, text_column_name)

    split_df = df['file_name'].str.split("_")

    df['artid'] = split_df.apply(lambda x: x[1])
    df['periodical'] = split_df.apply(lambda x: x[3])
    df['page_number'] = split_df.apply(lambda x: x[-1]).str.replace(".txt", "").astype(int)
    df['issue'] = df['file_name'].str.extract(r'issue_(.*?)_page', expand=False)

    return df