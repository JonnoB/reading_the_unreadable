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

import logging

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




def knit_strings2(s1: str, s2: str) -> str:
    """
    Knit two strings together based on their longest common substring.

    This function finds the longest common substring between s1 and s2,
    then combines them by appending the non-overlapping part of s2 to s1.
    If no common substring is found, it simply concatenates the two strings.


    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        str: A new string combining s1 and s2, with the overlapping part appearing only once.

    Example:
        >>> knit_strings("Hello world", "world of Python")
        'Hello world of Python'
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