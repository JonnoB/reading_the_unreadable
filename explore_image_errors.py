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
    max_ratio = 1.5                             # Maximum segment aspect ratio
    overlap_fraction = 0.2                      # Overlap fraction for splitting


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
def __():
    s1 = """A CASH BOX ROBBERY.

    EDWARD WILSON, fifty-six, described as a cloth-
    worker, was charged, at the West London Police-
    court, with being concerned in stealing a cash-box.
    Harry Frapurd, clerk to Mr. Walter Charles Ems, an
    estate agent, of Lower Phillimore-place, Kensington,
    deposed that on the 27th of July a man came into
    the office for £l worth of silver, which he gave him,
    and he left. Witness locked the cash-box and placed
    it in the desk. Three minutes afterwards a cab
    drove up, and he was called out to the prisoner, who
    was sitting in it. He stated that he had a house of
    furniture to sell, and inquired if he could sell it.
    Witness said "Yes," and asked for the address. The
    prisoner gave an address in Russell-road. While
    witness was talking to him another man came out of
    the office. He came up and touched him (the witness)
    on the arm, saying, "I see you are engaged. I'll
    come back in a quarter of an hour." The man went
    away, and the prisoner asked him (the witness) to
    tell the cabman to drive on. Witness told the cab-
    man, who drove off. Witness was out all day, and
    inquired at the address in Russell-road, and found
    that there was no furniture to be sold. The next
    day he found the desk wrenched open and the cash-
    box, containing about £12s. in gold and silver,
    gone. On Thursday he identified the prisoner at
    the Kensington Police-station. Detective-sergeant
    Brogden said on Thursday he saw the prisoner and
    another man on Westminster Bridge. He addressed
    him as "Jemmy," and said he should take him into
    custody for being concerned in a number of till
    robberies committed in different parts of London.
    He denied all knowledge of them. In answer to the
    witness the prisoner said he had seen the prisoner"""


    s2 = """the Kensington Police-station. Mr. Brogden said on Thursday he saw the prisoner and another man on Westminster Bridge. He addressed him as "Jemmy," and said he should take him into custody for being concerned in a number of till robberies committed in different parts of London. He denied all knowledge of them. In answer to the prisoner the sergeant said he had seen the prisoner several times since July, but it did not suit him to take him. Mr. Hopkins reminded the prisoner, who had been in trouble before, and refused bail.

    ALL the charges for drunkenness at the West London Police-court, on Saturday, were against women. One gave the name of Mary O'Dell. It was stated that she fell down in St. Ann's-road, Notting-hill, while drunk, and cut her head. Mr. Plowden inquired her age, and she said sixty-nine. The assistant-goaler said that she was entered on the sheet as seventy-nine. Mr. Plowden: Women generally take ten years off their age. (Laughter.) The assistant-goaler said that the prisoner had been charged several times, and in different names. The prisoner promised the magistrate not to come before him again. Mr. Plowden ordered her to pay a fine of 5s. The Prisoner: God bless you! thank you. (Laughter.)

    A REPENTANT THIEF.—On Saturday the police of the N Division were informed that a jewellery traveller's bag, which was stolen early in the week, had been returned with contents untouched. The facts of the case are these:—Last Monday a gentleman-looking man called at a house at Canning-road, Highbury, where resided a traveller for a City jewellery firm. He said he had been sent for the traveller's sample bag, and told such a plausible story that he got possession of the bag and about £200 worth of jewellery. When the traveller returned to his home he discovered that his bag and samples had been unlawfully obtained, and he at once went to the Upper-street (Islington) Station and gave a description of him. While the police were"""
    return s1, s2


@app.cell
def __(s1, s2):
    import difflib
    matcher = difflib.SequenceMatcher(None, s1, s2, autojunk=False)

    # Find the longest matching substring
    # match.a: start index of match in s1
    # match.b: start index of match in s2
    # match.size: length of the match
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    return difflib, match, matcher


@app.cell
def __(difflib):
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
    return (knit_strings2,)


@app.cell
def __(match):
    match
    return


@app.cell
def __(match, s1):
    s1[(match.a):(match.a+49)]
    return


@app.cell
def __(match, s2):
    s2[(match.b):(match.b+49)]
    return


@app.cell
def __(knit_strings2, s1, s2):
    knit_strings2(s1, s2)
    return


if __name__ == "__main__":
    app.run()
