"""
Bounding Box Processing Module

This module contains helper functions for preprocessing and manipulating bounding boxes
in document layout analysis for historical newspapers. The functions are primarily designed to work with DocLayout-yolo
output but may be compatible with other YOLO models.

Key Functions:
-------------
numba_fill_count : Function
    Optimized function for counting overlapping boxes using Numba.

sum_of_area : Function
    Calculates the total area covered by bounding boxes.

calculate_article_coverage : Function
    Computes pixel coverage statistics for articles on a page.

calculate_coverage_and_overlap : Function
    Processes entire dataset to calculate coverage and overlap metrics.

print_area_meta : Function
    Calculates metadata about printed page areas.

reclassify_abandon_boxes : Function
    Reclassifies boxes based on their position relative to abandon boxes.

assign_columns : Function
    Assigns column numbers and edges to bounding boxes.

create_page_blocks : Function
    Creates page blocks based on titles and content organization.

create_reading_order : Function
    Establishes reading order for boxes considering page blocks.

merge_boxes_within_column_width : Function
    Merges compatible boxes within column width constraints.

preprocess_bbox : Function
    Main wrapper function that performs all preprocessing steps.

plot_boxes_on_image : Function
    Visualizes bounding boxes on document images.

Dependencies:
------------
- pandas
- numpy
- numba
- opencv-python (cv2)
- matplotlib
- tqdm

Notes:
------
- Requires input data to have a 'page_id' column for unique page identification
- Designed for document layout analysis tasks
- Includes visualization capabilities for debugging and verification
"""



import pandas as pd
import numpy as np
from numba import jit
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


@jit(nopython=True)
def numba_fill_count(count_array, x1_adj, x2_adj, y1_adj, y2_adj):
    for i in range(len(x1_adj)):
        count_array[y1_adj[i]:y2_adj[i], x1_adj[i]:x2_adj[i]] += 1
    return count_array

def sum_of_area(df, mask_shape, y_min, x_min):
    count_array = np.zeros(mask_shape, dtype=np.int32)
    
    # Convert coordinates to numpy arrays and adjust
    x1_adj = (df['x1'] - x_min).astype(int).values
    x2_adj = (df['x2'] - x_min).astype(int).values
    y1_adj = (df['y1'] - y_min).astype(int).values
    y2_adj = (df['y2'] - y_min).astype(int).values
    
    # Use numba-optimized function
    count_array = numba_fill_count(count_array, x1_adj, x2_adj, y1_adj, y2_adj)

    return count_array

def calculate_article_coverage(group):
    """
    Calculates the number of pixels in the page that are covered by bounding boxes
    """
    # Use pre-calculated print dimensions
    print_height = group['print_height'].iloc[0]
    print_width = group['print_width'].iloc[0]
    x_min = group['print_x1'].iloc[0]
    y_min = group['print_y1'].iloc[0]
    
    mask_shape = (print_height, print_width)
    
    # Split data by article type
    text_data = group[group['class']!='figure']
    image_data = group[group['class']=='figure']
    
    # Calculate areas
    text_counts = sum_of_area(text_data, mask_shape, y_min, x_min)
    image_counts = sum_of_area(image_data, mask_shape, y_min, x_min)
    
    # Calculate coverage
    text_overlap_pixels = np.sum(text_counts > 1)  # Pixels where text boxes overlap
    text_coverage_pixels = np.sum(text_counts > 0)  # Total pixels covered by text (including overlap)
    text_image_overlap = np.sum((text_counts > 0) & (image_counts > 0))  # Overlap between text and images
    total_overlap =  np.sum((text_counts + image_counts) > 1)
    total_covered_pixels = np.sum((text_counts + image_counts) > 0)  # Total coverage
    print_area = group['print_area'].iloc[0]

    return {
        'print_height': print_height,
        'print_width': print_width,
        'x_min': x_min,
        'y_min': y_min,
        'text_coverage_pixels': text_coverage_pixels,
        'text_overlap_pixels': text_overlap_pixels,
        'text_image_overlap': text_image_overlap,
        'total_covered_pixels': total_covered_pixels,
        'total_overlap':total_overlap,
        'maximum_print_area': print_area,
        "perc_print_area_overlap":total_overlap/print_area,
        "perc_print_area_coverage":total_covered_pixels/print_area
    }

def calculate_coverage_and_overlap(df):
    """
    Process the entire dataframe using a for loop with tqdm, this aggregates to page level
    """
    # Get unique page_ids
    page_ids = df['page_id'].unique()
    
    # Initialize lists to store results
    results = []
    
    # Process each page_id
    for page_id in tqdm(page_ids, desc="Processing pages"):
        # Get data for this page
        page_data = df[df['page_id'] == page_id]
        
        # Calculate coverage
        coverage_data = calculate_article_coverage(page_data)
        coverage_data['page_id'] = page_id
        
        # Store results
        results.append(coverage_data)
    
    # Convert results to DataFrame at the end
    result_df = pd.DataFrame(results)

    result_df[['perc_print_area_overlap', 'perc_print_area_coverage']] = result_df[['perc_print_area_overlap', 'perc_print_area_coverage']].round(2)
    
    return result_df


def print_area_meta(df):
    """
    calculates information about the area of the printed page
    """
    df['print_y1'] = df.groupby('page_id')['y1'].transform('min')
    df['print_y2'] = df.groupby('page_id')['y2'].transform('max')
    df['print_x1'] = df.groupby('page_id')['x1'].transform('min')
    df['print_x2'] = df.groupby('page_id')['x2'].transform('max')
    df['print_height'] = (df['print_y2'] - df['print_y1'] ).astype(int)
    df['print_width'] = (df['print_x2'] - df['print_x1'] ).astype(int)
    df['print_area'] = df['print_height'] * df['print_width']

    return df
     

def reclassify_abandon_boxes(df, top_fraction=0.1):
    # Calculate the threshold line from the top of the printed area for each page
    threshold_line = df['print_y1'] + (df['print_height'] * top_fraction)
    
    # Find pages that have abandon boxes
    pages_with_abandon = df[df['class'] == 'abandon']['page_id'].unique()
    
    # Only proceed with reclassification for pages that have abandon boxes
    if len(pages_with_abandon) > 0:
        # Find maximum y2 of abandon boxes above threshold for each page
        abandon_lines = (
            df[
                (df['class'] == 'abandon') & 
                (df['page_id'].isin(pages_with_abandon)) &
                (df['y1'] <= threshold_line)
            ]
            .groupby('page_id')['y2']
            .max()
        )
        
        # Create abandon_line column only for pages with abandon boxes
        df['abandon_line'] = df['page_id'].map(abandon_lines)
        
        # Reclassify boxes only on pages with abandon boxes
        mask = (
            df['page_id'].isin(pages_with_abandon) & 
            (df['center_y'] <= df['abandon_line']) &
            df['abandon_line'].notna()
        )
        df.loc[mask, 'class'] = 'abandon'
        
    return df



def assign_columns(articles_df, width_overlap_threshold=0.1):
    """
    Assigns column numbers to boxes and their corresponding column edges.
    
    Parameters:
    - articles_df: DataFrame containing box coordinates and print area information
    - width_overlap_threshold: Minimum fraction of column width that needs to overlap
    """
    articles_df['column_number'] = None
    articles_df['c1'] = None  # Left column edge
    articles_df['c2'] = None  # Right column edge

    articles_df['column_counts'] = articles_df['column_counts'].fillna(1)
    
    for page_id in tqdm(articles_df['page_id'].unique(), desc="Assigning article columns in pages"):
        # Get print area info for this page from the first row of the page
        page_articles = articles_df[articles_df['page_id'] == page_id]
        page_info = page_articles.iloc[0]
        
        #fills NA values with 1, this prevents issues later in this function...
        # I hope it doesn't have some crazy knock on effect

        # Calculate column width and boundaries
        column_width = page_info['print_width'] / page_info['column_counts']
        column_bins = np.linspace(
            page_info['print_x1'], 
            page_info['print_x2'] + 0.001, 
            int(page_info['column_counts']) + 1
        )
        
        # Process each box
        for idx in page_articles.index:
            box = page_articles.loc[idx]
            box_columns = []
            
            # Check overlap with each column
            for col_num in range(len(column_bins) - 1):
                col_left = column_bins[col_num]
                col_right = column_bins[col_num + 1]
                
                # Calculate overlap
                overlap_left = max(box['x1'], col_left)
                overlap_right = min(box['x2'], col_right)
                
                if overlap_right > overlap_left:  # There is some overlap
                    overlap_width = overlap_right - overlap_left
                    
                    # Check if overlap is substantial
                    box_width = box['x2'] - box['x1']
                    if overlap_width >= (box_width * width_overlap_threshold):
                        box_columns.append(col_num + 1)  # 1-based column numbering
            
            # Assign column number and edges
            if len(box_columns) == 0:
                # No substantial overlap with any column
                articles_df.loc[idx, 'column_number'] = 0
                articles_df.loc[idx, 'c1'] = page_info['print_x1']  # Leftmost edge of print area
                articles_df.loc[idx, 'c2'] = page_info['print_x2']  # Rightmost edge of print area
            elif len(box_columns) == 1:
                # Box belongs to a single column
                col_num = box_columns[0] - 1  # Convert to 0-based index
                articles_df.loc[idx, 'column_number'] = box_columns[0]
                articles_df.loc[idx, 'c1'] = column_bins[col_num]
                articles_df.loc[idx, 'c2'] = column_bins[col_num + 1]
            else:
                # Box spans multiple columns
                articles_df.loc[idx, 'column_number'] = 0
                articles_df.loc[idx, 'c1'] = page_info['print_x1']  # Leftmost edge of print area
                articles_df.loc[idx, 'c2'] = page_info['print_x2']  # Rightmost edge of print area
                
    return articles_df


def create_page_blocks(df):
    """
    Creates page blocks based on titles (column 0) and their subsequent content.
    Assigns 0 to any boxes that don't fall into a defined block.
    """
    df_sorted = df.copy()

    # Initialize all page_blocks to 0
    df_sorted['page_block'] = 0

    # Process each page separately
    for page_id in df_sorted['page_id'].unique():
        page_df = df_sorted[df_sorted['page_id'] == page_id]

        # Find block separators
        block_separators = page_df[
            (page_df['column_number'] == 0) & 
            (page_df['class'] == 'title')
        ].sort_values('center_y')

        # For each block separator
        for i, (separator_idx, separator_row) in enumerate(block_separators.iterrows(), 1):
            # Assign block number to separator
            df_sorted.loc[separator_idx, 'page_block'] = i

            # Find next separator's y1 if it exists
            next_separator_y1 = float('inf')
            if i < len(block_separators):
                next_separator_y1 = block_separators.iloc[i]['y1']

            # Find all boxes that are below this separator but above next separator
            mask = (
                (df_sorted['page_id'] == page_id) &
                (df_sorted['y1'] >= separator_row['y1']) &  # Changed from y2 to y1
                (df_sorted['y1'] < next_separator_y1)
            )

            # Assign same block number to all boxes in this section
            df_sorted.loc[mask, 'page_block'] = i

        # Handle content before first block separator (assign to block 0)
        if len(block_separators) > 0:
            first_separator_y1 = block_separators.iloc[0]['y1']
            initial_content_mask = (
                (df_sorted['page_id'] == page_id) &
                (df_sorted['y1'] < first_separator_y1)
            )
            df_sorted.loc[initial_content_mask, 'page_block'] = 0

    return df_sorted


def create_reading_order(df):
    """
    Creates reading order taking into account page blocks.
    Removes perfectly overlapping boxes keeping the one with higher confidence.
    """
    df_sorted = create_page_blocks(df)

    # Remove perfect overlaps within same column and block
    df_cleaned = []
    for (page_id, block, col), group in df_sorted.groupby(['page_id', 'page_block', 'column_number']):
        # Sort by y1, center_y and confidence
        group_sorted = group.sort_values(['center_y', 'y1', 'confidence'], ascending=[True, True, False])

        # Keep track of processed boxes to remove duplicates
        kept_boxes = []
        for _, box in group_sorted.iterrows():
            # Check if current box perfectly overlaps with any kept box
            is_duplicate = any(
                (box['y1'] == kept_box['y1']) and 
                (box['y2'] == kept_box['y2']) and 
                (box['x1'] == kept_box['x1']) and 
                (box['x2'] == kept_box['x2'])
                for kept_box in kept_boxes
            )

            if not is_duplicate:
                kept_boxes.append(box)

        df_cleaned.extend(kept_boxes)

    df_cleaned = pd.DataFrame(df_cleaned)

    # Create reading order for cleaned dataframe
    df_cleaned['reading_order'] = (
        df_cleaned
        .sort_values(['page_id', 'page_block', 'column_number', 'y1', 'center_y'])
        .groupby('page_id')
        .cumcount() + 1
    )

    return df_cleaned.sort_index()



def calculate_overlap(box1, box2):
    """Calculate the vertical overlap percentage between two boxes"""
    # Find the overlapping region
    overlap_start = max(box1['y1'], box2['y1'])
    overlap_end = min(box1['y2'], box2['y2'])
    
    if overlap_end <= overlap_start:
        return 0.0
    
    # Calculate overlap percentage relative to the smaller box
    overlap_height = overlap_end - overlap_start
    box1_height = box1['y2'] - box1['y1']
    box2_height = box2['y2'] - box2['y1']
    min_height = min(box1_height, box2_height)
    
    return (overlap_height / min_height) * 100

def merge_boxes(box1, box2):
    """Merge two boxes into one"""
    return {
        'x1': min(box1['x1'], box2['x1']),
        'x2': max(box1['x2'], box2['x2']),
        'y1': min(box1['y1'], box2['y1']),
        'y2': max(box1['y2'], box2['y2']),
        'center_x': (min(box1['x1'], box2['x1']) + max(box1['x2'], box2['x2'])) / 2,
        'center_y': (min(box1['y1'], box2['y1']) + max(box1['y2'], box2['y2'])) / 2,
        'column_number': box1['column_number'],  # Assuming same column
        'page_id': box1['page_id'],  # Preserve page_id
        'reading_order': min(box1['reading_order'], box2['reading_order'])  # Take earlier reading order
    }

def merge_overlapping_boxes(df, min_overlap_percent=50):
    """Merge overlapping boxes within the same column and page"""
    # Sort by page_id, column number, and reading order
    df = df.sort_values(['page_id', 'column_number', 'reading_order'])
    
    # Process each page and column separately
    merged_boxes = []
    
    # Group by page_id first
    for page_id in df['page_id'].unique():
        page_df = df[df['page_id'] == page_id]
        
        # Then process each column within the page
        for column in page_df['column_number'].unique():
            column_boxes = page_df[page_df['column_number'] == column].to_dict('records')
            
            while len(column_boxes) > 0:
                current_box = column_boxes.pop(0)
                merged = False
                
                i = 0
                while i < len(column_boxes):
                    next_box = column_boxes[i]
                    
                    # Only consider consecutive boxes in reading order for merging
                    if abs(current_box['reading_order'] - next_box['reading_order']) > 1:
                        i += 1
                        continue
                        
                    overlap = calculate_overlap(current_box, next_box)
                    
                    if overlap >= min_overlap_percent:
                        # Merge boxes and remove the second box
                        current_box = merge_boxes(current_box, next_box)
                        column_boxes.pop(i)
                        merged = True
                    else:
                        i += 1
                
                merged_boxes.append(current_box)
    
    return pd.DataFrame(merged_boxes)

def merge_boxes_within_column_width(df, width_multiplier=1):
    """
    Merge bounding boxes that are not 'figure' or 'table' if the merged height doesn't exceed
    x times the column_width and classes match, maintaining reading order and working with multiple pages.

    Parameters:
    df: DataFrame with columns x1,x2,y1,y2,center_x,center_y,column_number,column_width,
        reading_order, page_id, and type
    width_multiplier: maximum allowed height as a multiple of column_width

    Returns:
    DataFrame with merged boxes
    """
    df = df.copy()
    df = df.sort_values(['page_id', 'reading_order'])

    merged_boxes = []

    for page_id in tqdm(df['page_id'].unique(), desc="Merging pages"):
        page_df = df[df['page_id'] == page_id]

        # Process all columns including column 0
        for col_num in page_df['column_number'].unique():
            col_df = page_df[page_df['column_number'] == col_num]

            current_box = None

            for _, row in col_df.iterrows():
                if row['class'] in ['figure', 'table']:
                    if current_box is not None:
                        merged_boxes.append(current_box)
                        current_box = None
                    merged_boxes.append(row.to_dict())
                    continue

                # For column 0, don't merge boxes, just add them directly
                if col_num == 0:
                    merged_boxes.append(row.to_dict())
                    continue

                if current_box is None:
                    current_box = row.to_dict()
                else:
                    # Calculate potential merged box height
                    merged_height = max(row['y2'], current_box['y2']) - min(row['y1'], current_box['y1'])

                    # Check if merge would exceed height limit and that the classes are the same
                    if (merged_height <= row['column_width'] * width_multiplier and current_box['class'] == row['class']):  

                        # Merge boxes
                        current_box['x1'] = min(current_box['x1'], row['x1'])
                        current_box['x2'] = max(current_box['x2'], row['x2'])
                        current_box['y1'] = min(current_box['y1'], row['y1'])
                        current_box['y2'] = max(current_box['y2'], row['y2'])
                        current_box['center_x'] = (current_box['x1'] + current_box['x2']) / 2
                        current_box['center_y'] = (current_box['y1'] + current_box['y2']) / 2
                    else:
                        # Add current box to results and start new one
                        merged_boxes.append(current_box)
                        current_box = row.to_dict()

            # Add last box if exists
            if current_box is not None:
                merged_boxes.append(current_box)

    result_df = pd.DataFrame(merged_boxes)
    result_df = result_df.sort_values(['page_id', 'reading_order'])
    return result_df

def adjust_y2_coordinates(df):
    """
    Adjusts y2 coordinates for boxes within each column of each block:
    - Sets y2 to the y1 of the next box in reading order
    - Removes overlaps between boxes
    - Keeps the last box in each column unchanged
    """
    # Create a copy of the dataframe
    df_adjusted = df.copy()

    # Sort within groups and shift y1 values
    df_adjusted['y2'] = (df_adjusted
        .sort_values(['page_id', 'page_block', 'column_number', 'y1'])
        .groupby(['page_id', 'page_block', 'column_number'])['y1']
        .shift(-1)
        .fillna(df_adjusted['y2']))  # Keep original y2 for last box in each group

    return df_adjusted


def adjust_x_coordinates(df):
    """
    Adjusts x coordinates based on column boundaries (c1 and c2):
    - If x1 > c1, sets x1 to c1
    - If x2 < c2, sets x2 to c2
    This extends boxes that are too narrow but preserves wider boxes.
    Boxes with class 'figure' are not adjusted.
    """
    df_adjusted = df.copy()

    # Ensure c1 and c2 are float64
    df_adjusted['c1'] = df_adjusted['c1'].astype('float64')
    df_adjusted['c2'] = df_adjusted['c2'].astype('float64')

    # Create mask for non-figure boxes
    non_figure_mask = df_adjusted['class'] != 'figure'

    # Adjust x1 where it's greater than c1 (excluding figures)
    mask_x1 = (df_adjusted['x1'] > df_adjusted['c1']) & non_figure_mask
    df_adjusted.loc[mask_x1, 'x1'] = df_adjusted.loc[mask_x1, 'c1'].astype('float64')

    # Adjust x2 where it's less than c2 (excluding figures)
    mask_x2 = (df_adjusted['x2'] < df_adjusted['c2']) & non_figure_mask
    df_adjusted.loc[mask_x2, 'x2'] = df_adjusted.loc[mask_x2, 'c2'].astype('float64')

    return df_adjusted


def basic_box_data(df):

    """ 
    Calculates the basic information about the bounding boxes on the page
    """

    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    df['ratio'] = df['height']/df['width']  
    df['center_x'] = df['width'] + df['x1']
    df['center_y'] = df['height'] + df['y1']
    
    # Calculate the column width to get the total number of columns on the page
    df_text = df.loc[df['class'].isin(['text']), ['page_id', 'width']].copy()
    df_text = df_text.groupby('page_id')['width'].median().rename('median_box_width')
    df = df.join(df_text, on = 'page_id')
    
    df = print_area_meta(df)
    df['column_counts'] =  np.floor(df['print_width']/df['median_box_width'])
    df['column_width'] = df['print_width']/df['column_counts']

    return df

def fill_column_gaps(bbox_df):
    """
    Add bounding boxes at the top and bottom of columns where they're missing.
    Only processes pages with multiple columns and a single block.

    Args:
        bbox_df (pd.DataFrame): DataFrame containing bounding box information

    Returns:
        pd.DataFrame: DataFrame with additional bounding boxes added
    """
    # Create mask for eligible pages
    page_col_counts = bbox_df.groupby('page_id')['column_number'].transform('max')
    page_block_counts = bbox_df.groupby('page_id')['page_block'].transform('nunique')
    eligible_mask = (page_col_counts > 1) & (page_block_counts == 1)

    # Split into eligible and ineligible dataframes
    eligible_df = bbox_df[eligible_mask].copy()
    ineligible_df = bbox_df[~eligible_mask].copy()

    new_boxes = []

    # Process only if there are eligible pages
    if not eligible_df.empty:
        for page_id in eligible_df['page_id'].unique():
            page_data = eligible_df[eligible_df['page_id'] == page_id]

            # Find global minimum y1 and maximum y2 for the page
            minimum_y1 = page_data['y1'].min()
            maximum_y2 = page_data['y2'].max()

            # Get filename and issue for this page
            filename = page_data['filename'].iloc[0]
            issue = page_data['issue'].iloc[0]

            # Process each column (excluding column 0)
            for col_num in page_data[page_data['column_number'] > 0]['column_number'].unique():
                col_data = page_data[page_data['column_number'] == col_num]

                if len(col_data) == 0:
                    continue

                # Get column boundaries
                c1 = col_data['c1'].iloc[0]
                c2 = col_data['c2'].iloc[0]

                # Find top and bottom boxes in the column
                top_box = col_data.loc[col_data['reading_order'].idxmin()]
                bottom_box = col_data.loc[col_data['reading_order'].idxmax()]

                # Add top box if needed
                if top_box['y1'] > minimum_y1:
                    new_boxes.append({
                        'page_id': page_id,
                        'filename': filename,
                        'issue': issue,
                        'class': 'text',
                        'y1': minimum_y1,
                        'y2': top_box['y1'],
                        'x1': top_box['x1'],
                        'x2': top_box['x2'],
                        'c1': c1,
                        'c2': c2,
                        'column_number': col_num,
                        'page_block': top_box['page_block'],
                        'reading_order': -1  # Will be recalculated later
                    })

                # Add bottom box if needed
                if bottom_box['y2'] < maximum_y2:
                    new_boxes.append({
                        'page_id': page_id,
                        'filename': filename,
                        'issue': issue,
                        'class': 'text',
                        'y1': bottom_box['y2'],
                        'y2': maximum_y2,
                        'x1': bottom_box['x1'],
                        'x2': bottom_box['x2'],
                        'c1': c1,
                        'c2': c2,
                        'column_number': col_num,
                        'page_block': bottom_box['page_block'],
                        'reading_order': -1  # Will be recalculated later
                    })

    # Add new boxes to the eligible DataFrame if any were created
    if new_boxes:
        new_boxes_df = pd.DataFrame(new_boxes)
        new_boxes_df = basic_box_data(new_boxes_df)
        new_boxes_df = print_area_meta(new_boxes_df)
        eligible_df = pd.concat([eligible_df, new_boxes_df], ignore_index=True)

    # Combine eligible and ineligible dataframes
    result_df = pd.concat([eligible_df, ineligible_df], ignore_index=True)

    return result_df


def postprocess_bbox(df, min_height = 10, width_multiplier = 1.5, remove_abandon = True, fill_columns = True):

    """ 
    This function performs all the post-processing on the bounding boxes to clean them up after being produced by DOCLAyout-Yolo
    and to make them ready for sending image crops to the image-to-text model. 
    The function assumes that the bounding boxes have been made by DocLayout-yolo. Other Yolo models may also work but this has not been tested.
    The function is a wrapper for functions from the bbox_functions module

    Files require a page_id column which uniquely identifies each page
    """

    bbox_all_df = df.copy()

    bbox_all_df['issue'] = bbox_all_df['filename'].str.split('_page_').str[0]

    bbox_all_df = basic_box_data(bbox_all_df)

    bbox_all_df = reclassify_abandon_boxes(bbox_all_df, top_fraction=0.1)

    bbox_df = bbox_all_df.copy()

    if remove_abandon:
        bbox_df = bbox_all_df.loc[bbox_all_df['class']!='abandon'].copy()

    # After re-classifying boxes as abandon and dropping the abandon boxes re-calculate the print area which should have changed
    bbox_df = print_area_meta(bbox_df)

    bbox_df = assign_columns(bbox_df)


    """ 
    This logic has been changed as it looks like keeping titles will make splitting up texts a lot easier and 
    that titles are pretty reliable if you do a little post processing.
    # change class when there is more than one column
    bbox_df['class'] = np.where((bbox_df['column_counts']>1) & 
                                (bbox_df['column_number']!=0) &
                               (~bbox_df['class'].isin(['figure', 'table'])),
                                'text',  # Value if condition is True
                                bbox_df['class'])  # Value if condition is False

    #Change class when there is only 1 column
    bbox_df['class'] = np.where((bbox_df['column_counts']==1) & 
                                (bbox_df['column_number']!=0) &
                                (~bbox_df['class'].isin(['figure', 'table', 'title'])), 
                                'text',  # Value if condition is True
                                bbox_df['class'])  # Value if condition is False
    """
    # This re-labels everything that is not abandon, text, table, or figure as title.
    bbox_df['class'] = np.where((~bbox_df['class'].isin(['figure', 'table', 'text', 'abandon'])), 
                                'title',  # Value if condition is True
                                bbox_df['class'])  # Value if condition is False

    bbox_df= create_reading_order(bbox_df)
    #Remove overlaps by adjusting the y2 coordinate to the subsequent bounding box
    # This causes boxes that overlap another box to bring there lower bound up to the top of the subsequent box
    # This can make large boxes much smaller, and results in overlaps being removed
    bbox_df = adjust_y2_coordinates(bbox_df)
    #adjust the x limits to the column width if box is narrower than the column
    bbox_df = adjust_x_coordinates(bbox_df)
    #Should the columns be filled?
    if fill_columns:
        bbox_df = fill_column_gaps(bbox_df)
   
    # Filter out boxes that are too small after y2 adjustment
    bbox_df['height'] = bbox_df['y2'] - bbox_df['y1']  
    bbox_df = bbox_df[bbox_df['height'] >= min_height]

    #remove small bounding boxes to make sending to LLM more efficient and results in larger text blocks
    if width_multiplier is not None:
        bbox_df = merge_boxes_within_column_width(bbox_df, width_multiplier=width_multiplier)

    # Adjust y2 again as the box deletion and merging has changed boundaries
    bbox_df = adjust_y2_coordinates(bbox_df)

    #due to merging and deletion re-do reading order
    bbox_df = create_reading_order(bbox_df)

    #add in bbox ID
    bbox_df['box_page_id'] = "B" + bbox_df['page_block'].astype(str) + "C"+bbox_df['column_number'].astype(str)  + "R" + bbox_df['reading_order'].astype(str) 
    bbox_df['ratio'] = bbox_df['height']/bbox_df['width']  #box ratio

    return bbox_df


def remove_duplicate_boxes(df):
    """
    Remove duplicate bounding boxes from the DataFrame.
    Duplicates are defined as boxes with the same class, coordinates, and image.
    This function is primarliy used for creating a dataset to be used for fine-tuning.
    It is likely (but not certain) the majority of the overlapping boxes are of class "abandon", 
    or title, as these two classes are created duriing.
    post-processing and are the classes presenting the most issues
    """
    # Calculate normalized coordinates
    df = df.copy()
    
    # Calculate image dimensions
    img_width = df['page_width'] + df['x1']
    img_height = df['page_height'] + df['y1']
    
    # Add normalized coordinates columns
    df['x_center_norm'] = ((df['x1'] + df['x2']) / 2) / img_width
    df['y_center_norm'] = ((df['y1'] + df['y2']) / 2) / img_height
    df['width_norm'] = (df['x2'] - df['x1']) / img_width
    df['height_norm'] = (df['y2'] - df['y1']) / img_height
    
    # Round normalized coordinates to 6 decimal places to avoid floating point issues
    for col in ['x_center_norm', 'y_center_norm', 'width_norm', 'height_norm']:
        df[col] = df[col].round(6)
    
    # Count initial rows
    initial_count = len(df)
    
    # Drop duplicates based on normalized coordinates, class, and image
    df = df.drop_duplicates(
        subset=['filename', 'class', 
                'x_center_norm', 'y_center_norm', 
                'width_norm', 'height_norm']
    )
    
    # Remove the temporary normalized coordinate columns
    df = df.drop(columns=['x_center_norm', 'y_center_norm', 
                         'width_norm', 'height_norm'])
    
    # Count removed duplicates
    removed_count = initial_count - len(df)
    
    print(f"Removed {removed_count} duplicate bounding boxes")
    print(f"Original count: {initial_count}")
    print(f"New count: {len(df)}")
    
    # Group by filename and count affected images
    affected_images = df.groupby('filename').size()
    affected_count = len(affected_images[affected_images > 0])
    print(f"Number of affected images: {affected_count}")
    
    return df

def plot_boxes_on_image(df, image_path, figsize=(15,15), show_reading_order=False):
    """
    Plot bounding boxes and their classifications on a page image with optional reading order visualization.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing bounding box coordinates and classifications.
        Required columns: 'x1', 'y1', 'x2', 'y2', 'class'
        Optional column: 'reading_order' (if show_reading_order=True)
    
    image_path : str
        Path to the input image file.
    
    figsize : tuple, optional (default=(15,15))
        Figure size in inches (width, height).
    
    show_reading_order : bool, optional (default=False)
        If True, draws arrows between boxes indicating reading order sequence.
        Requires 'reading_order' column in the DataFrame.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plotted image with bounding boxes.

    Notes:
    ------
    - The function uses predefined colors for specific classes:
        * 'text': Red
        * 'title': Green
        * 'figure': Blue
    - Additional classes are assigned colors automatically using a rainbow colormap.
    - Each box is labeled with its class name.
    - A legend is included showing all class types and their corresponding colors.
    """
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Display the image
    ax.imshow(image)

    # Define fixed colors for specific classes
    fixed_colors = {
        'text': '#FF0000',  # Red
        'title': '#00FF00',       # Green
        'figure': '#0000FF'       # Blue
    }

    # Create a color map for any other classes
    unique_classes = [cls for cls in df['class'].unique() if cls not in fixed_colors]
    if unique_classes:  # If there are other classes
        additional_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        additional_color_dict = dict(zip(unique_classes, additional_colors))
    else:
        additional_color_dict = {}

    # Combine fixed and additional colors
    class_color_dict = {**fixed_colors, **additional_color_dict}

    # Plot each bounding box
    for idx, row in df.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        box_class = row['class']

        # Create rectangle patch
        width = x2 - x1
        height = y2 - y1

        # Get color for this class
        color = class_color_dict[box_class]

        # Draw rectangle
        rect = plt.Rectangle((x1, y1), width, height,
                           fill=False,
                           color=color,
                           linewidth=2)
        ax.add_patch(rect)

        # Optionally add class labels
        ax.text(x1, y1-5, box_class, 
                color=color,
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))


    # Add reading order arrows if requested
    if show_reading_order and 'reading_order' in df.columns:
        # Sort by reading order
        df_sorted = df.sort_values('reading_order')

        # Calculate center points
        df_sorted['center_x'] = (df_sorted['x1'] + df_sorted['x2']) / 2
        df_sorted['center_y'] = (df_sorted['y1'] + df_sorted['y2']) / 2

        # Draw arrows between consecutive boxes
        for i in range(len(df_sorted)-1):
            start = (df_sorted.iloc[i]['center_x'], df_sorted.iloc[i]['center_y'])
            end = (df_sorted.iloc[i+1]['center_x'], df_sorted.iloc[i+1]['center_y'])

            ax.annotate('',
                       xy=end,
                       xytext=start,
                       arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     linewidth=2,
                                     mutation_scale=20),  # Makes the arrow head bigger
                       )

    # Remove axes
    ax.set_axis_off()

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, 
                                   label=class_name)
                      for class_name, color in class_color_dict.items()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    return fig  # Return the figure instead of showing it

# And then the loop function would be:
def save_plots_for_all_files(df, image_dir, output_dir, figsize=(15,15), show_reading_order = False):
    os.makedirs(output_dir, exist_ok=True)
    unique_images = df['filename'].unique()

    for filename in tqdm(unique_images, desc="Processing images"):
        image_df = df[df['filename'] == filename]
        image_path = os.path.join(image_dir, filename)

        # Get the figure from plot_boxes_on_image
        fig = plot_boxes_on_image(image_df, image_path, figsize=figsize, show_reading_order = show_reading_order)

        # Save the figure
        output_path = os.path.join(output_dir, f'annotated_{filename}')
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory