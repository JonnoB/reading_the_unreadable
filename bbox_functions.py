"""
These are helper functions related to the pre-processing stage, of manipulating the bounding box to make it 
ready to send to the LLM




"""



import pandas as pd
import numpy as np
from numba import jit
import os
from tqdm import tqdm




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
    total_covered_pixels = np.sum((text_counts + image_counts) > 0)  # Total coverage
    
    return pd.Series({
        'text_coverage_pixels': text_coverage_pixels,
        'text_overlap_pixels': text_overlap_pixels,
        'text_image_overlap': text_image_overlap,
        'total_covered_pixels': total_covered_pixels,
        'maximum_print_area': group['print_area'].iloc[0]
    })

def calculate_coverage_and_overlap(df):
    """
    Process the entire dataframe using a for loop with tqdm
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
        
        # Store results
        results.append({
            'page_id': page_id,
            'text_coverage_pixels': coverage_data['text_coverage_pixels'],
            'text_overlap_pixels': coverage_data['text_overlap_pixels'],
            'text_image_overlap': coverage_data['text_image_overlap'],
            'total_covered_pixels': coverage_data['total_covered_pixels']
        })
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # Calculate additional metrics if needed
    result_df['text_coverage_percent'] = (result_df['text_coverage_pixels'] / result_df['maximum_print_area']).round(2)
    result_df['text_overlap_percent'] = (result_df['text_overlap_pixels'] / result_df['text_coverage_pixels']).round(2)
    result_df['total_coverage_percent'] = (result_df['total_covered_pixels'] / result_df['maximum_print_area']).round(2)
    
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
     

def reclassify_abandon_boxes_broken(df, top_fraction=0.1):
    # Calculate the threshold line from the top of the printed area for each page
    threshold_line = df['print_y1'] + (df['print_height'] * top_fraction)
    
    # Find maximum y2 of abandon boxes above threshold for each page
    abandon_lines = (
        df[df['class'] == 'abandon']
        .loc[df['y1'] <= threshold_line]
        .groupby('page_id')['y2']
        .max()
    )
    
    # Create abandon_line column using threshold as default where no abandon boxes exist
    df['abandon_line'] = df['page_id'].map(abandon_lines).fillna(
        threshold_line
    )
    
    # Reclassify all boxes whose center_y is above the abandon line
    df['class'] = np.where(
        df['center_y'] <= df['abandon_line'],
        'abandon',
        df['class']
    )
    
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


# Example usage:
# df = reclassify_abandon_boxes(df, top_fraction=0.1)
# Example usage:
# df = process_all_pages(df, top_fraction=0.1)


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
    
    for page_id in articles_df['page_id'].unique():
        # Get print area info for this page from the first row of the page
        page_articles = articles_df[articles_df['page_id'] == page_id]
        page_info = page_articles.iloc[0]
        
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
                    if overlap_width >= (column_width * width_overlap_threshold):
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
        
        # Find titles (column 0)
        titles = page_df[(page_df['column_number'] == 0) & (page_df['class'] == 'title')].sort_values('center_y')
        
        # For each title
        for i, (title_idx, title_row) in enumerate(titles.iterrows(), 1):
            # Assign block number to title
            df_sorted.loc[title_idx, 'page_block'] = i
            
            # Find next title's y1 if it exists
            next_title_y1 = float('inf')
            if i < len(titles):
                next_title_y1 = titles.iloc[i]['y1']
            
            # Find all boxes that are below this title but above next title
            mask = (
                (df_sorted['page_id'] == page_id) &
                (df_sorted['y1'] >= title_row['y2']) &  # Below current title
                (df_sorted['y1'] < next_title_y1) &     # Above next title
                (df_sorted['column_number'] > 0)  &     # Not outside column structure
                (df_sorted['class']!='title')           # Not a title
            )
            
            # Assign same block number to all boxes in this section
            df_sorted.loc[mask, 'page_block'] = i
    
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
        group_sorted = group.sort_values(['y1', 'center_y', 'confidence'], ascending=[True, True, False])
        
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

