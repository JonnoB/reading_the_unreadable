
""" 

These functions are generally to do with the analysis and evaluation code


"""

import os
import pandas as pd
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List



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

def reshape_metrics(df, metric_col='cer_score', group_col='model', spread_col='dataset', 
                   agg_func='mean', round_digits=3):
    """
    Reshapes a dataframe by spreading the dataset column and calculating aggregated metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    metric_col : str, default='cer_score'
        Name of the column containing the metric values
    group_col : str, default='model'
        Name of the column to group by (will become index)
    spread_col : str, default='dataset'
        Name of the column to spread (will become columns)
    agg_func : str, default='mean'
        Any valid pandas aggregation function
    round_digits : int, default=3
        Number of decimal places to round to
        
    Returns:
    --------
    pandas.DataFrame
        Reshaped dataframe with datasets as columns and aggregated metrics
    """
    
    try:
        result = df.pivot_table(
            values=metric_col,
            index=group_col,
            columns=spread_col,
            aggfunc=agg_func
        ).round(round_digits)
        return result
        
    except (ValueError, TypeError) as e:
        raise ValueError(f"'{agg_func}' is not a valid aggregation function: {str(e)}")
    


def highlight_extreme(data, extreme='min', model_column=None):
    """
    Highlight the minimum or maximum value in each column with LaTeX bold formatting.
    Exclude the model_column from the extreme value calculation.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = data.copy()

    # Convert numeric columns to object type to avoid dtype warnings
    for col in data.columns:
        if col != model_column and pd.api.types.is_numeric_dtype(data[col]):
            df_copy[col] = df_copy[col].astype(object)

    # Apply bold formatting to extreme values
    for col in data.columns:
        if col == model_column:
            continue
        if extreme == 'min':
            extreme_val = data[col].min()
        elif extreme == 'max':
            extreme_val = data[col].max()
        else:
            raise ValueError("extreme must be 'min' or 'max'")

        df_copy.loc[data[col] == extreme_val, col] = f'\\textbf{{{extreme_val}}}'

    return df_copy

def dataframe_to_latex_with_bold_extreme(df, extreme='min', model_column=None, caption='', label=''):

    """ 
    A simple function to output correctly formatted tables
    """
    # Highlight the extreme values
    styled_df = highlight_extreme(df, extreme=extreme, model_column=model_column)

    # Convert to LaTeX with float formatting, caption, and label
    latex_table = styled_df.to_latex(index=False, escape=False, float_format="%.2f", caption=caption, label=label)

    return latex_table


def clean_text(text):

    """
    Clean the text so that all paragraphs are a single line, but keep paragraphs.
    
    """
    # Function to clean and format text
    
    # Replace hyphenated line breaks with temporary marker
    text = re.sub(r'-\n', 'HYPHENBREAK', text)
    
    # Replace double line breaks with a marker
    text = re.sub(r'\n\n+', 'DOUBLEPAGEBREAK', text)
    
    # Remove single line breaks
    text = re.sub(r'\n', ' ', text)
    
    # Restore hyphenated words (removing the hyphen)
    text = text.replace('HYPHENBREAK', '')
    
    # Restore double line breaks
    text = text.replace('DOUBLEPAGEBREAK', '\n\n')
    
    # Clean up extra spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def remove_line_breaks(series):
    """
    Clean the text in a pandas Series so that all paragraphs are a single line, but keep paragraphs.
    Vectorized version for better performance with large datasets.
    """
    # Make a copy to avoid SettingWithCopyWarning
    series = series.copy()
    
    # Apply all transformations using vectorized string operations
    series = series.str.replace(r'-\n', 'HYPHENBREAK', regex=True)
    series = series.str.replace(r'\n\n+', 'DOUBLEPAGEBREAK', regex=True)
    series = series.str.replace(r'\n', ' ', regex=True)
    series = series.str.replace('HYPHENBREAK', '')
    series = series.str.replace('DOUBLEPAGEBREAK', '\n\n')
    series = series.str.replace(r' +', ' ', regex=True)
    
    # Strip whitespace
    series = series.str.strip()
    
    return series


def is_title(series):
    """
    Determines if a string meets specific title criteria.

    The function checks if the input text meets the following conditions:
    1. The text must be entirely in uppercase
    2. Contains at least one group of letters (ignoring non-alphabetic characters and spaces) that:
       - Has 5 or more letters
       - Contains at least 2 vowels (A, E, I, O, U)

    Args:
        series (pd.Series): Series of strings to check
        
    Returns:
        pd.Series: Boolean mask indicating which strings meet title criteria
    """
    # Check if all uppercase
    is_upper_mask = series == series.str.upper()
    
    # Remove non-letters and spaces
    letters_only = series.str.replace(r'[^A-Z]', '', regex=True)
    
    # Check length condition
    length_mask = letters_only.str.len() >= 5
    
    # Count vowels
    vowel_counts = letters_only.str.count('[AEIOU]')
    vowel_mask = vowel_counts >= 2
    
    return is_upper_mask & length_mask & vowel_mask

def split_and_reclassify(df):
    """
    splitting and reclassifying text content.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing 'content' and 'class' columns
        
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with split paragraphs and classifications
    """
    # Rename 'class' to 'original_class' before processing
    df = df.rename(columns={'class': 'original_class'})
    
    # Chain operations efficiently
    split_rows = (df
                 .assign(content=lambda x: x['content'].str.split('\n\n'))
                 .explode('content')
                 .assign(content=lambda x: x['content'].str.strip())
                 .query('content != ""'))
    
    # Add sub_order efficiently
    split_rows['sub_order'] = split_rows.groupby(split_rows.index).cumcount() + 1
    
    # Use vectorized title checking
    split_rows['class'] = np.where(is_title(split_rows['content']),
                                  'title',
                                  split_rows['original_class'])
    
    return split_rows.reset_index(drop=True)

def merge_consecutive_titles(df):
    """
    Merges consecutive title rows in a DataFrame that meet specific criteria.

    This function identifies and combines consecutive rows where:
    - Both rows have 'class' value of 'title'
    - Both rows share the same 'page_id' and 'box_page_id'
    - The 'sub_order' values are consecutive
    
    The content of merged rows is combined using newline characters.
    After merging, the sub_order values are recalculated within each page_id and box_page_id group.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - 'class': String column indicating the class of the row
        - 'page_id': Identifier for the page
        - 'box_page_id': Identifier for the box within the page
        - 'sub_order': Integer column indicating the order within the box
        - 'content': String column containing the text content

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with merged consecutive titles and recalculated sub_order values.
        The returned DataFrame maintains all original columns but with:
        - Merged rows combined into single rows
        - Updated sub_order values
        - Reset index

    Notes:
    ------
    - The function creates a copy of the input DataFrame to preserve the original
    - Merged content is joined with newline characters
    - The index is reset in the returned DataFrame
    """    
    if len(df) == 0:
        return df.copy()
    
    # Convert to numpy arrays for faster operations
    is_title = (df['class'] == 'title').values
    page_ids = df['page_id'].values
    box_page_ids = df['box_page_id'].values
    sub_orders = df['sub_order'].values
    
    # Create masks for consecutive matches using numpy operations
    consecutive_mask = np.zeros(len(df), dtype=bool)
    consecutive_mask[:-1] = (
        is_title[:-1] & 
        is_title[1:] & 
        (page_ids[:-1] == page_ids[1:]) &
        (box_page_ids[:-1] == box_page_ids[1:]) &
        (sub_orders[:-1] + 1 == sub_orders[1:])
    )
    
    # Early return if no consecutive titles found
    if not consecutive_mask.any():
        return df.copy()
    
    # Create groups for consecutive titles
    merge_groups = (~consecutive_mask).cumsum()
    
    # Create DataFrame with group information
    merge_df = pd.DataFrame({
        'content': df['content'],
        'merge_group': merge_groups,
        'keep': ~np.append(consecutive_mask[1:], False)
    })
    
    # Create result DataFrame first
    result_df = df[merge_df['keep']].copy()
    
    # Store original merge groups for kept rows
    kept_groups = merge_df[merge_df['keep']]['merge_group'].values
    
    # Merge content efficiently using pandas aggregation
    merged_contents = (merge_df
                      .groupby('merge_group')['content']
                      .agg('\n'.join)
                      .reindex(kept_groups)
                      .values)
    
    # Update content in result_df efficiently
    result_df['content'] = merged_contents
    
    # Reset index and recalculate sub_order
    result_df = result_df.reset_index(drop=True)
    result_df['sub_order'] = result_df.groupby(['page_id', 'box_page_id']).cumcount() + 1
    
    # Sort by reading_order and sub_order within each page_id group
    result_df = result_df.sort_values(['reading_order', 'sub_order'])
    
    # Recalculate reading_order sequentially
    result_df['reading_order'] = result_df.groupby('page_id').cumcount() + 1
    
    return result_df.reset_index(drop=True)

def create_articles(df):
    """
    Merges consecutive rows in a DataFrame, combining titles with their subsequent text.
    Creates a new 'title' column containing the title text while also including it in the main content.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing required columns:
        - 'class', 'page_id', 'reading_order', 'sub_order', 'content', 'issue_id'

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with merged content, titles integrated into content, and a new 'title' column.
    """    
    if len(df) == 0:
        return df.copy()

    # Create a working copy of the DataFrame
    working_df = df.copy()

    # Sort the DataFrame
    working_df = working_df.sort_values(['page_id', 'reading_order', 'sub_order']).reset_index(drop=True)

    # Initialize merge groups and titles
    current_group = 0
    merge_groups = []
    group_titles = {}
    current_title = None

    # Iterate through rows to create proper merge groups
    for i in range(len(working_df)):
        curr_row = working_df.iloc[i]

        if i == 0:
            if curr_row['class'] == 'title':
                current_title = curr_row['content']
            merge_groups.append(current_group)
            continue

        prev_row = working_df.iloc[i-1]

        # Start new group if issue_ids don't match
        if prev_row['issue_id'] != curr_row['issue_id']:
            current_group += 1
            current_title = None

        # Update current title if we encounter a title
        if curr_row['class'] == 'title':
            current_group += 1
            current_title = curr_row['content']

        merge_groups.append(current_group)
        if current_title is not None:
            group_titles[current_group] = current_title

    working_df['merge_group'] = merge_groups

    # Process groups and create new DataFrame
    result_rows = []

    for group in working_df['merge_group'].unique():
        group_df = working_df[working_df['merge_group'] == group]

        # Get the title for this group
        group_title = group_titles.get(group, None)

        # If group contains multiple rows or has a title
        if len(group_df) >= 1:
            # Get the first non-title row of the group
            first_row = group_df[group_df['class'] != 'title'].iloc[0] if any(group_df['class'] != 'title') else group_df.iloc[0]

            # Collect all content (excluding title rows)
            content_parts = []
            if group_title:
                content_parts.append(group_title)
            content_parts.extend(group_df[group_df['class'] != 'title']['content'].tolist())

            # Merge the content
            merged_content = '\n\n'.join(content_parts)

            # Create new row
            new_row = first_row.copy()
            new_row['content'] = merged_content
            new_row['title'] = group_title  # Add title column
            new_row['class'] = 'content'    # Set class to content
            result_rows.append(new_row)

    # Create new DataFrame from the Series objects
    result_df = pd.DataFrame([row.to_dict() for row in result_rows])

    # Reset index and recalculate reading_order
    result_df = result_df.sort_values(['reading_order', 'sub_order'])
    result_df['reading_order'] = result_df.groupby('page_id').cumcount() + 1

    # Drop the merge_group column
    result_df = result_df.drop('merge_group', axis=1)

    return result_df.reset_index(drop=True)



def process_document_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single batch (subset) of the DataFrame through all three functions.

    Parameters:
    -----------
    df_batch : pd.DataFrame
        A subset of the main DataFrame containing documents from a single issue.

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame batch.
    """
    # Apply the three functions in sequence
    df_batch = split_and_reclassify(df_batch)
    df_batch = merge_consecutive_titles(df_batch)
    df_batch = create_articles(df_batch)
    return df_batch

def process_batch(df: pd.DataFrame, issue_ids: List[int]) -> pd.DataFrame:
    """
    Process a batch of issue_ids.

    Parameters:
    -----------
    df : pd.DataFrame
        The entire DataFrame.
    issue_ids : List[int]
        List of issue_ids to process.

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame batch.
    """
    batch_df = df[df['issue_id'].isin(issue_ids)].copy()
    return process_document_batch(batch_df)

def process_documents(df: pd.DataFrame, batch_size: int = 100, parallel: bool = False, n_jobs: int = -1) -> pd.DataFrame:
    """
    Process the entire DataFrame in batches by issue_id, with option for parallel processing.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing documents.
    batch_size : int
        Number of unique issue_ids to process in each batch.
    parallel : bool
        Whether to use parallel processing.
    n_jobs : int
        Number of parallel jobs to run (-1 for all available cores).

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with all transformations applied.
    """
    # Get unique issue_ids
    unique_issues = df['issue_id'].unique()

    # Create batches of issue_ids
    issue_batches = [unique_issues[i:i + batch_size] for i in range(0, len(unique_issues), batch_size)]

    processed_dfs = []

    if parallel:
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_jobs if n_jobs != -1 else None) as executor:
            futures = [executor.submit(process_batch, df, batch) for batch in issue_batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                processed_dfs.append(future.result())
    else:
        # Process batches sequentially
        for batch in tqdm(issue_batches, desc="Processing batches"):
            processed_batch = process_batch(df, batch)
            processed_dfs.append(processed_batch)

    # Combine all processed batches
    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Sort the final DataFrame
    final_df = final_df.sort_values(['issue_id', 'page_id', 'reading_order', 'sub_order'])

    return final_df.reset_index(drop=True)
