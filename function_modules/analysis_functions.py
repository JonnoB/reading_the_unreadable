
""" 

These functions are generally to do with the analysis and evaluation code


"""

import os
import pandas as pd
import re



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


def is_title(text):
    """
    Determines if a string meets specific title criteria.

    The function checks if the input text meets the following conditions:
    1. The text must be entirely in uppercase
    2. Contains at least one group of letters (ignoring non-alphabetic characters and spaces) that:
       - Has 5 or more letters
       - Contains at least 2 vowels (A, E, I, O, U)

    Args:
        text (str): The input string to be checked

    Returns:
        bool: True if the text meets all title criteria, False otherwise

    Examples:
        >>> is_title("HELLO WORLD")
        True  # "HELLOWORLD" has 5+ letters and 3 vowels
        >>> is_title("HI")
        False  # Too short and not enough vowels
        >>> is_title("Hello WORLD")
        False  # Not all uppercase
    """
    # Function to check if a string meets title criteria
    vowels = set('AEIOU')

    # Check if the text is all uppercase
    if not text.upper() == text:
        return False

    # Split by any character that's not a letter or space
    word_groups = re.split(r'[^A-Z\s]', text)

    # For each continuous group of words
    for group in word_groups:
        # Remove spaces and check if this group meets criteria
        letters_only = group.replace(' ', '')
        if len(letters_only) >= 5:  # Check length
            vowel_count = sum(1 for c in letters_only if c in vowels)
            if vowel_count >= 2:  # Check vowels
                return True
    return False

def split_and_reclassify(bbox_df):
    """
    Splits text content into paragraphs and reclassifies them based on specific criteria.

    This function processes a DataFrame containing bounding box information and text content.
    It splits the text content into paragraphs (based on double newlines), assigns order numbers,
    and potentially reclassifies paragraphs as titles based on their content.

    Parameters:
    -----------
    bbox_df : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - 'content': str, the text content to be split
        - 'class': str, the original classification of the text

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the following modifications:
        - Split paragraphs as separate rows
        - Added 'class2' column with potentially updated classifications
        - Added 'sub_order' column indicating the order of paragraphs
        - All original columns are preserved
        - Index is reset

    Notes:
    ------
    - Paragraphs are determined by double newline characters ('\n\n')
    - Empty paragraphs are skipped
    - If text contains only one paragraph, 'class2' will be same as original 'class'
    - For multiple paragraphs, each is evaluated using is_title() function to determine
      if it should be classified as a title

    """

    new_rows = []

    for idx, row in bbox_df.iterrows():
        text = row['content']
        paragraphs = text.split('\n\n')

        if len(paragraphs) == 1:
            row['class2'] = row['class']
            row['sub_order'] = 1
            new_rows.append(row)
        else:
            for i, para in enumerate(paragraphs, 1):
                if para.strip():  # Skip empty paragraphs
                    new_row = row.copy()
                    new_row['content'] = para.strip()
                    new_row['sub_order'] = i

                    # Check each paragraph against the new title criteria
                    if is_title(para.strip()):
                        new_row['class2'] = 'title'
                    else:
                        new_row['class2'] = row['class']

                    new_rows.append(new_row)

    # Create new dataframe from the processed rows
    new_df = pd.DataFrame(new_rows).reset_index(drop=True)

    return new_df


def merge_consecutive_titles(df):
    """
    Merges consecutive title rows in a DataFrame that meet specific criteria.

    This function identifies and combines consecutive rows where:
    - Both rows have 'class2' value of 'title'
    - Both rows share the same 'page_id' and 'box_page_id'
    - The 'sub_order' values are consecutive
    
    The content of merged rows is combined using newline characters.
    After merging, the sub_order values are recalculated within each page_id and box_page_id group.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - 'class2': String column indicating the class of the row
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
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Initialize list to store indices to drop
    indices_to_drop = []

    # Initialize variables to track current merge group
    current_content = []
    current_start_idx = None

    # Iterate through rows
    for i in range(len(result_df)):
        current_row = result_df.iloc[i]

        # If we're not at the last row, get next row for comparison
        if i < len(result_df) - 1:
            next_row = result_df.iloc[i + 1]

            # Check if current and next rows should be merged
            should_merge = (
                current_row['class2'] == 'title' and
                next_row['class2'] == 'title' and
                current_row['page_id'] == next_row['page_id'] and
                current_row['box_page_id'] == next_row['box_page_id'] and
                current_row['sub_order'] + 1 == next_row['sub_order']
            )

            if should_merge:
                # Start new merge group if not already started
                if current_start_idx is None:
                    current_start_idx = i
                    current_content = [current_row['content']]

                current_content.append(next_row['content'])
                indices_to_drop.append(i + 1)

            elif current_start_idx is not None:
                # Merge the accumulated content into the first row
                merged_content = '\n'.join(current_content)
                result_df.at[current_start_idx, 'content'] = merged_content

                # Reset tracking variables
                current_content = []
                current_start_idx = None

        elif current_start_idx is not None:
            # Handle the last merge group if exists
            merged_content = '\n'.join(current_content)
            result_df.at[current_start_idx, 'content'] = merged_content

    # Drop the merged rows using boolean indexing instead of index labels
    if indices_to_drop:
        result_df = result_df[~result_df.index.isin(indices_to_drop)]

    # Reset index
    result_df = result_df.reset_index(drop=True)

    # Recalculate sub_order within each box_page_id group
    result_df['sub_order'] = result_df.groupby(['page_id', 'box_page_id']).cumcount() + 1

    return result_df