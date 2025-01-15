
""" 

These functions are generally to do with the analysis and evaluation code


"""

import os
import pandas as pd
import numpy as np



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