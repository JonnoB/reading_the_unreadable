import jellyfish
import numpy as np
import pandas as pd
import os
from nervaluate import Evaluator
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from helper_functions import load_json_to_dataframe

def vectorized_levenshtein_similarity(leven_distances, lengths):
    """
    Vectorized calculation of Levenshtein similarity based on arrays of Levenshtein distances
    and the corresponding lengths of strings.

    Parameters:
    - leven_distances (np.array): An array of Levenshtein distances.
    - lengths (np.array): An array of the maximum lengths between compared strings.

    Returns:
    - np.array: An array of Levenshtein similarities.
    """
    # Ensure zero division is handled by returning 0 similarity in such cases
    similarities = np.where(lengths > 0, (lengths - leven_distances) / lengths, 0)
    return similarities

# Example usage within the calculate_metrics function
def calculate_metrics(file_name, raw_data_string, contents, wer, cer):
    wer_score = wer.compute(predictions=[raw_data_string], references=[contents])
    cer_score = cer.compute(predictions=[raw_data_string], references=[contents])
    leven_score = jellyfish.levenshtein_distance(raw_data_string, contents)
    
    # Here we prepare for vectorized computation
    leven_distances = np.array([leven_score])
    lengths = np.array([max(len(raw_data_string), len(contents))])
    
    #This is not very useful as I can use the error rate reduction
    #lev_sim = vectorized_levenshtein_similarity(leven_distances, lengths)[0]  # Get the first element since we're dealing with single values

    results_df = pd.DataFrame({
        'File Name': [file_name],
        'WER': [wer_score],
        'CER': [cer_score],
        'lev_dist': [leven_score]#,
    #    'lev_sim': [lev_sim]
    })

    return results_df


def evaluate_ocr_dataframe(dev_data_raw_df, dev_transcripts, wer, cer):
    """
    Evaluate a dataframe to calculate Word Error Rate (WER), Character Error Rate (CER),
    and Levenshtein Distance for each entry. It assumes each entry includes a file name
    pointing to a transcript file, and raw data content for comparison.

    WARNING: Only pre-processing is convert to lower, on bot raw data and recovered data

    Parameters:
    - dev_data_raw_df (pd.DataFrame): A DataFrame containing the dataset to be processed.
      It must include 'file_name' and 'content_html' columns.
    - dev_transcripts (str): The directory path where the transcript files referenced in
      dev_data_raw_df are stored.

    Returns:
    - pd.DataFrame: A combined DataFrame containing the file name, WER, CER, and Levenshtein
      Distance for each file in the dataset.

    This function iterates through each row in the input DataFrame, reads the transcript file
    contents, and uses the `calculate_metrics` function to calculate the necessary metrics. It
    compiles the results from all files into a single DataFrame.
    """

    results_list = []  # To store the dataframes generated for each file
    
    # Loop through each row in the DataFrame
    for index, row in dev_data_raw_df.iterrows():
        file_name = row['file_name']
        raw_data_string = row['content_html']
        file_path = os.path.join(dev_transcripts, file_name)

        # Open the file and read its contents
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = file.read()

        # Calculate metrics and append the resulting dataframe to the list
        metrics_df = calculate_metrics(file_name, raw_data_string.lower(), contents.lower(), wer, cer)
        results_list.append(metrics_df)
    
    # Combine all the DataFrames in the list into a single DataFrame
    combined_df = pd.concat(results_list, ignore_index=True)
    
    return combined_df


def load_txt_files_to_df(directory):
    """
    Loads the content of all '.txt' files within a specified directory into a pandas DataFrame.

    Parameters:
    - directory (str): The path to the directory containing '.txt' files.

    Returns:
    - pd.DataFrame: A DataFrame with a single column "content_html", where each row contains
      the content of one '.txt' file from the directory.
    """
    # Initialize a list to store the content of each text file
    content_list = []
    
    # Loop through each file in the directory
    for file_name in os.listdir(directory):
        # Check if the file is a '.txt' file
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)
            # Open the file and read its contents
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                content_list.append({'content_html': content, 'file_name':file_name})
    
    # Create a DataFrame with the contents
    df = pd.DataFrame(content_list)
    
    return df

def percentage_error_reduction(old_errors, new_errors):
  """
  Calculates the percentage reduction in error between two vectors.

  Args:
      old_errors (numpy.ndarray): A numpy array containing the old error rates.
      new_errors (numpy.ndarray): A numpy array containing the new error rates.

  Returns:
      numpy.ndarray: A numpy array containing the percentage reduction for each element.
  """

  # Handle division by zero with np.where
  zero_mask = np.where(old_errors == 0, True, False)
  reduction = np.where(zero_mask, float('inf'), (old_errors - new_errors) / old_errors * 100)
  return reduction


def get_metric_error_reduction(corrected_df, raw_df):
    """
    Calculate the percentage error reduction for WER, CER, and lev_dist between corrected and raw DataFrames.

    Args:
        corrected_df (pandas.DataFrame): DataFrame containing the corrected metrics (WER, CER, lev_dist).
        raw_df (pandas.DataFrame): DataFrame containing the raw metrics (WER, CER, lev_dist).

    Returns:
        pandas.DataFrame: A new DataFrame with columns 'File Name', 'WER_reduction', 'CER_reduction',
                          and 'lev_dist_reduction', representing the percentage error reduction for each file
                          and metric.

    Example:
        >>> corrected_df = pd.DataFrame({'File Name': ['file1', 'file2'], 'WER': [0.1, 0.2], 'CER': [0.05, 0.1], 'lev_dist': [10, 20]})
        >>> raw_df = pd.DataFrame({'File Name': ['file1', 'file2'], 'WER': [0.2, 0.3], 'CER': [0.1, 0.15], 'lev_dist': [20, 30], 'type': ['raw', 'raw']})
        >>> get_metric_error_reduction(corrected_df, raw_df)
           File Name  WER_reduction  CER_reduction  lev_dist_reduction
        0      file1           50.0           50.0                50.0
        1      file2           33.3           33.3                33.3
    """
    merged_df = corrected_df.merge(raw_df.drop(columns='type'), on='File Name', suffixes=['_corrected', '_raw'])
    
    # Calculate the percentage error reduction for each column
    wer_reduction = percentage_error_reduction(merged_df['WER_raw'], merged_df['WER_corrected'])
    cer_reduction = percentage_error_reduction(merged_df['CER_raw'], merged_df['CER_corrected'])
    lev_dist_reduction = percentage_error_reduction(merged_df['lev_dist_raw'], merged_df['lev_dist_corrected'])
    
    # Create a new DataFrame with the calculated relative errors
    relative_errors_df = pd.DataFrame({
        'File Name': merged_df['File Name'],
        'type':merged_df['type'],
        'WER': wer_reduction,
        'CER': cer_reduction,
        'lev_dist': lev_dist_reduction
    })
    
    return relative_errors_df


def evaluate_correction_performance(folder, transcripts_dir, wer_func, cer_func, type, remove_line_breaks = True):
    """
    Evaluate the performance of the OCR relative to the transcript ground truth.

    Args:
        folder (str): The directory path where the raw OCR text files are located.
        transcripts_dir (str): The directory path where the corresponding transcripts are located.
        wer_func (function): The function to calculate the Word Error Rate (WER).
        cer_func (function): The function to calculate the Character Error Rate (CER).
        type (str): The type of data being processed.

    Returns:
        pandas.DataFrame: The processed DataFrame containing the evaluation metrics.

    """
    dev_data_raw_df = load_txt_files_to_df(folder)

    if remove_line_breaks:
        dev_data_raw_df['content_html'] = dev_data_raw_df['content_html'].str.replace("\n", " ")
    
    # filter to only have the ones with a transcript
    dev_data_raw_df = dev_data_raw_df.loc[dev_data_raw_df['file_name'].isin(os.listdir(transcripts_dir))].reset_index(drop=True)
    
    eval_temp = evaluate_ocr_dataframe(dev_data_raw_df, transcripts_dir, wer_func, cer_func)
    eval_temp['type'] = type
    
    return eval_temp

def evaluate_correction_performance_folders(corrected_folder, transcript_folder, wer_func, cer_func, remove_line_breaks = True):
    """
    Calls `evaluate_correction_performance` on a folder of folders, allows the comparison of multiple models or prompts
    Evaluate the performance of LLM post-OCR recovery using WER, CER, and levenstiend distance

    This function uses the files in the corrected and transcript folders to evaluate how well the post-OCR correction has gone.
    Each file is evaluated and tagged with the file name for easy reference, sub-folders are tagged as 'type'

    Args:
        corrected_folder (str): The folder containing the corrected data.
        transcript_folder (str): The folder containing the ground truth transcripts.
        wer_func (function): The function to calculate the Word Error Rate (WER).
        cer_func (function): The function to calculate the Character Error Rate (CER).

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each folder.

    """
    performance_eval = []
    for folder in os.listdir(corrected_folder):
        
        eval_temp = evaluate_correction_performance(os.path.join(corrected_folder, folder),transcript_folder , wer_func, cer_func, folder, remove_line_breaks)

        performance_eval.append(eval_temp)

    performance_eval = pd.concat(performance_eval)
    return performance_eval


def calculate_entity_similarity(predicted_entities, ground_truth_entities):

    """
    Calculate the cosine similarity between the frequency vectors of entities extracted from predicted and ground truth data.

    This function first extracts the 'word' element from each entity in the predicted and ground truth entity lists. It then counts the occurrences of each unique word in both lists and constructs vectors from these counts. The cosine similarity between these two vectors is computed to measure how similar the entities are in terms of their content, regardless of their order or exact match.

    Args:
        predicted_entities (list of dict): A list of dictionaries where each dictionary represents an entity with at least the key 'word'.
        ground_truth_entities (list of dict): A list of dictionaries similar to `predicted_entities`, representing the ground truth for comparison.

    Returns:
        float: The cosine similarity between the frequency vectors of the predicted and ground truth entities.

    Example:
        predicted_entities = [
            {'entity': 'B-ORG', 'score': 0.6054342, 'index': 31, 'word': 'Revenge', 'start': 123, 'end': 130},
            {'entity': 'B-MISC', 'score': 0.50727797, 'index': 85, 'word': 'San', 'start': 362, 'end': 365}
        ]
        ground_truth_entities = [
            {'entity': 'B-ORG', 'score': 1.0, 'index': 31, 'word': 'Revenge', 'start': 123, 'end': 130},
            {'entity': 'B-MISC', 'score': 1.0, 'index': 85, 'word': 'Francisco', 'start': 362, 'end': 371}
        ]
        similarity = calculate_entity_similarity(predicted_entities, ground_truth_entities)
        print(f"Entity similarity: {similarity:.2f}")
    """
    # Extract words from predicted and ground truth entities
    predicted_words = [entity['word'] for entity in predicted_entities]
    ground_truth_words = [entity['word'] for entity in ground_truth_entities]
    
    # Count occurrences of words in predicted and ground truth entities
    predicted_counts = Counter(predicted_words)
    ground_truth_counts = Counter(ground_truth_words)
    
    # Create a list of all unique words present in either predicted or ground truth entities
    all_words = list(set(predicted_words + ground_truth_words))
    
    # Create vectors based on the counts of words
    predicted_vector = np.array([predicted_counts[word] for word in all_words])
    ground_truth_vector = np.array([ground_truth_counts[word] for word in all_words])
    
    # Compute cosine similarity between the vectors
    similarity = cosine_similarity([predicted_vector], [ground_truth_vector])[0][0]
    
    return similarity



def evaluate_ner_dict(ground_truth, predicted):
    """
    Evaluate the Named Entity Recognition (NER) predictions against the ground truth labels
    for a single pair of ground truth and predicted dictionaries.

    Parameters:
    - ground_truth (list): A list of dictionaries representing the ground truth annotations.
    - predicted (list): A list of dictionaries representing the predicted annotations.

    Returns:
    - float: The F1 score for the given pair of dictionaries.
    """
    ground_truth_labels = []
    predicted_labels = []
    if ground_truth:
        ground_truth_labels.append([{
            'label': entity['entity'],
            'start': entity['start'],
            'end': entity['end']
        } for entity in ground_truth])
    else:
        ground_truth_labels.append([])

    if predicted:
        predicted_labels.append([{
            'label': entity['entity'],
            'start': entity['start'],
            'end': entity['end']
        } for entity in predicted])
    else:
        predicted_labels.append([])

    tags = ['B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
    evaluator = Evaluator(ground_truth_labels, predicted_labels, tags=tags)

    # Calculate evaluation metrics for the dictionaries
    results, _ = evaluator.evaluate()
    return results['ent_type']['f1']


def evaluate_ner_dataframes(recovered_NER, gt_NER):
    """
    Evaluate the Named Entity Recognition (NER) performance by comparing recovered NER data with ground truth NER data.

    This function merges the ground truth and recovered NER data, calculates the similarity and F1 scores for each entity, and returns a summary DataFrame.

    Parameters:
    - recovered_NER (DataFrame): A pandas DataFrame containing the recovered NER data, indexed or containing a 'file_name' column to match with ground truth.
    - gt_NER (DataFrame): A pandas DataFrame containing the ground truth NER data, similarly indexed or structured.

    Returns:
    - DataFrame: A pandas DataFrame containing the original 'file_name', with calculated 'CoNES' (coefficient of NER similarity) and 'F1_score' for each file, excluding original NER columns to streamline the output.

    Notes:
    - Ensure that both DataFrames include a 'file_name' column for proper merging.
    - The function internally copies the DataFrames to avoid modifying the original data.
    """
    gt_NER = gt_NER.copy()

    recovered_NER = recovered_NER.copy()

    temp_ner = gt_NER.merge(recovered_NER, on='file_name', suffixes=['_gt', '_recovered'])

    temp_ner['CoNES'] = temp_ner.apply(lambda row: calculate_entity_similarity(row['NER_recovered'], row['NER_gt']), axis=1)

    temp_ner['F1_score'] = temp_ner.apply(lambda row: evaluate_ner_dict(row['NER_gt'], row['NER_recovered']), axis=1)

    temp_ner = temp_ner.drop(columns=['NER_gt', 'NER_recovered'])
    
    return temp_ner


def evaluate_ner_dataset(folder_path, gt_NER):
    """
    Evaluates Named Entity Recognition (NER) performance by comparing ground truth data with predictions found in a structured directory.
    
    This function assumes a specific directory structure where each subfolder in the given folder path contains prediction files.
    It calculates the F1 score and a custom similarity metric (CoNES score) for each document's NER predictions against the ground truth.
    The results are compiled into a single pandas DataFrame that includes the evaluation metrics for each document across all subfolders.

    Parameters:
    - folder_path (str): The path to the main folder containing subfolders with JSON prediction files.
    - gt_NER (pandas.DataFrame): A DataFrame containing the ground truth data with a 'file_name' column for merging.

    Returns:
    - pandas.DataFrame: A DataFrame with the evaluation results (F1 score and CoNES score) for each document, 
      along with a 'type' column indicating the subfolder from which the predictions were loaded.

    Raises:
    - FileNotFoundError: If the specified folder path does not exist.
    - ValueError: If there are discrepancies in the expected structure or content of the data files.

    Example:
    --------
    # Assuming 'ground_truth_data' is a DataFrame containing the ground truth NER tags
    # with a 'file_name' column that corresponds to filenames stored in the prediction folders.
    folder_path = 'path/to/NER_predictions'
    results_df = evaluate_ner_dataset(folder_path, ground_truth_data)
    print(results_df.head())
    """
    output = []

    dataset_folder = folder_path

    for folder in os.listdir(dataset_folder):


        target_folder = os.path.join(dataset_folder, folder)

        recovered_NER = load_json_to_dataframe(target_folder)

        temp_ner = evaluate_ner_dataframes(recovered_NER, gt_NER)

        temp_ner['type'] = folder

        output.append(temp_ner)

    return pd.concat(output, ignore_index=True)




