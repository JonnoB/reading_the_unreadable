


import os 
import PyPDF2
import pandas as pd
import numpy as np
import csv
import json
import re
import time
from openai import OpenAI

import logging
# Set the logging level for httpx to WARNING to suppress informational messages
logging.getLogger("httpx").setLevel(logging.WARNING)

client = OpenAI()
##
##
## Functions for creating the test/dev/train sets
##
##


def get_tokens_symbols(directory):

    """
    Goes through all the parquet files and returns a dataframe of token and symbol counts a page level
    """


    df_list = []

    for filename in os.listdir(directory):

        file_path = os.path.join(directory, filename)

        df = pd.read_parquet(file_path)
        df = df.loc[:, ['publication_id', 'page_number','total_tokens', 'symbol_count', 'issue_id']]

        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    df = df.groupby(['publication_id', 'issue_id', 'page_number']).sum().reset_index()

    return df

def stratified_target_sampling(df, group_col, value_col, target_value):
    """
    Perform stratified random sampling from different groups in a DataFrame until
    the sum of the total value of the sampled groups exceeds a target value.

    Sampling stops for a group when adding another randomly selected row would
    cause the total value sum of that group's sampled rows to exceed the target value.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        group_col (str): The column name indicating the group of each observation.
        value_col (str): The column name containing the value of each observation.
        target_value (float): The target value that, once exceeded, stops further sampling from a group.

    Returns:
        pandas.DataFrame: A new DataFrame containing the balanced sampled data.
    """
    sampled_indices = []

    for group, group_df in df.groupby(group_col):
        current_sum = 0
        # Shuffle the DataFrame to ensure random ordering. Keep the original index.
        shuffled_df = group_df.sample(frac=1)
        
        for index, row in shuffled_df.iterrows():
            if current_sum + row[value_col] > target_value:
                # Stop if adding this row would exceed the target value
                break
            current_sum += row[value_col]
            # Append the index of the original DataFrame
            sampled_indices.append(index)

    # Construct the DataFrame from the chosen indices using the original DataFrame's indices
    sampled_df = df.loc[sampled_indices]

    return sampled_df


def identify_file(folder, date):
    """
    Identify the largest file in a given folder that contains a specific date string in its name.

    Args:
        folder (str): The path to the folder where the files are located.
        date (str): The date string to search for in the file names.

    Returns:
        str: The name of the largest file containing the date string, or None if no such file is found.
    """
    file_list = os.listdir(folder)
    filtered_list = [file for file in file_list if date in file]
    if filtered_list:
        # Get the full path of each file
        full_paths = [os.path.join(folder, file) for file in filtered_list]
        # Get the file sizes
        file_sizes = [os.path.getsize(path) for path in full_paths]
        # Find the index of the largest file
        largest_file_index = file_sizes.index(max(file_sizes))
        # Return the largest file
        return filtered_list[largest_file_index]
    else:
        return None
    

def find_pdf_path(image_path, folder, date):
    """
    Find the path of the largest PDF file in a given folder that contains a specific date string in its name.

    Args:
        image_path (str): The base path where the folder is located.
        folder (str): The name of the folder where the PDF files are located.
        date (str): The date string to search for in the file names.

    Returns:
        str: The full path of the largest PDF file containing the date string, or None if no such file is found.
    """
    target_folder = os.path.join(image_path, folder)
    file_name = identify_file(target_folder, date)
    if file_name:
        return os.path.join(target_folder, file_name)
    else:
        return None


def extract_pages_from_pdf(source_pdf_path, page_number, output_pdf_path, page_range=0):
    """
    Extracts specific pages from a PDF based on the target page and the specified page range.
    If the page range is 0, only the target page is extracted. If the page range is 1, the target page
    and the pages before and after it (if they exist) are extracted. The function saves the extracted
    pages to a new PDF file.

    Args:
        source_pdf_path (str): Path to the source PDF file.
        page_number (int): The target page number to extract (1-indexed).
        output_pdf_path (str): Path to save the extracted pages as a PDF.
        page_range (int, optional): The range of pages to extract around the target page. Default is 0.

    Raises:
        ValueError: If no pages could be extracted from the PDF.
    """
    # Adjust for 0-indexed page numbers
    page_number -= 1

    with open(source_pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        writer = PyPDF2.PdfWriter()

        # Determine the range of pages to extract
        start_page = max(page_number - page_range, 0)
        end_page = min(page_number + page_range, num_pages - 1)

        # Add the pages to the writer
        pages_added = False
        for page in range(start_page, end_page + 1):
            if page >= 0 and page < num_pages:
                writer.add_page(reader.pages[page])
                pages_added = True

        # If no pages were added, add the specified page (if it exists)
        if not pages_added and page_number >= 0 and page_number < num_pages:
            writer.add_page(reader.pages[page_number])

        # Write the output PDF file if at least one page was added
        if writer.pages:
            with open(output_pdf_path, 'wb') as output_file:
                writer.write(output_file)
        else:
            raise ValueError("No pages could be extracted from the PDF.")
        

def process_pdfs(temp, output_folder="output_folder", verbose=False, page_range = 0):
    """
    Processes each PDF specified in the DataFrame `temp`, extracting a specific
    page and saving it to a specified output folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through rows in 'pdf_path' column
    for index, row in temp.iterrows():
        pdf_path = row['pdf_path']
        page_to_extract = row['page_number']  # Assume this column exists and is correctly populated
        filename = row['file_name'].replace(".txt", ".pdf")
        destination_path = os.path.join(output_folder, filename)

        # Extract and save the specified page from the PDF
        try:
            extract_pages_from_pdf(pdf_path, page_to_extract, destination_path, page_range)
            if verbose:
                print(f"Extracted page {page_to_extract + 1} from '{pdf_path}' and saved as '{filename}' to {output_folder}")
        except Exception as e:
            if verbose:
                print(f"Error processing '{pdf_path}': {e}")
        

def load_articles_from_date(date, directory, min_tokens = 100):

    """
    Load and concatenate articles from specified directory where articles are from a given date 
    and have at least a minimum number of tokens.

    This function is used for tracking down the ID's of transcribed articles

    This function scans through all files in the provided directory, filters articles based on the issue date 
    and minimum token count, and returns a concatenated DataFrame of these articles.

    Parameters:
        date (str): The issue date for the articles to load in YYYY-MM-DD format.
        directory (str): The directory path where article files are stored.
        min_tokens (int, optional): The minimum number of tokens that an article must have. Defaults to 100.

    Returns:
        pandas.DataFrame: A DataFrame containing all articles that meet the criteria.

    """

    df_list = []

    for filename in os.listdir(directory):

        file_path = os.path.join(directory, filename)

        df = pd.read_parquet(file_path)
        
        df = df.loc[df['issue_date']==pd.to_datetime(date).date(), :]

        df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)

        df = df.loc[df['total_tokens']>= min_tokens, :]

    return df


##
##
## Functions related to classifying the documents
##
##
def files_to_df_core_func(folder_path):

    """
    Create a pandas DataFrame from text files in a folder.

    This function reads the content of all text files in the specified folder
    and creates a DataFrame with columns for the file name, content, slug,
    periodical, page number, and issue. The slug, periodical, page number, and
    issue are extracted from the standardized file name format.

    performs only the basic operation not useful for NCSE data

    Args:
        folder_path (str): The path to the folder containing the text files.

    Returns:
        pandas.DataFrame: A DataFrame 
    """
    # Initialize an empty list to store the data from each file
    data = []

    # Iterate over all the .txt files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the contents of the file
            with open(file_path, "r") as file:
                content = file.read()
            
            # Append the file name and content to the data list
            data.append({"file_name": file_name, "content": content})

    # Create a DataFrame from the data list
    df =  pd.DataFrame(data)

    return df            

def files_to_df_func(folder_path):

    """
    Create a pandas DataFrame from text files in a folder.

    This function reads the content of all text files in the specified folder
    and creates a DataFrame with columns for the file name, content, slug,
    periodical, page number, and issue. The slug, periodical, page number, and
    issue are extracted from the standardized file name format.

    Args:
        folder_path (str): The path to the folder containing the text files.

    Returns:
        pandas.DataFrame: A DataFrame with columns for file name, content,
            slug, periodical, page number, and issue.
    """

    # Create a DataFrame from the data list
    df =  files_to_df_core_func(folder_path)

    split_df = df['file_name'].str.split("_")

    df['slug'] = split_df.apply(lambda x: x[1])
    df['periodical'] = split_df.apply(lambda x: x[3])
    df['page_number'] = split_df.apply(lambda x: x[-1]).str.replace(".txt", "").astype(int)
    df['issue'] = df['file_name'].str.extract(r'issue_(.*?)_page', expand=False)

    return df

def preprocess_text(text):
    # Example preprocessing: substitute multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text


def perform_ner_on_text(text, ner_pipeline):
    # Preprocess the text first
    preprocessed_text = preprocess_text(text)
    # Then pass the preprocessed text to the NER pipeline
    return ner_pipeline(preprocessed_text)

def default_serializer(obj):
    """ Handles non-serializable objects. """
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def perform_ner_on_folder(target_folder, save_folder, ner_pipeline):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Prepare the path for the log file
    log_file_path = os.path.join(save_folder, 'processing_log.csv')
    
    # Open CSV file in append mode to write logs continuously
    with open(log_file_path, 'a', newline='') as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=['File Name', 'Status'])
        log_writer.writeheader()  # Write CSV header

        # List all files in the target folder
        target_files = os.listdir(target_folder)

        for file_name in target_files:
            if file_name.endswith(".txt"):
                # Prepare paths and default log entry
                save_file = os.path.join(save_folder, file_name.replace('.txt', '.json'))
                log_entry = {'File Name': file_name, 'Status': 'Processed'}
                
                if not os.path.exists(save_file):
                    file_path = os.path.join(target_folder, file_name)
                    try:
                        # Read the content of the file
                        with open(file_path, "r") as file:
                            content = file.read()
                        # Perform NER and serialize results
                        NER_dict = perform_ner_on_text(content, ner_pipeline)
                        with open(save_file, 'w') as file:
                            json.dump(NER_dict, file, default=default_serializer)
                    except Exception as e:
                        # Update log entry with error message
                        log_entry['Status'] = f"Error: {str(e)}"
                else:
                    # Update log entry if the file already exists
                    log_entry['Status'] = 'Skipped (Already exists)'

                # Write log entry to CSV file
                log_writer.writerow(log_entry)


def perform_ner_on_results(text_folder, downstream_folder, ner_pipeline):
    """ 
    Cycles through a folder of folders and performs NER on the text files within, saving results using the same folder structure
    and logging the outcomes to a CSV file immediately after each file is processed.
    """
    # Loop through each sub-folder in the main text folder
    for folder in os.listdir(text_folder):
        target_folder = os.path.join(text_folder, folder)  # Define the target folder path
        save_folder = os.path.join(downstream_folder, folder)  # Define where to save the results

        # Process text files in the target folder and save them in the save folder
        perform_ner_on_folder(target_folder, save_folder,ner_pipeline)


def load_json_to_dataframe(folder_path):
    """
    Load all JSON files from a specified folder into a pandas DataFrame.
    The DataFrame will have two columns: 'file_name' and 'NER'.
    
    Parameters:
        folder_path (str): The path to the folder containing JSON files.
    
    Returns:
        pd.DataFrame: A DataFrame containing the names of the files and their corresponding JSON data.
    """
    # Initialize a list to store the data
    data = []
    
    # Loop through each file in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                content = json.load(file)
            # Append the file name and its content to the list
            data.append({'file_name': file_name, 'NER': content})
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    return df



##
## For checking if cultural and social knowledge is being used.
##

def repeat_prompt_experiment(prompt, check_prompt, client, repetitions = 5, model = "gpt-4-turbo-preview", auto_check = False, temperature = 0.2):
    """
    Sends the same prompt to the OpenAI API multiple times to assess how the language model uses socio-cultural knowledge.

    This function sends a specified `prompt` to the OpenAI API, retrieves the response, and then sends a secondary prompt constructed with the initial response. This process is repeated a specified number of times. The responses, along with their log probabilities and perplexity, are recorded and returned in a pandas DataFrame.

    Parameters:
        prompt (str): The initial prompt to send to the language model.
        check_prompt (str): A format string that incorporates the response into a follow-up prompt for further evaluation.
        client (OpenAI Client): The client instance used to interact with the OpenAI API.
        repetitions (int, optional): The number of times the experiment should repeat. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing the response, sum of log probabilities, perplexity, and correctness evaluation for each repetition.
    """

    response_list = []

    for i in np.arange(0, repetitions):

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens = 500, 
            temperature = temperature,
            logprobs=True#,
           # top_p = 0.9,
        )
        #get the log probability for each token
        logprobs = [token.logprob for token in response.choices[0].logprobs.content]

        answer = response.choices[0].message.content
        is_correct = None
        if auto_check:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": check_prompt.format(answer=answer)}
            ]
            #The checking model only needs to be gpt-3.5 for speed
            response_check = client.chat.completions.create(
                model = "gpt-4-turbo-preview",#'gpt-3.5-turbo',#
                messages = messages,
                max_tokens = 500, 
                temperature = 0.2#,
            # top_p = 0.9,
            )
            is_correct = response_check.choices[0].message.content

        response_list.append({
            'response':answer,
            'logprobs':np.sum(logprobs),
            'perplexity':np.exp(-np.mean(logprobs)),
            'correct':is_correct})
        
    return pd.DataFrame(response_list)

def process_jokes_context(df, save_folder, repetitions=100, model="gpt-4-turbo-preview", temperature = 0.2):
    """
    Process jokes and their contexts, save results to specified folder.

    Args:
    df (pd.DataFrame): DataFrame containing jokes, contexts, and prompts.
    save_folder (str): Path to the folder where results should be saved.
    repetitions (int): Number of repetitions for the experiment (default is 100).
    model (str): Model identifier to use (default is 'gpt-3.5-turbo').

    Returns:
    None. Outputs are saved to files and progress is printed.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created directory: {save_folder}")

    context_results = []
    for index, row in df.iterrows():
        save_file = os.path.join(save_folder, row['joke'] + "_" + row['context'] + ".csv")
        if not os.path.exists(save_file):
            start_time = time.time()  # Record the start time

            # Assuming 'repeat_prompt_experiment' is defined elsewhere
            result = repeat_prompt_experiment(row['prompt'], row['check'], client, repetitions=repetitions, model=model, temperature = temperature)

            result['context'] = row['context']
            result['joke'] = row['joke']
            result['sentence'] = row['sentence']
            result['temperature'] = temperature

            context_results.append(result)

            end_time = time.time()  # Record the end time
            duration = end_time - start_time  # Calculate the duration

            # Save result to CSV and print progress
            result.to_csv(save_file)
            print(f"Processed joke: {row['joke']} with context {row['context']} in {duration:.2f} seconds")
        else:
            print(f"{os.path.basename(save_file)} already exists, skipping")