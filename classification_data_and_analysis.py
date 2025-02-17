import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        """
        Great little poem
        issue EWJ-1862-11-01 page 24 reading order 2

        I'm happy as the bee, love, 
        The winsome honey-bee, 
        In sunny hours from summer flowers 
        Drinking sweets so thirstingly.

        ISSUE EWJ-1858-09-01 PAGE 66 READING ORDER 8
        Let the luscious south wind Breathe in lover's sighs, While the lazy gallants, Bask in ladies' eyes.
        """
    )
    return


@app.cell
def _():
    import random
    import os
    import glob
    from typing import List, Tuple
    import re
    import pandas as pd
    import pandas as np
    from function_modules.helper_functions_class import (
        create_genre_prompt,
        create_iptc_prompt,
    )
    from function_modules.send_to_lm_functions import (
        download_processed_jobs,
        process_json_files,
    )
    from datetime import datetime
    import json
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    from mistralai import Mistral

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    return (
        List,
        Mistral,
        Tuple,
        api_key,
        classification_report,
        client,
        confusion_matrix,
        create_genre_prompt,
        create_iptc_prompt,
        datetime,
        download_processed_jobs,
        glob,
        json,
        np,
        os,
        pd,
        plt,
        process_json_files,
        random,
        re,
        sns,
    )


@app.cell
def _(os):
    dataset_folder = "data/download_jobs/ncse/dataframes/post_processed"

    all_data_parquets = os.listdir(dataset_folder)
    return all_data_parquets, dataset_folder


@app.cell
def _(all_data_parquets, dataset_folder, os, pd):
    # Define the path for saving/loading the processed dataset
    output_file = os.path.join("data/classification", "text_type_training_set.parquet")

    # Check if the processed file already exists
    if os.path.exists(output_file):
        # Load the existing processed dataset
        data_set = pd.read_parquet(output_file)
    else:
        # Create the data folder if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Process the data as before
        data_set = []

        for periodical in all_data_parquets:
            df = pd.read_parquet(os.path.join(dataset_folder, periodical))
            df = df.loc[
                df["class"].isin(["title", "text"]) & (df["completion_tokens"] > 30)
            ]
            df = df.sample(2000, random_state=1850)
            df["bbox_uid"] = df["page_id"] + "_" + df["box_page_id"]
            df = df[["bbox_uid", "content"]]

            data_set.append(df)

        data_set = pd.concat(data_set, ignore_index=True)

        # Save the processed dataset
        data_set.to_parquet(output_file)
    return data_set, df, output_file, periodical


@app.cell
def _(datetime, json, os, pd):
    def create_text_batch_job(
        client,
        base_filename,
        df,
        prompt_function,
        model="pixtral-12b-2409",
        job_type="testing",
        max_tokens=2000,
    ):
        """
        Create a batch job for text data from a DataFrame.

        Args:
            client: Mistral client instance
            base_filename: string of desired filename
            df: DataFrame containing 'bbox_uid' and 'content' columns
            prompt_function: function that generates the prompt for each text
            model: model to use for inference
            job_type: type of job (for metadata)
            max_tokens: maximum tokens for response

        Returns:
            tuple: (job_id, target_filename)
        """
        target_filename = f"{base_filename}.jsonl"

        # Create JSONL content
        jsonl_content = create_text_jsonl_content(df, prompt_function, max_tokens)

        # Convert string content to bytes
        content_bytes = jsonl_content.encode("utf-8")

        # Upload the file
        batch_data = client.files.upload(
            file={"file_name": target_filename, "content": content_bytes},
            purpose="batch",
        )

        # Create the job
        created_job = client.batch.jobs.create(
            input_files=[batch_data.id],
            model=model,
            endpoint="/v1/chat/completions",
            metadata={"job_type": job_type, "target_filename": target_filename},
        )

        return created_job.id, target_filename

    def create_text_jsonl_content(df, prompt_function, max_tokens=2000):
        """
        Create JSONL content for text data.

        Args:
            df: DataFrame containing 'bbox_uid' and 'content' columns
            prompt_function: function that generates the prompt for each text
            max_tokens: maximum tokens for response

        Returns:
            str: The JSONL content as a string
        """
        jsonl_lines = []

        for _, row in df.iterrows():
            prompt = prompt_function(row["content"])

            entry = {
                "custom_id": row["bbox_uid"],
                "body": {
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                },
            }
            jsonl_lines.append(json.dumps(entry))

        return "\n".join(jsonl_lines)

    def check_and_create_batch_job(
        client,
        df,
        prompt_function,
        job_type,
        tracking_dir,
        tracking_filename=None,
        model="mistral-large-latest",
        base_filename_prefix=None,
        max_tokens=2000,
        additional_metadata=None,
    ):
        """
        Check if job has been run before and create a new batch job if it hasn't.

        Args:
            client: Mistral client instance
            df: DataFrame containing data to process
            prompt_function: function to create prompts
            job_type: string identifying the type of job (e.g., 'genre_classification', 'sentiment_analysis')
            tracking_dir: directory path where tracking files should be stored
            tracking_filename: optional, specific name for the tracking file (default: {job_type}_job.csv)
            model: model to use for inference (default: "mistral-large-latest")
            base_filename_prefix: optional, prefix for the batch job filename
                                (default: derived from job_type)
            max_tokens: maximum tokens for response (default: 2000)
            additional_metadata: optional dict of additional metadata to store in tracking file

        Returns:
            tuple: (job_id, filename) or (None, None) if job already exists
        """
        # Set default values based on job_type if not provided
        if tracking_filename is None:
            tracking_filename = f"{job_type}_job.csv"

        if base_filename_prefix is None:
            base_filename_prefix = job_type.replace("_", "-")

        # Create full paths
        os.makedirs(tracking_dir, exist_ok=True)
        tracking_file = os.path.join(tracking_dir, tracking_filename)

        # Check if tracking file exists and load it
        if os.path.exists(tracking_file):
            print(f"Found existing job tracking file for {job_type}.")
            return None, None

        # Create a new job
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{base_filename_prefix}_{timestamp}"

        # Create the batch job
        job_id, filename = create_text_batch_job(
            client=client,
            base_filename=base_filename,
            df=df,
            prompt_function=prompt_function,
            model=model,
            job_type=job_type,
            max_tokens=max_tokens,
        )

        # Prepare job information
        job_info = {
            "job_id": [job_id],
            "filename": [filename],
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "model": [model],
            "job_type": [job_type],
            "num_records": [len(df)],
        }

        # Add additional metadata if provided
        if additional_metadata:
            for key, value in additional_metadata.items():
                job_info[key] = [value]

        # Save to CSV
        job_info_df = pd.DataFrame(job_info)
        job_info_df.to_csv(tracking_file, index=False)

        print(f"Created new {job_type} job {job_id} with filename {filename}")
        print(f"Job information saved to {tracking_file}")

        return job_id, filename

    # Usage examples:
    return (
        check_and_create_batch_job,
        create_text_batch_job,
        create_text_jsonl_content,
    )


@app.cell
def _(check_and_create_batch_job, client, create_genre_prompt, data_set):
    check_and_create_batch_job(
        client,
        data_set,
        prompt_function=create_genre_prompt,
        job_type="text_type_classification",
        tracking_dir="data/classification/silver_data/processed_data",
        tracking_filename=None,
        model="mistral-large-latest",
        base_filename_prefix=None,
        max_tokens=200,
        additional_metadata=None,
    )
    return


@app.cell
def _(client, download_processed_jobs, os, pd):
    # Define paths
    jobs_file = "data/classification/silver_data/processed_data/text_type_classification_job.csv"
    output_dir = "data/classification/silver_data/text_type"
    log_file = "data/classification/text_type_log.csv"

    # Read the jobs file to get the filename
    if os.path.exists(jobs_file):
        job_info = pd.read_csv(jobs_file)
        expected_jsonl = os.path.join(output_dir, job_info["filename"].iloc[0])

        # Check if the output file already exists
        if os.path.exists(expected_jsonl):
            print(f"Output file {expected_jsonl} already exists. Skipping download.")
        else:
            print(f"Downloading results to {expected_jsonl}")
            download_processed_jobs(
                client, jobs_file=jobs_file, output_dir=output_dir, log_file=log_file
            )
    else:
        print(f"Jobs file {jobs_file} not found. No jobs to download.")
    return expected_jsonl, job_info, jobs_file, log_file, output_dir


@app.cell
def _(json, pd):
    def process_dictionary_responses(jsonl_path):
        """
        Process JSON responses containing dictionary classifications and return a dictionary
        where keys are custom_ids and values are the classification dictionaries.

        Parameters:
        -----------
        jsonl_path : str
            Path to the JSON file containing the responses

        Returns:
        --------
        dict
            Dictionary with custom_ids as keys and classification dictionaries as values
        """
        try:
            # Load JSON data
            with open(jsonl_path, "r") as file:
                json_data = json.load(file)

            # Initialize results dictionary
            results = {}

            # Process each response
            for item in json_data:
                try:
                    custom_id = item["custom_id"]
                    content = item["response"]["body"]["choices"][0]["message"][
                        "content"
                    ]

                    # Clean up the content string to convert it to a dictionary
                    # Remove potential markdown formatting
                    content = (
                        content.replace("```json\n", "").replace("\n```", "").strip()
                    )

                    # Convert string representation of dictionary to actual dictionary
                    classification = eval(
                        content
                    )  # Be careful with eval - use only with trusted data

                    # Store in results
                    results[custom_id] = classification

                except Exception as e:
                    print(
                        f"Error processing item with custom_id {item.get('custom_id', 'unknown')}: {str(e)}"
                    )
                    continue

            return results

        except Exception as e:
            print(f"Error processing file {jsonl_path}: {str(e)}")
            return {}

    def classifications_dict_to_df(classifications_dict):
        """
        Convert a dictionary of classifications into a DataFrame.
        Handles cases where value dictionaries might be strings that need to be evaluated.

        Parameters:
        -----------
        classifications_dict : dict
            Dictionary where keys are custom_ids and values are classification dictionaries
            or string representations of dictionaries

        Returns:
        --------
        pd.DataFrame
            DataFrame with custom_id and all classification elements as columns
        """
        rows = []
        for custom_id, class_dict in classifications_dict.items():
            # Start with the custom_id
            row = {"custom_id": custom_id}

            try:
                # Handle string representation of dictionary
                if isinstance(class_dict, str):
                    # Clean up the string (remove potential JSON formatting)
                    class_dict = (
                        class_dict.replace("```json\n", "").replace("\n```", "").strip()
                    )
                    # Convert string to dictionary
                    class_dict = eval(
                        class_dict
                    )  # Be careful with eval - use only with trusted data

                # Update row with classification data
                if isinstance(class_dict, dict):
                    row.update(class_dict)
                else:
                    print(
                        f"Warning: Unexpected value type for custom_id {custom_id}: {type(class_dict)}"
                    )
                    row["value"] = class_dict

            except Exception as e:
                print(f"Error processing custom_id {custom_id}: {str(e)}")
                print(f"Value: {class_dict}")
                # Add the raw value to help with debugging
                row["raw_value"] = str(class_dict)

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows).rename(columns={"class": "class_code"})

        # Optional: Sort by custom_id
        df = df.sort_values("custom_id").reset_index(drop=True)

        return df
    return classifications_dict_to_df, process_dictionary_responses


@app.cell
def _(classifications_dict_to_df, data_set, process_dictionary_responses):
    classified_text_type = process_dictionary_responses(
        "data/classification/silver_data/text_type/text_type_classification.jsonl"
    )

    text_class_df = classifications_dict_to_df(classified_text_type)

    text_class_df["text_class"] = text_class_df["class_code"].map(
        {
            0: "news report",
            1: "editorial",
            2: "letter",
            3: "advert",
            4: "review",
            5: "poem/song/story",
            6: "other",
        }
    )

    text_class_df["text_class2"] = text_class_df["class_code"].map(
        {
            0: "article",
            1: "article",
            2: "article",
            3: "advert",
            4: "article",
            5: "poem/song/story",
            6: "other",
        }
    )

    class_to_int = {"article": 0, "advert": 1, "poem/song/story": 2, "other": 3}

    # Use the mapping to convert the textual class names into integer codes
    text_class_df["class_code2"] = text_class_df["text_class2"].map(class_to_int)

    text_class_df = text_class_df.merge(
        data_set, left_on="custom_id", right_on="bbox_uid"
    )
    text_class_df["issue_id"] = text_class_df["custom_id"].str.split(
        "_page", n=1, expand=True
    )[0]
    text_class_df = text_class_df.dropna(subset=["class_code"])
    text_class_df["class_code"] = text_class_df["class_code"].astype(int)
    sampled_issue_ids_sample_method = (
        text_class_df["issue_id"].drop_duplicates().sample(frac=0.8, random_state=1852)
    )

    # Step 2: Create a new boolean column 'train' using the sampled issue_ids
    text_class_df["is_train"] = text_class_df["issue_id"].isin(
        sampled_issue_ids_sample_method
    )

    text_class_df = text_class_df.drop(columns = ['custom_id', 'class_code', 'value', 'text_class', 'issue_id'])

    text_class_df = text_class_df.rename(columns = {'text_class2':'text_class', 'class_code2':'class_code'})

    text_class_df = text_class_df[['bbox_uid', 'content', 'class_code',  'is_train']]

    text_class_df.to_parquet("data/classification/silver_data/silver_text_type.parquet")
    return (
        class_to_int,
        classified_text_type,
        sampled_issue_ids_sample_method,
        text_class_df,
    )


@app.cell
def _(text_class_df):
    print(text_class_df.head())
    return


@app.cell
def _(text_class_df):
    text_class_df.groupby(["is_train", "class_code"]).size()
    return


@app.cell
def _(class_to_int, pd, text_class_df):
    test_set_preds = pd.read_parquet(
        "data/classification/predicted_text_type_testset.parquet"
    ).merge(text_class_df[["bbox_uid", "text_class", "is_train"]])

    test_set_preds = test_set_preds.loc[~test_set_preds["is_train"]]

    int_to_class = {v: k for k, v in class_to_int.items()}

    # Add human-readable labels to your dataframe
    test_set_preds.rename(columns={"predicted_class": "predicted_codes"}, inplace=True)
    test_set_preds["predicted_class"] = test_set_preds["predicted_codes"].map(
        int_to_class
    )
    return int_to_class, test_set_preds


@app.cell
def _(test_set_preds):
    test_set_preds.groupby("predicted_codes").size()
    return


@app.cell
def _(test_set_preds):
    test_set_preds
    return


@app.cell
def _(
    class_to_int,
    classification_report,
    confusion_matrix,
    int_to_class,
    plt,
    sns,
    test_set_preds,
):
    # Calculate overall metrics
    print("Classification Report:")
    print(
        classification_report(
            test_set_preds["text_class"], test_set_preds["predicted_class"]
        )
    )

    # Create and plot confusion matrix
    cm = confusion_matrix(
        test_set_preds["text_class"], test_set_preds["predicted_class"]
    )
    plt.figure(figsize=(10, 8))

    # Get labels in correct order
    labels = [int_to_class[i] for i in range(len(class_to_int))]

    # Plot the confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        #  xticklabels=labels,
        #    yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()  # This helps prevent label cutoff
    plt.show()

    # Look at some misclassifications
    misclassified = test_set_preds[
        test_set_preds["text_class"] != test_set_preds["predicted_class"]
    ]
    return cm, labels, misclassified


@app.cell
def _(mo):
    mo.md(
        """
        # Create IPTC silver training set

        Once the entire dataset has been created the IPTC silver training set can be sampled and created
        """
    )
    return


@app.cell
def _(os, pd):
    _text_classification_folder = "data/classification_predictions"

    all_text_classification_preds = os.listdir(_text_classification_folder)

    text_class_temp = pd.read_parquet(
        os.path.join(_text_classification_folder, all_text_classification_preds[1])
    )
    return all_text_classification_preds, text_class_temp


@app.cell
def _(os, pd):
    _output_file = os.path.join(
        "data/classification", "IPTC_class_training_set.parquet"
    )
    _dataset_folder = "data/classification/silver_data/text_IPTC"

    _all_data_parquets = os.listdir(_dataset_folder)

    # Check if the processed file already exists
    if os.path.exists(_output_file):
        # Load the existing processed dataset
        IPTC_data_set = pd.read_parquet(_output_file)
    else:
        IPTC_data_set = []

        for _periodical in _all_data_parquets:
            _df = pd.read_parquet(os.path.join(_dataset_folder, _periodical))
            _df = _df.loc[
                _df["class"].isin(["title", "text"])
                & (_df["completion_tokens"] > 30)
                & (_df["predicted_class"] == 0)
            ]
            _df = _df.sample(2000, random_state=1850)
            _df["bbox_uid"] = _df["page_id"] + "_" + _df["box_page_id"]
            _df = _df[["bbox_uid", "content"]]

            IPTC_data_set.append(_df)

        IPTC_data_set = pd.concat(IPTC_data_set, ignore_index=True)

        # Save the processed dataset
        IPTC_data_set.to_parquet(_output_file)
    return (IPTC_data_set,)


@app.cell
def _(
    IPTC_data_set,
    check_and_create_batch_job,
    client,
    create_iptc_prompt,
):
    check_and_create_batch_job(
        client,
        IPTC_data_set,
        prompt_function=create_iptc_prompt,
        job_type="text_IPTC_classification",
        tracking_dir="data/classification/silver_data/processed_data",
        tracking_filename=None,
        model="mistral-large-latest",
        base_filename_prefix=None,
        max_tokens=200,
        additional_metadata=None,
    )
    return


@app.cell
def _(client, download_processed_jobs, os, pd):
    # Define paths
    _jobs_file = "data/classification/silver_data/processed_data/text_IPTC_classification_job.csv"
    _output_dir = "data/classification/silver_data/text_IPTC"
    _log_file = "data/classification/text_IPTC_log.csv"

    # Read the jobs file to get the filename
    if os.path.exists(_jobs_file):
        _job_info = pd.read_csv(_jobs_file)
        _expected_jsonl = os.path.join(_output_dir, _job_info["filename"].iloc[0])

        # Check if the output file already exists
        if os.path.exists(_expected_jsonl):
            print(f"Output file {_expected_jsonl} already exists. Skipping download.")
        else:
            print(f"Downloading results to {_expected_jsonl}")
            download_processed_jobs(
                client, jobs_file=_jobs_file, output_dir=_output_dir, log_file=_log_file
            )
    else:
        print(f"Jobs file {_jobs_file} not found. No jobs to download.")
    return


@app.cell
def _(classifications_dict_to_df, pd, process_dictionary_responses):
    classified_IPTC_type = process_dictionary_responses(
        "data/classification/silver_data/text_IPTC/text_IPTC_classification.jsonl"
    )

    IPTC_class_df_raw = classifications_dict_to_df(classified_IPTC_type)[
        ["custom_id", "class_code"]
    ]

    # Define the mapping
    iptc_mapping = {
        0: "arts_culture_entertainment_media",
        1: "crime_law_justice",
        2: "disaster_accident_emergency",
        3: "economy_business_finance",
        4: "education",
        5: "environment",
        6: "health",
        7: "human_interest",
        8: "labour",
        9: "lifestyle_leisure",
        10: "politics",
        11: "religion",
        12: "science_technology",
        13: "society",
        14: "sport",
        15: "conflict_war_peace",
        16: "weather",
        17: "NA",
    }

    # Create one-hot encoding
    exploded_df = IPTC_class_df_raw["class_code"].explode()
    one_hot = pd.get_dummies(exploded_df)

    # Rename the columns using the mapping
    one_hot.columns = [iptc_mapping[int(col)] for col in one_hot.columns]

    # Group by original index
    one_hot_grouped = one_hot.groupby(level=0).max()

    # Join back to original dataframe
    IPTC_class_df = IPTC_class_df_raw.join(one_hot_grouped)
    return (
        IPTC_class_df,
        IPTC_class_df_raw,
        classified_IPTC_type,
        exploded_df,
        iptc_mapping,
        one_hot,
        one_hot_grouped,
    )


@app.cell
def _(IPTC_class_df):
    class_counts = IPTC_class_df.drop(["custom_id", "class_code"], axis=1).sum()

    # Sort in descending order to see most frequent first
    class_counts_sorted = class_counts.sort_values(ascending=False)

    class_counts_sorted
    return class_counts, class_counts_sorted


@app.cell
def _(mo):
    mo.md(r"""## Join on the content tidy and save""")
    return


@app.cell
def _(IPTC_class_df, IPTC_data_set):
    IPTC_class_df_out = IPTC_class_df.merge(
        IPTC_data_set, left_on="custom_id", right_on=["bbox_uid"]
    ).drop(columns=(["bbox_uid", "NA", "sport", "weather", "environment"]))

    IPTC_class_df_out["issue_id"] = IPTC_class_df_out["custom_id"].str.split(
        "_page", n=1, expand=True
    )[0]
    _sampled_issue_ids_sample_method = (
        IPTC_class_df_out["issue_id"]
        .drop_duplicates()
        .sample(frac=0.8, random_state=1852)
    )

    # Step 2: Create a new boolean column 'train' using the sampled issue_ids
    IPTC_class_df_out["is_train"] = IPTC_class_df_out["issue_id"].isin(
        _sampled_issue_ids_sample_method
    )
    return (IPTC_class_df_out,)


@app.cell
def _(IPTC_class_df_raw, IPTC_data_set):
    IPTC_class_df_for_train = IPTC_class_df_raw.merge(
        IPTC_data_set, left_on="custom_id", right_on=["bbox_uid"]
    ).drop(columns=["bbox_uid"])

    IPTC_class_df_for_train["issue_id"] = IPTC_class_df_for_train[
        "custom_id"
    ].str.split("_page", n=1, expand=True)[0]
    _sampled_issue_ids_sample_method = (
        IPTC_class_df_for_train["issue_id"]
        .drop_duplicates()
        .sample(frac=0.8, random_state=1852)
    )

    # Step 2: Create a new boolean column 'train' using the sampled issue_ids
    IPTC_class_df_for_train["is_train"] = IPTC_class_df_for_train["issue_id"].isin(
        _sampled_issue_ids_sample_method
    )

    IPTC_class_df_for_train = IPTC_class_df_for_train.rename(columns = {'custom_id':'bbox_uid'})

    IPTC_class_df_for_train = IPTC_class_df_for_train.drop(columns = ['issue_id'])

    IPTC_class_df_for_train = IPTC_class_df_for_train[['bbox_uid', 'content', 'class_code', 'is_train']]

    IPTC_class_df_for_train.to_parquet(
        "data/classification/silver_data/silver_IPTC_class.parquet"
    )
    return (IPTC_class_df_for_train,)


@app.cell
def _(IPTC_class_df_for_train):
    IPTC_class_df_for_train
    return


@app.cell
def _(IPTC_class_df_for_train):
    print(IPTC_class_df_for_train.head())
    return


@app.cell
def _(mo):
    mo.md("""# IPTC performance""")
    return


@app.cell
def _(iptc_mapping, pd, text_class_df):
    test_set_preds_IPTC = pd.read_parquet(
        "data/classification/predicted_IPTC_type_testset.parquet"
    ).merge(text_class_df[["bbox_uid", "text_class2", "is_train"]])

    # Create a new mapping dictionary for column renaming
    new_columns = {}

    # For each original column name
    for col in test_set_preds_IPTC.columns:
        # If the column starts with 'class_' and doesn't end with 'probability'
        if col.startswith("class_") and not col.endswith("probability"):
            # Extract the number from the column name
            num = int(col.split("_")[1])
            # Create new name using the iptc_mapping
            new_columns[col] = iptc_mapping[num]
        # If the column ends with 'probability'
        elif col.endswith("probability"):
            # Extract the number from the column name
            num = int(col.split("_")[1])
            # Create new name using the iptc_mapping and add '_probability'
            new_columns[col] = f"{iptc_mapping[num]}_probability"
        else:
            # Keep other columns as they are
            new_columns[col] = col

    # Rename the columns
    test_set_preds_IPTC = test_set_preds_IPTC.rename(columns=new_columns)
    return col, new_columns, num, test_set_preds_IPTC


@app.cell
def _(test_set_preds_IPTC):
    test_set_preds_IPTC
    return


@app.cell
def _(
    IPTC_class_df,
    classification_report,
    iptc_mapping,
    test_set_preds_IPTC,
):
    IPTC_gt_df = IPTC_class_df.sort_values("custom_id").loc[
        IPTC_class_df["custom_id"].isin(test_set_preds_IPTC["bbox_uid"])
    ][sorted(iptc_mapping.values())]

    IPTC_preds_df = test_set_preds_IPTC.sort_values("bbox_uid")[
        sorted(iptc_mapping.values())
    ]

    report = classification_report(
        IPTC_gt_df,
        IPTC_preds_df,
        target_names=list(iptc_mapping.values()),  # Using your mapping
        zero_division=0,
    )
    return IPTC_gt_df, IPTC_preds_df, report


@app.cell
def _(report):
    report
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
