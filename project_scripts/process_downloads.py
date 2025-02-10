"""
This script  converts the json files to dataframes and post processes the dataframes

"""

from function_modules.send_to_lm_functions import process_json_files
from function_modules.analysis_functions import remove_line_breaks, process_documents
import os
import pandas as pd

from pathlib import Path

# Change working directory to project root
os.chdir(Path(__file__).parent.parent)


path = "data/download_jobs/ncse"
dataframe_folder = "data/download_jobs/ncse/dataframes"
raw_dataframes = os.path.join(dataframe_folder, "raw")
post_processed_dataframe = os.path.join(dataframe_folder, "post_processed")
article_dataframe = os.path.join(dataframe_folder, "post_processed_articles")

os.makedirs(raw_dataframes, exist_ok=True)
os.makedirs(post_processed_dataframe, exist_ok=True)
os.makedirs(article_dataframe, exist_ok=True)

folders = [
    f
    for f in os.listdir(path)
    if os.path.isdir(os.path.join(path, f)) and f != "dataframes"
]


for folder in folders:
    json_folder = os.path.join(path, folder)
    output_parquet = os.path.join(raw_dataframes, f"{folder}.parquet")
    post_processed_output = os.path.join(post_processed_dataframe, f"{folder}.parquet")
    article_output_path = os.path.join(article_dataframe, f"{folder}.parquet")

    # Check if the output parquet file already exists
    if not os.path.exists(output_parquet):
        df = process_json_files(
            json_folder=json_folder,
            output_path=output_parquet,
            num_workers=None,  # Will use CPU count - 1
        )
        del df
    else:
        print(f"Skipping {folder} raw dataframe as {output_parquet} already exists")

    if not os.path.exists(post_processed_output):
        print(f"Load {folder} for post processing")

        returned_docs = pd.read_parquet(output_parquet)
        returned_docs["content"] = (
            returned_docs["content"].str.strip("`").str.replace("tsv", "", n=1)
        )
        # Add the condition to change 'class' to 'text'
        returned_docs.loc[
            (returned_docs["completion_tokens"] > 50)
            & (returned_docs["class"] == "title"),
            "class",
        ] = "text"
        returned_docs.loc[returned_docs["class"] != "table", "content"] = (
            remove_line_breaks(
                returned_docs.loc[returned_docs["class"] != "table", "content"]
            )
        )

        # save the file which is now cleaned up
        returned_docs.to_parquet(post_processed_output)

        print(f"Post-processing complete. Results saved to {post_processed_output}")

    else:
        print(
            f"Skipping {folder} post processed dataframe as {output_parquet} already exists"
        )
    if not os.path.exists(article_output_path):
        returned_docs = pd.read_parquet(post_processed_output)

        articles_df = process_documents(
            returned_docs, batch_size=1, parallel=True, n_jobs=1
        )

        articles_df.to_parquet(article_output_path)

        del returned_docs

        del articles_df

    else:
        print(f"Skipping {folder} article dataframe as {output_parquet} already exists")
