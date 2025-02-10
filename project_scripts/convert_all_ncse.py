##############
##
## This script is run to convert the pdfs into smaller png files which can be used directly for machine learning.
## The PNGs are converted to 1 bit files to minimise the size.
## THe script is run twice once with dpi = 200 and once with dpi = 72. The 200 dpi can be used by Pixtral to
## extract the text, whilst the smaller 72 dpi can be used as training data to improve the text box detection on the
## pre-made bounding boxes.
##
############

from function_modules.util_funcs import convert_pdf_to_image
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import traceback

from pathlib import Path

# Change working directory to project root
os.chdir(Path(__file__).parent.parent)
image_folder = os.environ["image_folder"]
os.path.join(
    image_folder,
)

image_dpi = 120
save_folder = os.path.join(image_folder, f"converted/all_files_png_{image_dpi}")
source_folder = image_folder

# save_folder = f'data/converted/all_files_png_{image_dpi}'
# source_folder = os.path.join('data/all_periodicals')

subfolder_names = [
    "Northern_Star_issue_PDF_files",
    "Leader_issue_PDF_files",
    "Monthly_Repository_issue_PDF_files",
    "English_Womans_Journal_issue_PDF_files",
    "Publishers_Circular_issue_PDF_files",
    "Tomahawk_issue_PDF_files",
]

subfolder_names = ["Publishers_Circular_issue_PDF_files"]


# Define file paths
log_file = os.path.join(save_folder, "conversion_log.csv")
page_info_file = os.path.join(save_folder, "page_size_info.parquet")

# Initialize or load existing log
if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
else:
    log_df = pd.DataFrame(
        columns=["timestamp", "subfolder", "file", "status", "error_message"]
    )

# Load existing page info if it exists
if os.path.exists(page_info_file):
    existing_page_info = pd.read_parquet(page_info_file)
else:
    existing_page_info = pd.DataFrame()


def update_log(subfolder, file, status, error_message=""):
    global log_df
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subfolder": subfolder,
        "file": file,
        "status": status,
        "error_message": error_message,
    }
    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    log_df.to_csv(log_file, index=False)


def update_page_info(new_info, subfolder, file):
    global existing_page_info

    # Create DataFrame from new info
    new_df = pd.DataFrame(new_info)

    # Add subfolder and file information
    new_df["subfolder"] = subfolder
    new_df["source_file"] = file

    # Remove any existing entries for this file (in case of reprocessing)
    if not existing_page_info.empty:
        existing_page_info = existing_page_info[
            ~(
                (existing_page_info["subfolder"] == subfolder)
                & (existing_page_info["source_file"] == file)
            )
        ]

    # Append new information
    existing_page_info = pd.concat([existing_page_info, new_df], ignore_index=True)

    # Save to parquet
    backup_file = page_info_file + ".backup"
    try:
        existing_page_info.to_parquet(backup_file)
        os.replace(backup_file, page_info_file)
    except Exception as e:
        print(f"Error saving page info: {str(e)}")


for subfolder in subfolder_names:
    print(f"Processing the {subfolder} folder")
    subfolder_path = os.path.join(source_folder, subfolder)
    file_names = os.listdir(subfolder_path)
    save_subfolder = os.path.join(save_folder, subfolder)
    os.makedirs(save_subfolder, exist_ok=True)

    for file in tqdm(file_names):
        # Check if file was already successfully processed
        if (
            not log_df.empty
            and len(
                log_df[
                    (log_df["subfolder"] == subfolder)
                    & (log_df["file"] == file)
                    & (log_df["status"] == "success")
                ]
            )
            > 0
        ):
            continue

        try:
            page_info = convert_pdf_to_image(
                os.path.join(subfolder_path, file),
                output_folder=save_subfolder,
                dpi=image_dpi,
                image_format="PNG",
                use_greyscale=True,
            )

            update_page_info(
                page_info, subfolder, file
            )  # Save page info after each file
            update_log(subfolder, file, "success")
        except Exception as e:
            error_message = str(e) + "\n" + traceback.format_exc()
            update_log(subfolder, file, "failed", error_message)
