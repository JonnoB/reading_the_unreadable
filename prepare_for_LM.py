import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # The process for preparing images with bounding boxes for sending to Pixtral

        This notebook finds the process to take a page image and convert it into a format that can be processed by the language model. 
        By this point the page will have had bounding boxes detected using docyolo-layout, and have been post processed. 

        This notebook shows the code necessary to 

        - Load an image or series of images
        - Crop out each region of interest
        - Split up the regions of interest
        - Create the batch json file
        - Send to Pixtral

        In this process each Issue will be a batch, this will allow for batches that are neither to large or too small, and contain all relevant information.

        I will trial the process on an issue of the leader specifically `CLD-1850-04-20`
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import os
    import cv2
    from pathlib import Path


    from bbox_functions import preprocess_bbox, save_plots_for_all_files
    return Path, cv2, os, pd, preprocess_bbox, save_plots_for_all_files


@app.cell
def __(os, pd, preprocess_bbox):
    folder_path = 'data/leader_test_issue/'

    output_folder = "data/leader_test_cropped"


    bbox_df = pd.read_parquet('data/periodical_bboxes/Leader_issue_PDF_files_1056.parquet')
    # Just subsetting to test issues of the leader to make sure the process works as expected
    test_issues = os.listdir(folder_path)

    # Just the subset of the files in the test set
    bbox_df = bbox_df.loc[bbox_df['filename'].isin(test_issues)]

    bbox_df['page_id'] = bbox_df['filename'].str.replace('.png', "")

    bbox_df = preprocess_bbox(bbox_df, 5)

    # All the preprossing of the data should be done in a single step and saved as it is probably quite slow.
    return bbox_df, folder_path, output_folder, test_issues


@app.cell
def __(bbox_df):
    bbox_df
    return


@app.cell
def __():
    return


@app.cell
def __(cv2, os):
    def crop_and_save_boxes(df, images_folder, output_folder):
        """
        Crop bounding boxes from images and save them as individual files.

        Parameters:
        df: DataFrame containing 'filename', 'page_id', 'x1', 'x2', 'y1', 'y2', 'box_page_id'
        images_folder: Path to folder containing the original images
        output_folder: Path where cropped images will be saved
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Group by filename to process each image once
        for filename, group in df.groupby('filename'):
            # Construct full path to image
            image_path = os.path.join(images_folder, filename)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            height, width = image.shape[:2]

            # Process each bounding box in the current image
            for _, row in group.iterrows():
                try:
                    # Get coordinates (ensure they are integers)
                    x1 = max(0, int(row['x1']))
                    y1 = max(0, int(row['y1']))
                    x2 = min(width, int(row['x2']))
                    y2 = min(height, int(row['y2']))

                    # Check if coordinates are valid
                    if x2 <= x1 or y2 <= y1:
                        print(f"Invalid coordinates for box {row['box_page_id']} in {filename}")
                        continue

                    # Crop the image
                    cropped = image[y1:y2, x1:x2]

                    # Check if cropped image is empty
                    if cropped.size == 0:
                        print(f"Empty crop for box {row['box_page_id']} in {filename}")
                        continue

                    # Create output filename
                    output_filename = f"{row['page_id']}_{row['box_page_id']}.png"
                    output_path = os.path.join(output_folder, output_filename)

                    # Save the cropped image
                    success = cv2.imwrite(output_path, cropped)
                    if not success:
                        print(f"Failed to save image for box {row['box_page_id']} in {filename}")

                except Exception as e:
                    print(f"Error processing box {row['box_page_id']} in {filename}: {str(e)}")
                    continue

    return (crop_and_save_boxes,)


@app.cell
def __(bbox_df, crop_and_save_boxes, folder_path, output_folder):
    # Run the cropping function
    crop_and_save_boxes( bbox_df, folder_path, output_folder)
    return


@app.cell
def __(bbox_df):
    bbox_df.loc[bbox_df['box_page_id'].isin(['B1C0R3', 'B0C0R1', 'B0C2R5']) & 
    bbox_df['page_id'].isin(['CLD-1850-04-20_page_24', 'CLD-1850-04-20_page_1'])]
    return


@app.cell
def __(bbox_df):
    bbox_df.columns
    return


@app.cell
def __():
    """
    save_plots_for_all_files(bbox_df, 
                             image_dir = 'data/leader_test_issue/', 
                             output_dir = 'data/leader_test_bounding', 
                             show_reading_order=True)

                             """
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
