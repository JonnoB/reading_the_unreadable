import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md("""# This script simply creates the images of the test pages with the bounding boxes overlaid""")
    return


@app.cell
def __():
    from PIL import Image, ImageDraw
    from function_modules.helper_functions import create_page_dict, scale_bbox
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import json
    import os

    bounding_boxes_json = 'data/test_ncse_bounding_boxes.json'

    image_folder = 'data/ncse_test_jpg/'
    save_image_folder = 'data/image_with_bounding/ncse_test'

    def find_matching_string(row_id, string_list):
        return next((s for s in string_list if row_id in s), None)

    meta_df = pd.read_csv('data/ncse_test_data_metafile.csv')

    meta_df['file_name'] = meta_df['page_id'].apply(lambda x: find_matching_string(str(x), os.listdir(image_folder)))



    with open(bounding_boxes_json) as file:
        bounding_boxes = json.load(file)

    def plot_bounding_boxes(image_path, bounding_boxes, original_size=None):
        """
        Load an image and plot bounding boxes on top of it.

        Args:
            image_path (str): Path to the image file
            bounding_boxes (dict): Dictionary of bounding boxes where each box is 
                                 a dict with 'x0', 'y0', 'x1', 'y1' coordinates
            original_size (tuple): Optional tuple of (width, height) of the original image
                                 If provided, bounding boxes will be scaled accordingly

        Returns:
            PIL.Image: Image with bounding boxes drawn on it
        """
        # Load the image
        image = Image.open(image_path)

        # Create a copy to draw on
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        # Get current image size
        current_size = image.size

        # For each bounding box
        for box_id, coords in bounding_boxes.items():
            # If original size is provided, scale the coordinates
            if original_size:
                [x0, y0, x1, y1] = scale_bbox(
                    [coords["x0"], coords["y0"], coords["x1"], coords["y1"]],
                    original_size,
                    current_size
                )
            else:
                x0, y0, x1, y1 = coords["x0"], coords["y0"], coords["x1"], coords["y1"]

            # Draw rectangle
            draw.rectangle([x0, y0, x1, y1], outline='red', width=2)

            # Optionally draw the box ID
            draw.text((x0, y0-20), str(box_id), fill='red')

        return draw_image

    # Example usage:
    def display_image_with_boxes(image_path, meta_df, page_id, bounding):
        """
        Display an image with its bounding boxes using matplotlib.

        Args:
            image_path (str): Path to the image file
            meta_df (pandas.DataFrame): DataFrame containing metadata
            page_id (str): ID of the page to process
        """
        # Get the original image size from meta_df
        original_size = (
            meta_df[meta_df['page_id'] == page_id]['width'].iloc[0],
            meta_df[meta_df['page_id'] == page_id]['height'].iloc[0]
        )

        # Create page dictionary for this specific page
        #page_dict = create_page_dict(meta_df)
        bounding_boxes = bounding[str(page_id)]

        # Plot the image with boxes
        image_with_boxes = plot_bounding_boxes(
            image_path,
            bounding_boxes,
            original_size
        )

        # Display using matplotlib
        plt.figure(figsize=(15, 20))
        plt.imshow(image_with_boxes)
        plt.axis('off')
        #plt.show()

    # Usage example:
    # display_image_with_boxes('path/to/image.png', meta_df, 'page_id_1')
    return (
        Image,
        ImageDraw,
        bounding_boxes,
        bounding_boxes_json,
        create_page_dict,
        display_image_with_boxes,
        file,
        find_matching_string,
        image_folder,
        json,
        meta_df,
        np,
        os,
        pd,
        plot_bounding_boxes,
        plt,
        save_image_folder,
        scale_bbox,
    )


@app.cell
def __(meta_df):
    meta_df
    return


@app.cell
def __(
    bounding_boxes,
    display_image_with_boxes,
    image_folder,
    meta_df,
    os,
    plt,
    save_image_folder,
):
    os.makedirs(save_image_folder, exist_ok=True)

    for index, row in meta_df[['file_name', 'page_id']].drop_duplicates().iterrows():

        target_file = os.path.join(image_folder, row['file_name'])
        save_file = os.path.join(save_image_folder, row['file_name'])

        display_image_with_boxes(image_path = target_file, 
                                 meta_df = meta_df, 
                                 page_id = row['page_id'], bounding= bounding_boxes)

        plt.savefig(os.path.join(save_image_folder, row['file_name']))
        plt.close()
    return index, row, save_file, target_file


@app.cell
def __(meta_df):
    meta_df
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
