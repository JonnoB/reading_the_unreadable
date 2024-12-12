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
        """
        # Visualise pages
        This notebook is simply to help visualise the pages with bounding boxes on top
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import os
    import numpy as np
    from helper_functions import scale_bbox

    converted_folder = '/media/jonno/ncse/converted/all_files_png_72'


    bbox_data_df = pd.read_parquet(os.path.join('data', 'ncse_data_metafile.parquet'))
    page_conversion_df = pd.read_parquet(os.path.join(converted_folder, 'page_size_info.parquet'))
    page_conversion_df['filename'] = page_conversion_df['output_file'].apply(os.path.basename)

    file_name_to_id_map = pd.read_parquet('data/file_name_to_id_map.parquet').merge(page_conversion_df, on = 'filename')

    periodical_folders = pd.DataFrame({'folder_name':['English_Womans_Journal_issue_PDF_files', 'Leader_issue_PDF_files', 'Monthly_Repository_issue_PDF_files',
                         'Northern_Star_issue_PDF_files', 'Publishers_Circular_issue_PDF_files', 'Tomahawk_issue_PDF_files'],
                         'publication_id':[24,20, 22, 27, 26, 19]})
    return (
        Image,
        bbox_data_df,
        converted_folder,
        file_name_to_id_map,
        np,
        os,
        page_conversion_df,
        patches,
        pd,
        periodical_folders,
        plt,
        scale_bbox,
    )


@app.cell
def __(bbox_data_df):
    bbox_data_df
    return


@app.cell
def __(Image, patches, pd, plt, scale_bbox):
    def plot_image_with_boxes(
        #image_root: str,
        image_df: pd.DataFrame,
        boxes_df: pd.DataFrame,
        page_id: str,
        figsize=(10, 10),
        box_color='red',
        box_linewidth=2,
        save_path=None,
        padding_color='white',
        original_size=None,  
        new_size=None      
    ):
        """
        Plot an image with its corresponding bounding boxes, padding the image if boxes extend beyond frame.

        Parameters:
        - image_root: str, root directory containing the images
        - image_df: DataFrame with columns ['page_id', 'filename']
        - boxes_df: DataFrame with columns ['page_id', 'x0', 'x1', 'y0', 'y1']
        - page_id: str, the ID of the page to plot
        - figsize: tuple, size of the figure (width, height)
        - box_color: str, color of the bounding boxes
        - box_linewidth: int, width of the bounding box lines
        - save_path: str or None, if provided, saves the plot to this path
        - padding_color: str, color of the padding
        """

        # Get the filename for this page_id
        #filename = image_df[image_df['page_id'] == page_id]['filename'].iloc[0]
        #image_path = os.path.join(image_root, filename)
        image_path = image_df[image_df['page_id'] == page_id]['output_file'].iloc[0]

        # Load the image
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Get boxes for this page_id
        page_boxes = boxes_df[boxes_df['page_id'] == page_id].copy()  # Create a copy to avoid modifying original

        # Apply scaling if parameters are provided
        if original_size and new_size:
            page_boxes = scale_bbox(page_boxes, original_size, new_size)
            # Use the scaled coordinates
            coord_cols = ['scaled_x0', 'scaled_x1', 'scaled_y0', 'scaled_y1']
        else:
            # Use the original coordinates
            coord_cols = ['x0', 'x1', 'y0', 'y1']

        # Calculate required padding using the appropriate coordinates
        min_x = min(page_boxes[coord_cols[0]].min()-20, 0)
        min_y = min(page_boxes[coord_cols[2]].min()-20, 0)
        max_x = max(page_boxes[coord_cols[1]].max()+20, img_width)
        max_y = max(page_boxes[coord_cols[3]].max()+20, img_height)

        # Calculate padding dimensions
        pad_left = abs(min(min_x, 0))
        pad_top = abs(min(min_y, 0))
        pad_right = max(max_x - img_width, 0)
        pad_bottom = max(max_y - img_height, 0)

        # Create new image with padding
        new_width = int(img_width + pad_left + pad_right)
        new_height = int(img_height + pad_top + pad_bottom)

        # Create new image with padding
        if img.mode == 'RGB':
            new_img = Image.new('RGB', (new_width, new_height), padding_color)
        else:
            new_img = Image.new('L', (new_width, new_height), padding_color)

        # Paste original image onto padded image
        new_img.paste(img, (int(pad_left), int(pad_top)))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(new_img)

        # Draw each bounding box with adjusted coordinates
        for _, box in page_boxes.iterrows():
            rect = patches.Rectangle(
                (box[coord_cols[0]] + pad_left, box[coord_cols[2]] + pad_top),
                box[coord_cols[1]] - box[coord_cols[0]],
                box[coord_cols[3]] - box[coord_cols[2]],
                linewidth=box_linewidth,
                edgecolor=box_color,
                facecolor='none'
            )
            ax.add_patch(rect)

        # Remove axes
        ax.axis('off')

        # Tight layout to remove extra white space
        plt.tight_layout()

        # Save if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
    return (plot_image_with_boxes,)


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map.loc[file_name_to_id_map['publication_id']==20]

    #file_name_to_id_map.loc[file_name_to_id_map['periodical_abbrev']== 'NS3']
    return


@app.cell
def __(file_name_to_id_map):
    (file_name_to_id_map.groupby('periodical_abbrev')['overlap_fract'].apply(lambda x: (x > 0.1).astype(int).mean()))
    return


app._unparsable_cell(
    r"""
    page_id = 106249 

    #167499 NS3 (1307*1.805, 2197*1.805)
    #168959 NS4
    165363

    bboxes = bbox_data_df.loc[bbox_data_df['page_id']==page_id].iloc[[4,1##]] #2,7

    pub_id = bboxes['publication_id'].unique()[0]


    size_df = file_name_to_id_map.loc[file_name_to_id_map['page_id']==page_id].filter(regex = 'width|height')
    print(size_df)

    bboxes
    """,
    name="__"
)


@app.cell
def __():
    (1307*1.805, 2197*1.805)
    return


@app.cell
def __(
    bboxes,
    file_name_to_id_map,
    page_id,
    plot_image_with_boxes,
    size_df,
):
    plot_image_with_boxes(file_name_to_id_map,
                          bboxes,
                          page_id,    
                          original_size=(size_df['width'].values, size_df['height'].values),#(2359, 3965),  
                          new_size=  (size_df['final_width'].values, size_df['final_height'].values)#(size_df['final_width'].values, size_df['final_height'].values)
                         )
    return


if __name__ == "__main__":
    app.run()
