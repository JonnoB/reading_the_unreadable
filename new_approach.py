import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # What is this about

        Now I am using Doclayout yolo I am re-writing everything. 

        I believe it is key to infer the underlying structure of the page, as this can be used to work out what needs to be changed

        I need to be able to create the reading order as quickly as possible as this will help me work out what goes with what.


            I have a dataframe which contains the x1,x2,y1,y2, coordinates, the center_x, center_y of bounding boxes representing the elements of a printed newspaper. I also have column_number, which says to which column a box belongs. I want to create the reading order going from the lowest column to the highest. For each column the reading order is the center_y ordered by size. The first box of a column follows the last box of the previous column. Could you construct a function that finds the reading order please. also needs to take into account the page_id, as the bounding boxes for multiple pages are in the same dataframe
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import os
    from bbox_functions import (print_area_meta, reclassify_abandon_boxes, assign_columns, 
    create_reading_order)#, merge_overlapping_boxes)
    import numpy as np
    from tqdm import tqdm
    return (
        assign_columns,
        create_reading_order,
        np,
        os,
        pd,
        print_area_meta,
        reclassify_abandon_boxes,
        tqdm,
    )


@app.cell
def __():
    return


@app.cell
def __(pd):
    def merge_boxes_within_column_width(df, width_multiplier=1.5):
        """
        Merge bounding boxes that are not 'figure' or 'table' if the merged height doesn't exceed
        x times the column_width, maintaining reading order and working with multiple pages.

        Parameters:
        df: DataFrame with columns x1,x2,y1,y2,center_x,center_y,column_number,column_width,
            reading_order, page_id, and type
        width_multiplier: maximum allowed height as a multiple of column_width

        Returns:
        DataFrame with merged boxes
        """
        df = df.copy()
        df = df.sort_values(['page_id', 'reading_order'])

        merged_boxes = []

        for page_id in df['page_id'].unique():
            page_df = df[df['page_id'] == page_id]

            for col_num in page_df[page_df['column_number'] > 0]['column_number'].unique():
                col_df = page_df[page_df['column_number'] == col_num]

                current_box = None

                for _, row in col_df.iterrows():
                    if row['class'] in ['figure', 'table']:
                        if current_box is not None:
                            merged_boxes.append(current_box)
                            current_box = None
                        merged_boxes.append(row.to_dict())
                        continue

                    if current_box is None:
                        current_box = row.to_dict()
                    else:
                        # Calculate potential merged box height
                        merged_height = max(row['y2'], current_box['y2']) - min(row['y1'], current_box['y1'])

                        # Check if merge would exceed height limit
                        if merged_height <= row['column_width'] * width_multiplier:
                            # Merge boxes
                            current_box['x1'] = min(current_box['x1'], row['x1'])
                            current_box['x2'] = max(current_box['x2'], row['x2'])
                            current_box['y1'] = min(current_box['y1'], row['y1'])
                            current_box['y2'] = max(current_box['y2'], row['y2'])
                            current_box['center_x'] = (current_box['x1'] + current_box['x2']) / 2
                            current_box['center_y'] = (current_box['y1'] + current_box['y2']) / 2
                        else:
                            # Add current box to results and start new one
                            merged_boxes.append(current_box)
                            current_box = row.to_dict()

                # Add last box if exists
                if current_box is not None:
                    merged_boxes.append(current_box)

        result_df = pd.DataFrame(merged_boxes)
        result_df = result_df.sort_values(['page_id', 'reading_order'])

        return result_df

    def adjust_y2_coordinates(df):
        """
        Adjusts y2 coordinates for boxes within each column of each block:
        - Sets y2 to the y1 of the next box in reading order
        - Removes overlaps between boxes
        - Keeps the last box in each column unchanged
        """
        # Create a copy of the dataframe
        df_adjusted = df.copy()
        
        # Sort within groups and shift y1 values
        df_adjusted['y2'] = (df_adjusted
            .sort_values(['page_id', 'page_block', 'column_number', 'y1'])
            .groupby(['page_id', 'page_block', 'column_number'])['y1']
            .shift(-1)
            .fillna(df_adjusted['y2']))  # Keep original y2 for last box in each group

        return df_adjusted

    def adjust_x_coordinates(df):
        """
        Adjusts x coordinates based on column boundaries (c1 and c2):
        - If x1 > c1, sets x1 to c1
        - If x2 < c2, sets x2 to c2
        This extends boxes that are too narrow but preserves wider boxes.
        """
        df_adjusted = df.copy()

        # Ensure c1 and c2 are float64
        df_adjusted['c1'] = df_adjusted['c1'].astype('float64')
        df_adjusted['c2'] = df_adjusted['c2'].astype('float64')

        # Adjust x1 where it's greater than c1
        mask_x1 = df_adjusted['x1'] > df_adjusted['c1']
        df_adjusted.loc[mask_x1, 'x1'] = df_adjusted.loc[mask_x1, 'c1'].astype('float64')

        # Adjust x2 where it's less than c2
        mask_x2 = df_adjusted['x2'] < df_adjusted['c2']
        df_adjusted.loc[mask_x2, 'x2'] = df_adjusted.loc[mask_x2, 'c2'].astype('float64')

        return df_adjusted
    return (
        adjust_x_coordinates,
        adjust_y2_coordinates,
        merge_boxes_within_column_width,
    )


@app.cell
def __(
    adjust_x_coordinates,
    adjust_y2_coordinates,
    assign_columns,
    create_reading_order,
    merge_boxes_within_column_width,
    np,
    pd,
    print_area_meta,
    reclassify_abandon_boxes,
):
    image_size = 1024

    bbox_all_df = pd.read_parquet(f'data/bbox_results_{image_size}.parquet')

    # THIS IS TEMPORARY UNTIL I DECIDE A PAGE_id convention
    bbox_all_df['page_id'] = bbox_all_df['filename']

    bbox_all_df['width'] = bbox_all_df['x2'] - bbox_all_df['x1']
    bbox_all_df['height'] = bbox_all_df['y2'] - bbox_all_df['y1']
    bbox_all_df['center_x'] = bbox_all_df['width'] + bbox_all_df['x1']
    bbox_all_df['center_y'] = bbox_all_df['height'] + bbox_all_df['y1']

    # Calculate the column width to get the total number of columns on the page
    bbox_all_df_text = bbox_all_df.loc[bbox_all_df['class'].isin(['plain text']), ['page_id', 'width']].copy()
    bbox_all_df_text = bbox_all_df_text.groupby('page_id')['width'].median().rename('median_box_width')
    bbox_all_df = bbox_all_df.join(bbox_all_df_text, on = 'page_id')

    bbox_all_df = print_area_meta(bbox_all_df)
    bbox_all_df['column_counts'] =  np.floor(bbox_all_df['print_width']/bbox_all_df['median_box_width'])
    bbox_all_df['column_width'] = bbox_all_df['print_width']/bbox_all_df['column_counts']

    bbox_all_df = reclassify_abandon_boxes(bbox_all_df, top_fraction=0.1)

    bbox_df = bbox_all_df.loc[bbox_all_df['class']!='abandon'].copy()
    bbox_df = bbox_all_df
    # After re-classifying boxes as abandon and dropping the abandon boxes re-calculate the print area which should have changed
    bbox_df = print_area_meta(bbox_df)

    bbox_df = assign_columns(bbox_df)

    #Temporary to merge overlapping boxes
    bbox_df= create_reading_order(bbox_df)

    # change class when there is more than one column
    bbox_df['class'] = np.where((bbox_df['column_counts']>1) & 
                                (bbox_df['column_number']!=0) &
                                (~bbox_df['class'].isin(['figure', 'table'])),  # Close the condition parenthesis here
                                'plain text',  # Value if condition is True
                                bbox_df['class'])  # Value if condition is False

    #Change class when there is only 1 column
    bbox_df['class'] = np.where((bbox_df['column_counts']==1) & 
                                (bbox_df['column_number']!=0) &
                                (~bbox_df['class'].isin(['figure', 'table', 'title'])),  # Close the condition parenthesis here
                                'plain text',  # Value if condition is True
                                bbox_df['class'])  # Value if condition is False

    #bbox_df = bbox_df.loc[bbox_df['filename']=='NS2_pageid_163094_pagenum_4_1843-04-01.jpg']

    bbox_df = adjust_y2_coordinates(bbox_df)
    bbox_df = adjust_x_coordinates(bbox_df)

    bbox_df = merge_boxes_within_column_width(bbox_df)


    bbox_df= create_reading_order(bbox_df)
    return bbox_all_df, bbox_all_df_text, bbox_df, image_size


@app.cell
def __(bbox_df):
    bbox_df.loc[bbox_df['filename']=='TTW_pageid_160259_pagenum_12_1867-12-21.jpg']
    return


@app.cell
def __(bbox_df):
    bbox_df['class'].unique()
    return


@app.cell
def __(np, os, tqdm):
    import cv2
    import matplotlib.pyplot as plt


    def plot_boxes_on_image(df, image_path, figsize=(15,15), show_reading_order=False):
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Display the image
        ax.imshow(image)

        # Define fixed colors for specific classes
        fixed_colors = {
            'plain text': '#FF0000',  # Red
            'title': '#00FF00',       # Green
            'figure': '#0000FF'       # Blue
        }

        # Create a color map for any other classes
        unique_classes = [cls for cls in df['class'].unique() if cls not in fixed_colors]
        if unique_classes:  # If there are other classes
            additional_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
            additional_color_dict = dict(zip(unique_classes, additional_colors))
        else:
            additional_color_dict = {}

        # Combine fixed and additional colors
        class_color_dict = {**fixed_colors, **additional_color_dict}

        # Plot each bounding box
        for idx, row in df.iterrows():
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            box_class = row['class']

            # Create rectangle patch
            width = x2 - x1
            height = y2 - y1

            # Get color for this class
            color = class_color_dict[box_class]

            # Draw rectangle
            rect = plt.Rectangle((x1, y1), width, height,
                               fill=False,
                               color=color,
                               linewidth=2)
            ax.add_patch(rect)

            # Optionally add class labels
            ax.text(x1, y1-5, box_class, 
                    color=color,
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))

        # Add reading order arrows if requested
        if show_reading_order and 'reading_order' in df.columns:
            # Sort by reading order
            df_sorted = df.sort_values('reading_order')

            # Calculate center points
            df_sorted['center_x'] = (df_sorted['x1'] + df_sorted['x2']) / 2
            df_sorted['center_y'] = (df_sorted['y1'] + df_sorted['y2']) / 2

            # Draw arrows between consecutive boxes
            for i in range(len(df_sorted)-1):
                start = (df_sorted.iloc[i]['center_x'], df_sorted.iloc[i]['center_y'])
                end = (df_sorted.iloc[i+1]['center_x'], df_sorted.iloc[i+1]['center_y'])

                ax.annotate('',
                           xy=end,
                           xytext=start,
                           arrowprops=dict(arrowstyle='->',
                                         color='black',
                                         linewidth=2,
                                         mutation_scale=20),  # Makes the arrow head bigger
                           )

        # Remove axes
        ax.set_axis_off()

        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, 
                                       label=class_name)
                          for class_name, color in class_color_dict.items()]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        return fig  # Return the figure instead of showing it

    # And then the loop function would be:
    def save_plots_for_all_files(df, image_dir, output_dir, figsize=(15,15)):
        os.makedirs(output_dir, exist_ok=True)
        unique_images = df['filename'].unique()

        for filename in tqdm(unique_images, desc="Processing images"):
            image_df = df[df['filename'] == filename]
            image_path = os.path.join(image_dir, filename)

            # Get the figure from plot_boxes_on_image
            fig = plot_boxes_on_image(image_df, image_path, figsize=figsize)

            # Save the figure
            output_path = os.path.join(output_dir, f'annotated_{filename}')
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)  # Close the figure to free memory
    return cv2, plot_boxes_on_image, plt, save_plots_for_all_files


@app.cell
def __(bbox_df, os, plot_boxes_on_image):
    file_name = 'MRP_pageid_123314_pagenum_54_1807-05-02.jpg'
    file_name = 'TTW_pageid_160259_pagenum_12_1867-12-21.jpg'
    print_box = bbox_df.loc[bbox_df['reading_order']==2]


    plot_boxes_on_image(print_box.loc[print_box['filename']==file_name], 
                        image_path = os.path.join('data/ncse_test_jpg', file_name), show_reading_order=True)
    return file_name, print_box


@app.cell
def __():
    #save_plots_for_all_files(bbox_df, image_dir = 'data/ncse_test_jpg', output_dir = f'data/custom_bounding_{image_size}')
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
