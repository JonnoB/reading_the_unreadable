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
    create_reading_order,
    merge_boxes_within_column_width, adjust_y2_coordinates, 
    adjust_x_coordinates,
    plot_boxes_on_image, save_plots_for_all_files)#, merge_overlapping_boxes)
    import numpy as np
    from tqdm import tqdm
    return (
        adjust_x_coordinates,
        adjust_y2_coordinates,
        assign_columns,
        create_reading_order,
        merge_boxes_within_column_width,
        np,
        os,
        pd,
        plot_boxes_on_image,
        print_area_meta,
        reclassify_abandon_boxes,
        save_plots_for_all_files,
        tqdm,
    )


@app.cell
def __():
    def adjust_y2_coordinates2(df):
        """
        Adjusts y2 coordinates for boxes within each column of each block:
        - Sets y2 to the y1 of the next box in reading order
        - Removes overlaps between boxes
        - Keeps the last box in each column unchanged
        """
        # Create a copy of the dataframe
        df_adjusted = df.copy()

        # Sort within groups and shift y1 values
        df_adjusted['y22'] = (df_adjusted
            .sort_values(['page_id', 'page_block', 'column_number', 'y1'])
            .groupby(['page_id', 'page_block', 'column_number'])['y1']
            .shift(-1)
            .fillna(df_adjusted['y2']))  # Keep original y2 for last box in each group

        return df_adjusted
    return (adjust_y2_coordinates2,)


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

    # After re-classifying boxes as abandon and dropping the abandon boxes re-calculate the print area which should have changed
    bbox_df = print_area_meta(bbox_df)

    bbox_df = assign_columns(bbox_df)

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

    #Temporary to merge overlapping boxes
    bbox_df= create_reading_order(bbox_df)


    #bbox_df = bbox_df.loc[bbox_df['filename']=='NS2_pageid_163094_pagenum_4_1843-04-01.jpg']



    before_adjust = bbox_df.copy()
    bbox_df = adjust_y2_coordinates(bbox_df)
    temp = bbox_df.copy()

    bbox_df = adjust_x_coordinates(bbox_df)

    bbox_df = merge_boxes_within_column_width(bbox_df)


    bbox_df= create_reading_order(bbox_df)

    bbox_df['box_page_id'] = "B" + bbox_df['page_block'].astype(str) + "C"+bbox_df['column_number'].astype(str)  + "R" + bbox_df['reading_order'].astype(str) 
    return (
        bbox_all_df,
        bbox_all_df_text,
        bbox_df,
        before_adjust,
        image_size,
        temp,
    )


@app.cell
def __(bbox_df):
    bbox_df
    return


@app.cell
def __():
    return


@app.cell
def __(bbox_df):
    # Save the image bounding boxes

    bbox_df.to_csv('data/ncse_testset_bounding.csv')
    return


@app.cell
def __(temp):
    print(temp.loc[temp['filename']=='MRP_pageid_123497_pagenum_13_1807-10-02.jpg', ['page_block', 'column_number' ,'y1', 'y2' ,'x1', 'x2', 'c1', 'c2']])
    return


@app.cell
def __():
    return


@app.cell
def __(bbox_df, os, plot_boxes_on_image):
    file_name = 'MRP_pageid_123497_pagenum_13_1807-10-02.jpg'
    #file_name = 'TTW_pageid_160259_pagenum_12_1867-12-21.jpg'
    print_box = bbox_df


    plot_boxes_on_image(print_box.loc[print_box['filename']==file_name], 
                        image_path = os.path.join('data/ncse_test_jpg', file_name), show_reading_order=True)
    return file_name, print_box


@app.cell
def __():

    """
    save_plots_for_all_files(bbox_df, 
                             image_dir = 'data/ncse_test_jpg', 
                             output_dir = f'data/custom_bounding_{image_size}', 
                             show_reading_order=True)

    """
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
