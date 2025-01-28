import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(__file__):
    import os
    import pandas as pd
    from function_modules.bbox_functions import calculate_coverage_and_overlap, plot_boxes_on_image, postprocess_bbox
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np

    # Change working directory to project root
    os.chdir(Path(__file__).parent)

    bbox_folder = 'data/periodical_bboxes/post_process'
    bbox_folder_raw = 'data/periodical_bboxes/raw'
    bbox_df = pd.concat([pd.read_parquet(os.path.join(bbox_folder, file)) for file in os.listdir(bbox_folder)])
    raw_bbox_df = pd.concat([pd.read_parquet(os.path.join(bbox_folder_raw, file)) for file in os.listdir(bbox_folder_raw)])
    raw_bbox_df['page_id'] = raw_bbox_df['filename']

    periodical_mapping = pd.DataFrame({'periodical':['TEC', 'FTEC', 'TTW', 'ATTW', 'ETTW', 'FTTW', 'EWJ', 'FEWJ',
           'EMRP', 'FMRP', 'MRP', 'SMRP', 'CLD', 'FLDR', 'LDR', 'NSS', 'NS2',
           'NS8', 'NS4', 'NS3', 'NS5', 'NS7', 'NS6', 'NS9', 'SNSS'], 
    'periodical_code':['TEC', 'TEC', 'TTW','TTW','TTW','TTW', 'EWJ', 'EWJ','MRP', 'MRP','MRP','MRP', 'CLD', 'CLD','CLD', 'NS', 'NS','NS','NS','NS','NS','NS','NS','NS','NS', ]}
    )

    image_folder = os.environ['image_folder']

    path_mapping = {
        'CLD': os.path.join(image_folder, 'converted/all_files_png_120/Leader_issue_PDF_files'),
        'EWJ': os.path.join(image_folder, 'converted/all_files_png_120/English_Womans_Journal_issue_PDF_files'),
        'MRP': os.path.join(image_folder, 'converted/all_files_png_120/Monthly_Repository_issue_PDF_files'),
        'TTW': os.path.join(image_folder, 'converted/all_files_png_120/Tomahawk_issue_PDF_files'),
        'TEC': os.path.join(image_folder, 'converted/all_files_png_120/Publishers_Circular_issue_PDF_files'),
        'NS': os.path.join(image_folder, 'converted/all_files_png_200/Northern_Star_issue_PDF_files')
    }
    return (
        Path,
        bbox_df,
        bbox_folder,
        bbox_folder_raw,
        calculate_coverage_and_overlap,
        image_folder,
        np,
        os,
        path_mapping,
        pd,
        periodical_mapping,
        plot_boxes_on_image,
        plt,
        postprocess_bbox,
        raw_bbox_df,
    )


@app.cell
def _(bbox_df, calculate_coverage_and_overlap):
    few_bbox = bbox_df.loc[bbox_df['filename'].isin(bbox_df['filename'].unique()[0:20])]


    post_process_coverage_df = calculate_coverage_and_overlap(few_bbox)


    # post_process_coverage_df = calculate_coverage_and_overlap(bbox_df)
    return few_bbox, post_process_coverage_df


@app.cell
def _():
    return


@app.cell
def _(post_process_coverage_df):
    post_process_coverage_df
    return


@app.cell
def _(
    bbox_df,
    calculate_coverage_and_overlap,
    os,
    path_mapping,
    periodical_mapping,
    plot_boxes_on_image,
    plt,
):
    target_page ='NS2-1838-01-13_page_4'#  'NS2-1838-01-20_page_1' #


    _temp_bbox = bbox_df.loc[(bbox_df['page_id']== target_page) & (bbox_df['class']!='tablex'), :]

    periodical_code = periodical_mapping.loc[periodical_mapping['periodical']==target_page.split("-")[0], 'periodical_code'].iloc[0]

    plot_boxes_on_image(_temp_bbox, os.path.join(path_mapping[periodical_code],target_page+'.png' ))
    plt.show()
    calculate_coverage_and_overlap(_temp_bbox)
    return periodical_code, target_page


@app.cell
def _(mo):
    mo.md(
        """
        I am trying to parse the layout of archival newspapers. I have used an object detector to identify and classify the areas of text. However, the object detector has only worked relatively well requireing substantial post-processing. Currently my post processing is very good, however it fails in one very clear element. The approach sucessfully  identifies the columns of text and extends text boxes down to the subsequent box. However, in the case that the top box in the column is missing, or the bottom box, no additional box is added. 
        I want to rectify this.

        My data has a reading_order giving the order of the bounding boxes column_number, c1, and c2 which are the x1, x2 of the column, 'class' the class of the bounding box, it also has a column called 'page_block' as some pages have complex layouts, finally it has a "page_id" which uniquely identifies the page

        For pages with more than one column but only 1 block I want to find the minimum y1 and maximum y2 for all columns (these will be called minimum_y1, and minimum_y2), I then want to add a new bounding box to each column whose y1 is the minimum_y1 and whose y2 is the y1 of the bounding box with the lowest reading_order in that column, then a bounding box will be added whose y1 is the y2 of the bounding box with the highest reading order in that column, the y2 of the new bounding box will be the maxmimum_y2.

        This process will be performed for all columns of value 1 or higher.


        Does this makes sense, what are your thoughts?
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        from function_modules.bbox_functions import basic_box_data, reclassify_abandon_boxes, print_area_meta, assign_columns, create_reading_order, adjust_y2_coordinates, adjust_x_coordinates, merge_boxes_within_column_width
        import numpy as np


        def fill_column_gaps(bbox_df):
            \"""
            Add bounding boxes at the top and bottom of columns where they're missing.
            Only processes pages with multiple columns and a single block.

            Args:
                bbox_df (pd.DataFrame): DataFrame containing bounding box information

            Returns:
                pd.DataFrame: DataFrame with additional bounding boxes added
            \"""
            # Create mask for eligible pages
            page_col_counts = bbox_df.groupby('page_id')['column_number'].transform('max')
            page_block_counts = bbox_df.groupby('page_id')['page_block'].transform('nunique')
            eligible_mask = (page_col_counts > 1) & (page_block_counts == 1)

            # Split into eligible and ineligible dataframes
            eligible_df = bbox_df[eligible_mask].copy()
            ineligible_df = bbox_df[~eligible_mask].copy()

            new_boxes = []

            # Process only if there are eligible pages
            if not eligible_df.empty:
                for page_id in eligible_df['page_id'].unique():
                    page_data = eligible_df[eligible_df['page_id'] == page_id]

                    # Find global minimum y1 and maximum y2 for the page
                    minimum_y1 = page_data['y1'].min()
                    maximum_y2 = page_data['y2'].max()

                    # Get filename and issue for this page
                    filename = page_data['filename'].iloc[0]
                    issue = page_data['issue'].iloc[0]

                    # Process each column (excluding column 0)
                    for col_num in page_data[page_data['column_number'] > 0]['column_number'].unique():
                        col_data = page_data[page_data['column_number'] == col_num]

                        if len(col_data) == 0:
                            continue

                        # Get column boundaries
                        c1 = col_data['c1'].iloc[0]
                        c2 = col_data['c2'].iloc[0]

                        # Find top and bottom boxes in the column
                        top_box = col_data.loc[col_data['reading_order'].idxmin()]
                        bottom_box = col_data.loc[col_data['reading_order'].idxmax()]

                        # Add top box if needed
                        if top_box['y1'] > minimum_y1:
                            new_boxes.append({
                                'page_id': page_id,
                                'filename': filename,
                                'issue': issue,
                                'class': 'text',
                                'y1': minimum_y1,
                                'y2': top_box['y1'],
                                'x1': top_box['x1'],
                                'x2': top_box['x2'],
                                'c1': c1,
                                'c2': c2,
                                'column_number': col_num,
                                'page_block': top_box['page_block'],
                                'reading_order': -1  # Will be recalculated later
                            })

                        # Add bottom box if needed
                        if bottom_box['y2'] < maximum_y2:
                            new_boxes.append({
                                'page_id': page_id,
                                'filename': filename,
                                'issue': issue,
                                'class': 'text',
                                'y1': bottom_box['y2'],
                                'y2': maximum_y2,
                                'x1': bottom_box['x1'],
                                'x2': bottom_box['x2'],
                                'c1': c1,
                                'c2': c2,
                                'column_number': col_num,
                                'page_block': bottom_box['page_block'],
                                'reading_order': -1  # Will be recalculated later
                            })

            # Add new boxes to the eligible DataFrame if any were created
            if new_boxes:
                new_boxes_df = pd.DataFrame(new_boxes)
                new_boxes_df = basic_box_data(new_boxes_df)
                new_boxes_df = print_area_meta(new_boxes_df)
                eligible_df = pd.concat([eligible_df, new_boxes_df], ignore_index=True)

            # Combine eligible and ineligible dataframes
            result_df = pd.concat([eligible_df, ineligible_df], ignore_index=True)

            return result_df


        def postprocess_bbox(df, min_height = 10, width_multiplier = 1.5, remove_abandon = True, fill_columns = True):

            \""" 
            This function performs all the post-processing on the bounding boxes to clean them up after being produced by DOCLAyout-Yolo
            and to make them ready for sending image crops to the image-to-text model. 
            The function assumes that the bounding boxes have been made by DocLayout-yolo. Other Yolo models may also work but this has not been tested.
            The function is a wrapper for functions from the bbox_functions module

            Files require a page_id column which uniquely identifies each page
            \"""

            bbox_all_df = df.copy()

            bbox_all_df['issue'] = bbox_all_df['filename'].str.split('_page_').str[0]

            bbox_all_df = basic_box_data(bbox_all_df)

            bbox_all_df = reclassify_abandon_boxes(bbox_all_df, top_fraction=0.1)

            bbox_df = bbox_all_df.copy()

            if remove_abandon:
                bbox_df = bbox_all_df.loc[bbox_all_df['class']!='abandon'].copy()

            # After re-classifying boxes as abandon and dropping the abandon boxes re-calculate the print area which should have changed
            bbox_df = print_area_meta(bbox_df)

            bbox_df = assign_columns(bbox_df)


            \""" 
            This logic has been changed as it looks like keeping titles will make splitting up texts a lot easier and 
            that titles are pretty reliable if you do a little post processing.
            # change class when there is more than one column
            bbox_df['class'] = np.where((bbox_df['column_counts']>1) & 
                                        (bbox_df['column_number']!=0) &
                                       (~bbox_df['class'].isin(['figure', 'table'])),
                                        'text',  # Value if condition is True
                                        bbox_df['class'])  # Value if condition is False

            #Change class when there is only 1 column
            bbox_df['class'] = np.where((bbox_df['column_counts']==1) & 
                                        (bbox_df['column_number']!=0) &
                                        (~bbox_df['class'].isin(['figure', 'table', 'title'])), 
                                        'text',  # Value if condition is True
                                        bbox_df['class'])  # Value if condition is False
            \"""
            # This re-labels everything that is not abandon, text, table, or figure as title.
            bbox_df['class'] = np.where((~bbox_df['class'].isin(['figure', 'table', 'text', 'abandon'])), 
                                        'title',  # Value if condition is True
                                        bbox_df['class'])  # Value if condition is False

            bbox_df= create_reading_order(bbox_df)
            #Remove overlaps by adjusting the y2 coordinate to the subsequent bounding box
            # This causes boxes that overlap another box to bring there lower bound up to the top of the subsequent box
            # This can make large boxes much smaller, and results in overlaps being removed
            bbox_df = adjust_y2_coordinates(bbox_df)
            #adjust the x limits to the column width if box is narrower than the column
            bbox_df = adjust_x_coordinates(bbox_df)
            #Should the columns be filled?
            if fill_columns:
                bbox_df = fill_column_gaps(bbox_df)

            # Filter out boxes that are too small after y2 adjustment
            bbox_df['height'] = bbox_df['y2'] - bbox_df['y1']  
            bbox_df = bbox_df[bbox_df['height'] >= min_height]

            #remove small bounding boxes to make sending to LLM more efficient and results in larger text blocks
            if width_multiplier is not None:
                bbox_df = merge_boxes_within_column_width(bbox_df, width_multiplier=width_multiplier)

            # Adjust y2 again as the box deletion and merging has changed boundaries
            bbox_df = adjust_y2_coordinates(bbox_df)

            #due to merging and deletion re-do reading order
            bbox_df = create_reading_order(bbox_df)

            #add in bbox ID
            bbox_df['box_page_id'] = "B" + bbox_df['page_block'].astype(str) + "C"+bbox_df['column_number'].astype(str)  + "R" + bbox_df['reading_order'].astype(str) 
            bbox_df['ratio'] = bbox_df['height']/bbox_df['width']  #box ratio

            return bbox_df
        """
    )
    return


@app.cell
def _(
    calculate_coverage_and_overlap,
    np,
    os,
    path_mapping,
    periodical_code,
    plot_boxes_on_image,
    plt,
    postprocess_bbox,
    raw_bbox_df,
    target_page,
):
    _raw_temp = raw_bbox_df.loc[(raw_bbox_df['page_id']== (target_page+'.png')) & (raw_bbox_df['class']!='tablex'), :]
    _raw_temp['class'] = np.where(_raw_temp['class']=='plain text', 'text', _raw_temp['class'])

    col_filled_df = postprocess_bbox(_raw_temp, fill_columns = True)


    _temp_bbox = col_filled_df

    plot_boxes_on_image(_temp_bbox, os.path.join(path_mapping[periodical_code],target_page+'.png' ))
    plt.show()
    calculate_coverage_and_overlap(_temp_bbox)
    return (col_filled_df,)


@app.cell
def _(
    bbox_df,
    calculate_coverage_and_overlap,
    np,
    postprocess_bbox,
    raw_bbox_df,
):
    _raw_temp = raw_bbox_df.loc[raw_bbox_df['filename'].isin(bbox_df['filename'].unique()[0:20])]
    _raw_temp['class'] = np.where(_raw_temp['class']=='plain text', 'text', _raw_temp['class'])


    sample_df = postprocess_bbox(_raw_temp, fill_columns = True)
    sample_df['box_area'] = sample_df['width'] * sample_df['height']
    sample_df['fract_print_area'] = sample_df['box_area']/sample_df['print_area']

    sample_no_proc_overlap = calculate_coverage_and_overlap(sample_df)
    return sample_df, sample_no_proc_overlap


@app.cell
def _(sample_df):
    sample_df
    return


@app.cell
def _(sample_no_proc_overlap):
    sample_no_proc_overlap
    return


@app.cell
def _(post_process_coverage_df):
    post_process_coverage_df
    return


@app.cell
def _(os, pd):


    save_folder = "data/overlap_coverage"
    source_folders = ["post_process", 'post_process_fill', 'post_process_raw']

    # List to store individual dataframes
    dfs = []

    for method in source_folders:
        method_folder = os.path.join(save_folder, method)
        
        print(f"Loading files from method: {method}")
        
        # Process each file in the method folder
        for file in os.listdir(method_folder):
            
            # Load the parquet file
            df = pd.read_parquet(os.path.join(method_folder, file))
            
            # Add method column
            df['method'] = method
            df['periodical'] = file
            
            dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)


    return (
        combined_df,
        df,
        dfs,
        file,
        method,
        method_folder,
        save_folder,
        source_folders,
    )


@app.cell
def _(combined_df):
    combined_df
    return


@app.cell
def _(combined_df):
    import seaborn as sns

    sns.displot(data = combined_df, x = 'perc_print_area_overlap', hue = 'method', kind='hist',
               col = 'periodical', col_wrap=3)
    return (sns,)


@app.cell
def _(combined_df, sns):
    sns.displot(data = combined_df, x = 'perc_print_area_coverage', hue = 'method', kind='kde',
               col = 'periodical', col_wrap=3)
    return


@app.cell
def _(combined_df):
    combined_df.groupby([ 'periodical', 'method'])['perc_print_area_overlap'].agg(['mean', 'median'])
    return


@app.cell
def _(combined_df):
    combined_df.groupby([ 'periodical', 'method'])['perc_print_area_coverage'].agg(['mean', 'median'])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
