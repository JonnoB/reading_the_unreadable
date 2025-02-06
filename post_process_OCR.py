import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    from function_modules.bbox_functions import plot_boxes_on_image
    from function_modules.analysis_functions import is_title, split_and_reclassify, merge_consecutive_titles, remove_line_breaks
    import os 
    from tqdm import tqdm
    import numpy as np 
    import time
    import seaborn as sns
    import matplotlib.pyplot as plt
    import cv2

    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    from typing import List

    image_path = os.environ['image_folder']

    data = os.path.join('data', "download_jobs/EWJ.parquet")

    data_path = os.path.join('data', "download_jobs", "ncse")

    all_data_files = [file for file in os.listdir(data_path) if '.parquet' in file ]

    periodical_mapping = pd.DataFrame({'periodical':['TEC', 'FTEC', 'TTW', 'ATTW', 'ETTW', 'FTTW', 'EWJ', 'FEWJ',
           'EMRP', 'FMRP', 'MRP', 'SMRP', 'CLD', 'FLDR', 'LDR', 'NSS', 'NS2',
           'NS8', 'NS4', 'NS3', 'NS5', 'NS7', 'NS6', 'NS9', 'SNSS'], 
    'periodical_code':['TEC', 'TEC', 'TTW','TTW','TTW','TTW', 'EWJ', 'EWJ','MRP', 'MRP','MRP','MRP', 'CLD', 'CLD','CLD', 'NS', 'NS','NS','NS','NS','NS','NS','NS','NS','NS', ]}
    )

    path_mapping = {
        'CLD': os.path.join(image_path, 'converted/all_files_png_120/Leader_issue_PDF_files'),
        'EWJ': os.path.join(image_path, 'converted/all_files_png_120/English_Womans_Journal_issue_PDF_files'),
        'MRP': os.path.join(image_path, 'all_files_png_120/Monthly_Repository_issue_PDF_files'),
        'TTW': os.path.join(image_path, 'converted/all_files_png_120/Tomahawk_issue_PDF_files'),
        'TEC': os.path.join(image_path, 'converted/all_files_png_120/Publishers_Circular_issue_PDF_files'),
        'NS': os.path.join(image_path, 'converted/all_files_png_200/Northern_Star_issue_PDF_files')
    }

    raw_bbox_path = "data/periodical_bboxes/raw"
    bbox_path = "data/periodical_bboxes/post_process"

    raw_bboxes_df = [pd.read_parquet(os.path.join(raw_bbox_path, _file_path)) for _file_path in os.listdir(raw_bbox_path)] 
    raw_bboxes_df = pd.concat(raw_bboxes_df, ignore_index=True)

    bboxes_df = [pd.read_parquet(os.path.join(bbox_path, _file_path)) for _file_path in os.listdir(bbox_path)] 
    bboxes_df = pd.concat(bboxes_df, ignore_index=True)
    return (
        List,
        Path,
        ProcessPoolExecutor,
        all_data_files,
        as_completed,
        bbox_path,
        bboxes_df,
        cv2,
        data,
        data_path,
        image_path,
        is_title,
        merge_consecutive_titles,
        np,
        os,
        path_mapping,
        pd,
        periodical_mapping,
        plot_boxes_on_image,
        plt,
        raw_bbox_path,
        raw_bboxes_df,
        remove_line_breaks,
        sns,
        split_and_reclassify,
        time,
        tqdm,
    )


@app.cell
def _(all_data_files, data_path, os, pd):
    test2 = []

    for _file in all_data_files:

        test2.append(pd.read_parquet(os.path.join(data_path, _file)))

    test2 = pd.concat(test2, ignore_index=True)
    test2['filename'] = test2['page_id']+'.png'
    return (test2,)


@app.cell
def _(test2):
    test2
    return


@app.cell
def _(test2):
    test2
    return


@app.cell
def _(
    bboxes_df,
    cv2,
    np,
    os,
    path_mapping,
    periodical_mapping,
    plt,
    raw_bboxes_df,
    test2,
):
    def plot_boxes_on_image2(df, image_path, figsize=(15,15), show_reading_order=False, fill_boxes=True, fill_alpha=0.2):
        """
        Plot bounding boxes and their classifications on a page image with optional reading order visualization.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing bounding box coordinates and classifications.
            Required columns: 'x1', 'y1', 'x2', 'y2', 'class'
            Optional column: 'reading_order' (if show_reading_order=True)

        image_path : str
            Path to the input image file.

        figsize : tuple, optional (default=(15,15))
            Figure size in inches (width, height).

        show_reading_order : bool, optional (default=False)
            If True, draws arrows between boxes indicating reading order sequence.
            Requires 'reading_order' column in the DataFrame.

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the plotted image with bounding boxes.

        Notes:
        ------
        - The function uses predefined colors for specific classes:
            * 'text': Red
            * 'title': Green
            * 'figure': Blue
        - Additional classes are assigned colors automatically using a rainbow colormap.
        - Each box is labeled with its class name.
        - A legend is included showing all class types and their corresponding colors.
        """
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Display the image
        ax.imshow(image)

        # Define fixed colors for specific classes
        fixed_colors = {
            'text': '#FF0000',  # Red
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
                               fill=fill_boxes,  # Use the fill_boxes parameter
                               facecolor=color if fill_boxes else 'none',  # Set fill color if filling
                               alpha=fill_alpha if fill_boxes else 1,  # Set alpha if filling
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
        return fig 



    def trim_white_space(image):
        # Assuming white is [255, 255, 255]
        mask = (image != 255).any(axis=2)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return image[ymin:ymax, xmin:xmax]

    def pad_to_match_height(image1, image2):
        max_height = max(image1.shape[0], image2.shape[0])

        # Pad image1 if necessary
        if image1.shape[0] < max_height:
            pad_height = max_height - image1.shape[0]
            padding = np.full((pad_height, image1.shape[1], image1.shape[2]), 255, dtype=np.uint8)
            image1 = np.vstack((image1, padding))

        # Pad image2 if necessary
        if image2.shape[0] < max_height:
            pad_height = max_height - image2.shape[0]
            padding = np.full((pad_height, image2.shape[1], image2.shape[2]), 255, dtype=np.uint8)
            image2 = np.vstack((image2, padding))

        return image1, image2

    combined_image_folder = 'data/combined_images'

    os.makedirs(combined_image_folder,exist_ok=True)

    for _filename in test2['filename'].unique():

            _periodical_code = periodical_mapping.loc[periodical_mapping['periodical']==_filename.split("-")[0], 
        'periodical_code'].iloc[0]
            #  Identify image folder using the periodical ID
            _image_path = path_mapping[_periodical_code]

            # Create first plot and convert to image
            fig1 = plot_boxes_on_image2(bboxes_df[bboxes_df['filename']==_filename], 
                                      image_path = os.path.join(_image_path, _filename), 
                                      show_reading_order=True)
            # Convert figure to numpy array
            fig1.canvas.draw()
            image1 = np.array(fig1.canvas.renderer._renderer)

            # rename classes so the images are easier to compare
            _temp = raw_bboxes_df[raw_bboxes_df['filename']==_filename].copy()
            _temp['class'] = np.where(_temp['class']=='plain text', 'text', _temp['class'])
            _temp['class'] = np.where(_temp['class'].isin(['text', 'figure', 'table']), _temp['class'], 'other')
            fig2 = plot_boxes_on_image2(_temp, 
                                      image_path = os.path.join(_image_path, _filename), 
                                      show_reading_order=True)
            fig2.canvas.draw()
            image2 = np.array(fig2.canvas.renderer._renderer)

            # Combine images
            _image1_trimmed = trim_white_space(image1)
            _image2_trimmed = trim_white_space(image2)
            _image1_padded, _image2_padded = pad_to_match_height(_image1_trimmed, _image2_trimmed)
            combined_image = np.hstack((_image2_padded, _image1_padded))

            # Display combined image
            plt.figure(figsize=(30,15))
            plt.imshow(combined_image)
            plt.axis('off')
            plt.imsave(os.path.join(combined_image_folder, _filename), combined_image)

            # Clean up by closing figures to free memory
            plt.close('all')
    return (
        combined_image,
        combined_image_folder,
        fig1,
        fig2,
        image1,
        image2,
        pad_to_match_height,
        plot_boxes_on_image2,
        trim_white_space,
    )


@app.cell
def _(bboxes_df, os, periodical_mapping, plot_boxes_on_image, plt):
    _save_folder = 'data/image_with_bounding/test_set_post_process'

    os.makedirs(_save_folder, exist_ok=True)

    for _target_page in os.listdir('data/converted/ncse_bbox_test_set_hyphen'):
        _periodical_code = periodical_mapping.loc[periodical_mapping['periodical']==_target_page.split("-")[0], 'periodical_code'].iloc[0]
        #  Identify image folder using the periodical ID
        _image_path = 'data/converted/ncse_bbox_test_set_hyphen'

        _image = plot_boxes_on_image(bboxes_df[bboxes_df['filename']==_target_page], 
                            image_path = os.path.join(_image_path, _target_page), show_reading_order=True)

        _image.savefig(os.path.join(_save_folder, _target_page))
        plt.close()
    return


@app.cell
def _(
    List,
    ProcessPoolExecutor,
    as_completed,
    merge_consecutive_titles,
    pd,
    split_and_reclassify,
    tqdm,
):
    def create_articles(df):
        """
        Merges consecutive rows in a DataFrame, combining titles with their subsequent text.
        Creates a new 'title' column containing the title text while also including it in the main content.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame containing required columns:
            - 'class', 'page_id', 'reading_order', 'sub_order', 'content', 'issue_id'

        Returns:
        --------
        pandas.DataFrame
            A new DataFrame with merged content, titles integrated into content, and a new 'title' column.
        """    
        if len(df) == 0:
            return df.copy()

        # Create a working copy of the DataFrame
        working_df = df.copy()

        # Sort the DataFrame
        working_df = working_df.sort_values(['page_id', 'reading_order', 'sub_order']).reset_index(drop=True)

        # Initialize merge groups and titles
        current_group = 0
        merge_groups = []
        group_titles = {}
        current_title = None

        # Iterate through rows to create proper merge groups
        for i in range(len(working_df)):
            curr_row = working_df.iloc[i]

            if i == 0:
                if curr_row['class'] == 'title':
                    current_title = curr_row['content']
                merge_groups.append(current_group)
                continue

            prev_row = working_df.iloc[i-1]

            # Start new group if issue_ids don't match
            if prev_row['issue_id'] != curr_row['issue_id']:
                current_group += 1
                current_title = None

            # Update current title if we encounter a title
            if curr_row['class'] == 'title':
                current_group += 1
                current_title = curr_row['content']

            merge_groups.append(current_group)
            if current_title is not None:
                group_titles[current_group] = current_title

        working_df['merge_group'] = merge_groups

        # Process groups and create new DataFrame
        result_rows = []

        for group in working_df['merge_group'].unique():
            group_df = working_df[working_df['merge_group'] == group]

            # Get the title for this group
            group_title = group_titles.get(group, None)

            # If group contains multiple rows or has a title
            if len(group_df) >= 1:
                # Get the first non-title row of the group
                first_row = group_df[group_df['class'] != 'title'].iloc[0] if any(group_df['class'] != 'title') else group_df.iloc[0]

                # Collect all content (excluding title rows)
                content_parts = []
                if group_title:
                    content_parts.append(group_title)
                content_parts.extend(group_df[group_df['class'] != 'title']['content'].tolist())

                # Merge the content
                merged_content = '\n\n'.join(content_parts)

                # Create new row
                new_row = first_row.copy()
                new_row['content'] = merged_content
                new_row['title'] = group_title  # Add title column
                new_row['class'] = 'content'    # Set class to content
                result_rows.append(new_row)

        # Create new DataFrame from the Series objects
        result_df = pd.DataFrame([row.to_dict() for row in result_rows])

        # Reset index and recalculate reading_order
        result_df = result_df.sort_values(['reading_order', 'sub_order'])
        result_df['reading_order'] = result_df.groupby('page_id').cumcount() + 1

        # Drop the merge_group column
        result_df = result_df.drop('merge_group', axis=1)

        return result_df.reset_index(drop=True)



    def process_document_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single batch (subset) of the DataFrame through all three functions.

        Parameters:
        -----------
        df_batch : pd.DataFrame
            A subset of the main DataFrame containing documents from a single issue.

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch.
        """
        # Apply the three functions in sequence
        df_batch = split_and_reclassify(df_batch)
        df_batch = merge_consecutive_titles(df_batch)
        df_batch = create_articles(df_batch)
        return df_batch

    def process_batch(df: pd.DataFrame, issue_ids: List[int]) -> pd.DataFrame:
        """
        Process a batch of issue_ids.

        Parameters:
        -----------
        df : pd.DataFrame
            The entire DataFrame.
        issue_ids : List[int]
            List of issue_ids to process.

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame batch.
        """
        batch_df = df[df['issue_id'].isin(issue_ids)].copy()
        return process_document_batch(batch_df)

    def process_documents(df: pd.DataFrame, batch_size: int = 100, parallel: bool = False, n_jobs: int = -1) -> pd.DataFrame:
        """
        Process the entire DataFrame in batches by issue_id, with option for parallel processing.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing documents.
        batch_size : int
            Number of unique issue_ids to process in each batch.
        parallel : bool
            Whether to use parallel processing.
        n_jobs : int
            Number of parallel jobs to run (-1 for all available cores).

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame with all transformations applied.
        """
        # Get unique issue_ids
        unique_issues = df['issue_id'].unique()

        # Create batches of issue_ids
        issue_batches = [unique_issues[i:i + batch_size] for i in range(0, len(unique_issues), batch_size)]

        processed_dfs = []

        if parallel:
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=n_jobs if n_jobs != -1 else None) as executor:
                futures = [executor.submit(process_batch, df, batch) for batch in issue_batches]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                    processed_dfs.append(future.result())
        else:
            # Process batches sequentially
            for batch in tqdm(issue_batches, desc="Processing batches"):
                processed_batch = process_batch(df, batch)
                processed_dfs.append(processed_batch)

        # Combine all processed batches
        final_df = pd.concat(processed_dfs, ignore_index=True)

        # Sort the final DataFrame
        final_df = final_df.sort_values(['issue_id', 'page_id', 'reading_order', 'sub_order'])

        return final_df.reset_index(drop=True)

    # Parallel processing
    # df_processed = process_documents(returned_docs, batch_size=100, parallel=True, n_jobs=-1)
    return (
        create_articles,
        process_batch,
        process_document_batch,
        process_documents,
    )


@app.cell
def _(
    create_articles,
    merge_consecutive_titles,
    os,
    pd,
    remove_line_breaks,
    split_and_reclassify,
):
    returned_docs_folder = 'data/download_jobs/ncse/dataframes/raw'


    #returned_docs = pd.concat([pd.read_parquet(os.path.join(returned_docs_folder, x)) for x in #os.listdir(returned_docs_folder)])

    returned_docs = pd.read_parquet(os.path.join(returned_docs_folder, 'Leader_issue_PDF_files.parquet'))
    returned_docs['content'] = returned_docs['content'].str.strip('`').str.replace('tsv', '', n=1)
    # Add the condition to change 'class' to 'text'
    returned_docs.loc[(returned_docs['completion_tokens'] > 50) & (returned_docs['class'] == 'title'), 'class'] = 'text'
    returned_docs.loc[returned_docs['class'] != 'table', 'content'] = remove_line_breaks(returned_docs.loc[returned_docs['class'] != 'table', 'content'])


    returned_docs2 = returned_docs.loc[returned_docs['issue_id']=='CLD-1860-10-06']
    # text boxes are split and the split boxes re-classified if some are in fact titles
    df = split_and_reclassify(returned_docs2)
    # The consecctive titles are merged
    df = merge_consecutive_titles(df)
    # The all consectutive non-titles are merged to form articles
    df = create_articles(df)
    return df, returned_docs, returned_docs2, returned_docs_folder


@app.cell
def _(process_documents, returned_docs):
    test = process_documents(returned_docs, batch_size = 1, parallel= True, n_jobs = 3)
    return (test,)


@app.cell
def _(returned_docs):
    returned_docs
    return


@app.cell
def _(temp5):
    temp5.sort_values(['page_number','reading_order'  ])[['page_number', 'reading_order', 'class', 'content', 'title']]
    return


@app.cell
def _(temp5):
    temp5.groupby('class').size()
    return


@app.cell
def _(os, pd, periodical_mapping, sns):
    returned_docs_folder2 = 'data/download_jobs/ncse/dataframes/raw'


    # Get all parquet files in the folder
    parquet_files = [f for f in os.listdir(returned_docs_folder2) if f.endswith('.parquet')]

    # List to store individual dataframes
    dataframes = []

    # Read all parquet files and store dataframes
    for file in parquet_files:
        _df = pd.read_parquet(os.path.join(returned_docs_folder2, file))
        _df['content'] = _df['content'].str.strip('`').str.replace('tsv', '', n=1)
        _df['periodical'] = _df['issue_id'].str.split('-').str[0]
        _df = _df.drop(columns = 'content')
        dataframes.append(_df)

    # Concatenate all dataframes into a single dataframe
    _combined_df = pd.concat(dataframes, ignore_index=True).merge(periodical_mapping, on='periodical', how='left')

    sns.displot(
        data=_combined_df.loc[(_combined_df['completion_tokens']<100) & (_combined_df['class']=='title')],
        x='completion_tokens',
        col='periodical_code',
        col_wrap=3,  # Adjust based on the number of unique periodicals
        kind='kde',
        height=4,
        aspect=1.5
    )
    return dataframes, file, parquet_files, returned_docs_folder2


@app.cell
def _(returned_docs):
    returned_docs['total_tokens'].describe()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
