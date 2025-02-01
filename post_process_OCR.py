import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    from function_modules.bbox_functions import plot_boxes_on_image
    import os 
    from tqdm import tqdm

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
        Path,
        all_data_files,
        bbox_path,
        bboxes_df,
        data,
        data_path,
        image_path,
        os,
        path_mapping,
        pd,
        periodical_mapping,
        plot_boxes_on_image,
        raw_bbox_path,
        raw_bboxes_df,
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
    os,
    path_mapping,
    periodical_mapping,
    raw_bboxes_df,
    test2,
):
    import numpy as np 
    import matplotlib.pyplot as plt
    import cv2

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
        cv2,
        fig1,
        fig2,
        image1,
        image2,
        np,
        pad_to_match_height,
        plot_boxes_on_image2,
        plt,
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
def _(raw_bboxes_df, test2):
    raw_bboxes_df[raw_bboxes_df['filename'].isin(test2['filename'].unique())].groupby('class').size()
    return


@app.cell
def _(bboxes_df, test2):
    bboxes_df[bboxes_df['page_id'].isin(test2['page_id'].unique())].groupby('class').size()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Split boxes where a line is entirely in caps

        It seems pretty common for a line that is entirely in caps to be a title. This means I can make a heuristic whereby a box of class 'text' is split if the line is all caps, the all caps lines are classed as "title" and all consecutive title rows are merged together
        """
    )
    return


@app.cell
def _(bboxes_df, test2):
    bboxes_df[bboxes_df['filename'].isin(test2['filename'].unique())]
    return


@app.cell
def _(pd):
    def is_title(text):
        # Function to check if a string meets title criteria
        vowels = set('AEIOU')

        # Check if the text is all uppercase
        if not text.upper() == text:
            return False

        # Split by any character that's not a letter or space
        import re
        word_groups = re.split(r'[^A-Z\s]', text)

        # For each continuous group of words
        for group in word_groups:
            # Remove spaces and check if this group meets criteria
            letters_only = group.replace(' ', '')
            if len(letters_only) >= 5:  # Check length
                vowel_count = sum(1 for c in letters_only if c in vowels)
                if vowel_count >= 2:  # Check vowels
                    return True
        return False

    def split_and_reclassify(bbox_df):


        new_rows = []

        for idx, row in bbox_df.iterrows():
            text = row['content']
            paragraphs = text.split('\n\n')

            if len(paragraphs) == 1:
                row['class2'] = row['class']
                row['sub_order'] = 1
                new_rows.append(row)
            else:
                for i, para in enumerate(paragraphs, 1):
                    if para.strip():  # Skip empty paragraphs
                        new_row = row.copy()
                        new_row['content'] = para.strip()
                        new_row['sub_order'] = i

                        # Check each paragraph against the new title criteria
                        if is_title(para.strip()):
                            new_row['class2'] = 'title'
                        else:
                            new_row['class2'] = row['class']

                        new_rows.append(new_row)

        # Create new dataframe from the processed rows
        new_df = pd.DataFrame(new_rows).reset_index(drop=True)

        return new_df


    def merge_consecutive_titles(df):
        # Create a copy to avoid modifying the original dataframe
        result_df = df.copy()

        # Initialize list to store indices to drop
        indices_to_drop = []

        # Initialize variables to track current merge group
        current_content = []
        current_start_idx = None

        # Iterate through rows
        for i in range(len(result_df)):
            current_row = result_df.iloc[i]

            # If we're not at the last row, get next row for comparison
            if i < len(result_df) - 1:
                next_row = result_df.iloc[i + 1]

                # Check if current and next rows should be merged
                should_merge = (
                    current_row['class2'] == 'title' and
                    next_row['class2'] == 'title' and
                    current_row['page_id'] == next_row['page_id'] and
                    current_row['box_page_id'] == next_row['box_page_id'] and
                    current_row['sub_order'] + 1 == next_row['sub_order']
                )

                if should_merge:
                    # Start new merge group if not already started
                    if current_start_idx is None:
                        current_start_idx = i
                        current_content = [current_row['content']]

                    current_content.append(next_row['content'])
                    indices_to_drop.append(i + 1)

                elif current_start_idx is not None:
                    # Merge the accumulated content into the first row
                    merged_content = '\n'.join(current_content)
                    result_df.at[current_start_idx, 'content'] = merged_content

                    # Reset tracking variables
                    current_content = []
                    current_start_idx = None

            elif current_start_idx is not None:
                # Handle the last merge group if exists
                merged_content = '\n'.join(current_content)
                result_df.at[current_start_idx, 'content'] = merged_content

        # Drop the merged rows using boolean indexing instead of index labels
        if indices_to_drop:
            result_df = result_df[~result_df.index.isin(indices_to_drop)]

        # Reset index
        result_df = result_df.reset_index(drop=True)

        # Recalculate sub_order within each box_page_id group
        result_df['sub_order'] = result_df.groupby(['page_id', 'box_page_id']).cumcount() + 1

        return result_df

    # Usage:
    # First run the split_and_reclassify function
    # df_split = split_and_reclassify(bbox_df)
    # Then merge the consecutive titles and recalculate sub_order
    # df_final = merge_consecutive_titles(df_split)

    # Test cases
    test_texts = [
        "THE STAR",              # Should be title
        "SIR",                   # Should not be title (too short)
        "---",                   # Should not be title
        "...",                   # Should not be title
        "N.I.A.B.P",            # Should not be title (broken by periods)
        "BREAKING NEWS",         # Should be title
        "HELLO WORLD",          # Should be title
        "THE END.",             # Should be title (ignoring period)
        "A.B HELLO WORLD C.D",  # Should be title (HELLO WORLD meets criteria)
        "NO VOWELS NTH"         # Should not be title (not enough vowels)
    ]

    # Test the function
    for text in test_texts:
        print(f"{text}: {'Is title' if is_title(text) else 'Not title'}")
    return (
        is_title,
        merge_consecutive_titles,
        split_and_reclassify,
        test_texts,
        text,
    )


@app.cell
def _(merge_consecutive_titles, pd, split_and_reclassify):
    test_set_df = pd.read_csv('data/download_jobs/experiments/dataframe/NCSE_deskew_True_max_ratio_1.csv')
    test_set_df['class'] = 'text'

    temp = split_and_reclassify(test_set_df)

    temp2 = merge_consecutive_titles(temp)
    return temp, temp2, test_set_df


@app.cell
def _(temp):
    temp[['class', 'class2', 'sub_order', 'content']]
    return


@app.cell
def _(temp2):
    temp2[['class', 'class2', 'sub_order', 'content']]
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
