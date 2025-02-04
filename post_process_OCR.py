import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    from function_modules.bbox_functions import plot_boxes_on_image
    from function_modules.analysis_functions import is_title, split_and_reclassify, merge_consecutive_titles
    import os 
    from tqdm import tqdm
    import numpy as np 
    import time


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
        is_title,
        merge_consecutive_titles,
        np,
        os,
        path_mapping,
        pd,
        periodical_mapping,
        plot_boxes_on_image,
        raw_bbox_path,
        raw_bboxes_df,
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
    np,
    os,
    path_mapping,
    periodical_mapping,
    raw_bboxes_df,
    test2,
):
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
def _(is_title):
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
        "NYNTH"         # Should not be title (not enough vowels)
    ]

    # Test the function
    for text in test_texts:
        print(f"{text}: {'Is title' if is_title(text) else 'Not title'}")
    return test_texts, text


@app.cell
def _(pd, split_and_reclassify):
    test_set_df = pd.read_csv('data/download_jobs/experiments/dataframe/NCSE_deskew_True_max_ratio_1.csv')
    #test_set_df['class'] = 'text'

    temp = split_and_reclassify(test_set_df)
    return temp, test_set_df


@app.cell
def _(merge_consecutive_titles, temp):


    temp2 = merge_consecutive_titles(temp)
    return (temp2,)


@app.cell
def _(temp2):
    temp2
    return


@app.cell
def _(temp):
    temp[['class', 'class2', 'sub_order', 'content']]
    return


@app.cell
def _(temp2):
    temp2[['class', 'class2', 'sub_order', 'content']]
    return


@app.cell
def _(os, pd, split_and_reclassify):
    returned_docs_folder = 'data/download_jobs/ncse/dataframes'


    returned_docs = pd.concat([pd.read_parquet(os.path.join(returned_docs_folder, x)) for x in os.listdir(returned_docs_folder)])


    #test_set_df['class'] = 'text'

    temp3 = split_and_reclassify(returned_docs)
    return returned_docs, returned_docs_folder, temp3


@app.cell
def _(merge_consecutive_titles, temp3):
    temp4 = merge_consecutive_titles(temp3)
    return (temp4,)


@app.cell
def _(returned_docs):
    returned_docs
    return


@app.cell
def _(returned_docs):
    returned_docs['total_tokens'].sum()/59000
    return


@app.cell
def _(returned_docs):
    returned_docs['total_tokens'].describe()
    return


@app.cell
def _():
    (20121/1e6)*0.15
    return


@app.cell
def _(is_title, np, pd, time):
    def split_and_reclassify2(bbox_df):
        """
        Splits text content into paragraphs and reclassifies them based on specific criteria.

        This function processes a DataFrame containing bounding box information and text content.
        It splits the text content into paragraphs (based on double newlines), assigns order numbers,
        and potentially reclassifies paragraphs as titles based on their content.

        Parameters:
        -----------
        bbox_df : pandas.DataFrame
            Input DataFrame containing at least the following columns:
            - 'content': str, the text content to be split
            - 'class': str, the original classification of the text

        Returns:
        --------
        pandas.DataFrame
            A new DataFrame with the following modifications:
            - Split paragraphs as separate rows
            - Added 'class2' column with potentially updated classifications
            - Added 'sub_order' column indicating the order of paragraphs
            - All original columns are preserved
            - Index is reset

        Notes:
        ------
        - Paragraphs are determined by double newline characters ('\n\n')
        - Empty paragraphs are skipped
        - If text contains only one paragraph, 'class2' will be same as original 'class'
        - For multiple paragraphs, each is evaluated using is_title() function to determine
          if it should be classified as a title

        """

        # Split the content into paragraphs and create a new DataFrame
        split_rows = bbox_df.assign(content=bbox_df['content'].str.split('\n\n')).explode('content')

        # Remove empty paragraphs
        split_rows = split_rows[split_rows['content'].str.strip() != '']

        # Add sub_order column
        split_rows['sub_order'] = split_rows.groupby(split_rows.index).cumcount() + 1

        # Apply the is_title function to determine class2
        split_rows['class2'] = split_rows.apply(lambda row: 'title' if is_title(row['content'].strip()) else row['class'], axis=1)

        # Reset index
        split_rows = split_rows.reset_index(drop=True)

        return split_rows



    def split_and_reclassify_optimized(bbox_df):
        """
        Optimized version of split_and_reclassify function.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        df = bbox_df.copy()

        # Create lists to store the expanded data
        contents = []
        sub_orders = []
        class2 = []
        repeat_indices = []

        # Process each row once
        for idx, content in enumerate(df['content']):
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            if len(paragraphs) == 1:
                contents.append(paragraphs[0])
                sub_orders.append(1)
                class2.append(df.iloc[idx]['class'])
                repeat_indices.append(idx)
            else:
                contents.extend(paragraphs)
                sub_orders.extend(range(1, len(paragraphs) + 1))
                class2.extend(['title' if is_title(p) else df.iloc[idx]['class'] 
                             for p in paragraphs])
                repeat_indices.extend([idx] * len(paragraphs))

        # Create new DataFrame efficiently
        new_df = df.iloc[repeat_indices].copy()
        new_df['content'] = contents
        new_df['sub_order'] = sub_orders
        new_df['class2'] = class2

        return new_df.reset_index(drop=True)



    def create_test_data(n_rows):
        return pd.DataFrame({
            'content': [f"Para 1\n\nPara 2\n\nPara 3" * np.random.randint(1, 4) for _ in range(n_rows)],
            'class': ['class1'] * n_rows
        })

    def benchmark_comparison(n_rows=500000):
        test_df = create_test_data(n_rows)

        # Benchmark pandas-centric approach
        start = time.time()
        result1 = split_and_reclassify2(test_df)
        time1 = time.time() - start

        # Benchmark list-based approach
        start = time.time()
        result2 = split_and_reclassify_optimized(test_df)
        time2 = time.time() - start

        print(f"Pandas-centric approach: {time1:.2f} seconds")
        print(f"List-based approach: {time2:.2f} seconds")

        # Verify results are equivalent
        print("\nResults equivalent:", result1.equals(result2))

    # Run benchmark
    benchmark_comparison()
    return (
        benchmark_comparison,
        create_test_data,
        split_and_reclassify2,
        split_and_reclassify_optimized,
    )


@app.cell
def _(
    create_test_data,
    is_title,
    np,
    pd,
    split_and_reclassify2,
    split_and_reclassify_optimized,
    time,
):
    def split_and_reclassify_optimized_v2(bbox_df):
        # Perform all string operations in one go
        split_rows = (bbox_df
                     .assign(content=lambda x: x['content'].str.split('\n\n'))
                     .explode('content')
                     .assign(content=lambda x: x['content'].str.strip())
                     .query('content != ""'))

        # Add sub_order efficiently
        split_rows['sub_order'] = split_rows.groupby(split_rows.index).cumcount() + 1

        # Vectorize the is_title check if possible
        # This depends on your is_title implementation
        if hasattr(pd.Series, 'str.istitle'):  # If is_title is just checking istitle()
            split_rows['class2'] = np.where(split_rows['content'].str.istitle(),
                                          'title',
                                          split_rows['class'])
        else:
            # Fall back to apply if we need custom is_title logic
            split_rows['class2'] = split_rows.apply(
                lambda row: 'title' if is_title(row['content']) else row['class'],
                axis=1
            )

        return split_rows.reset_index(drop=True)

    def split_and_reclassify_optimized_v3(bbox_df):
        # Pre-allocate the DataFrame with estimated size
        estimated_size = (bbox_df['content'].str.count('\n\n') + 1).sum()

        # Split and explode in one operation
        split_rows = pd.concat([
            bbox_df.assign(
                content=lambda x: x['content'].str.split('\n\n'),
                _split_idx=lambda x: x.index
            ).explode('content')
        ])

        # In-place operations where possible
        split_rows.loc[:, 'content'] = split_rows['content'].str.strip()
        split_rows = split_rows[split_rows['content'] != '']

        # Efficient sub_order calculation
        split_rows.loc[:, 'sub_order'] = split_rows.groupby('_split_idx').cumcount() + 1

        # Vectorized class2 assignment
        if hasattr(pd.Series, 'str.istitle'):
            split_rows.loc[:, 'class2'] = np.where(
                split_rows['content'].str.istitle(),
                'title',
                split_rows['class']
            )
        else:
            split_rows.loc[:, 'class2'] = split_rows.apply(
                lambda row: 'title' if is_title(row['content']) else row['class'],
                axis=1
            )

        # Clean up and return
        return split_rows.drop('_split_idx', axis=1).reset_index(drop=True)


    def benchmark_all_versions(n_rows=200000):
        test_df = create_test_data(n_rows)

        versions = {
            'Original Pandas': split_and_reclassify2,
            'List-based': split_and_reclassify_optimized,
            'Optimized Pandas v2': split_and_reclassify_optimized_v2,
            'Optimized Pandas v3': split_and_reclassify_optimized_v3
        }

        results = {}
        for name, func in versions.items():
            start = time.time()
            result = func(test_df)
            duration = time.time() - start
            results[name] = {'time': duration, 'result': result}
            print(f"{name}: {duration:.2f} seconds")

        # Verify all results are equivalent
        first_result = next(iter(results.values()))['result']
        for name, res in results.items():
            if name != list(results.keys())[0]:
                print(f"\n{name} equivalent to original: {first_result.equals(res['result'])}")

    benchmark_all_versions()
    return (
        benchmark_all_versions,
        split_and_reclassify_optimized_v2,
        split_and_reclassify_optimized_v3,
    )


@app.cell
def _(create_test_data, np, re, split_and_reclassify2, time):
    def is_title_vectorized(series):
        """
        Vectorized version of is_title function for pandas Series.
        """
        # Check if all uppercase
        is_upper_mask = series == series.str.upper()

        # Create the regex pattern for splitting
        pattern = r'[^A-Z\s]'

        # Function to check a single group of words
        def check_group(group):
            letters_only = group.replace(' ', '')
            if len(letters_only) >= 5:
                vowel_count = sum(1 for c in letters_only if c in 'AEIOU')
                return vowel_count >= 2
            return False

        # Vectorized function to process each text
        def process_text(text):
            if not isinstance(text, str):
                return False
            if not text == text.upper():
                return False
            word_groups = re.split(pattern, text)
            return any(check_group(group) for group in word_groups)

        # Use numpy vectorize for better performance
        vectorized_process = np.vectorize(process_text)

        return vectorized_process(series)

    def split_and_reclassify_final(bbox_df):
        """
        Optimized version with vectorized title checking.
        """
        # Chain operations efficiently
        split_rows = (bbox_df
                     .assign(content=lambda x: x['content'].str.split('\n\n'))
                     .explode('content')
                     .assign(content=lambda x: x['content'].str.strip())
                     .query('content != ""'))

        # Add sub_order efficiently
        split_rows['sub_order'] = split_rows.groupby(split_rows.index).cumcount() + 1

        # Vectorized class2 assignment with optimized title checking
        split_rows['class2'] = np.where(is_title_vectorized(split_rows['content']),
                                      'title',
                                      split_rows['class'])

        return split_rows.reset_index(drop=True)


    def is_title_vectorized_v2(series):
        """
        Alternative vectorized approach using more pandas string methods.
        """
        # Check if all uppercase
        is_upper_mask = series == series.str.upper()

        # Remove non-letters and spaces
        letters_only = series.str.replace(r'[^A-Z]', '', regex=True)

        # Check length condition
        length_mask = letters_only.str.len() >= 5

        # Count vowels
        vowel_counts = letters_only.str.count('[AEIOU]')
        vowel_mask = vowel_counts >= 2

        return is_upper_mask & length_mask & vowel_mask

    def split_and_reclassify_final_v2(bbox_df):
        """
        Version with alternative vectorized title checking.
        """
        split_rows = (bbox_df
                     .assign(content=lambda x: x['content'].str.split('\n\n'))
                     .explode('content')
                     .assign(content=lambda x: x['content'].str.strip())
                     .query('content != ""'))

        split_rows['sub_order'] = split_rows.groupby(split_rows.index).cumcount() + 1

        # Use alternative vectorized title checking
        split_rows['class2'] = np.where(is_title_vectorized_v2(split_rows['content']),
                                      'title',
                                      split_rows['class'])

        return split_rows.reset_index(drop=True)

    def benchmark_final_versions(n_rows=200000):
        test_df = create_test_data(n_rows)

        versions = {
            'Original': split_and_reclassify2,
            'Vectorized v1': split_and_reclassify_final,
            'Vectorized v2': split_and_reclassify_final_v2
        }

        results = {}
        for name, func in versions.items():
            start = time.time()
            result = func(test_df)
            duration = time.time() - start
            results[name] = {'time': duration, 'result': result}
            print(f"{name}: {duration:.2f} seconds")

        # Verify results are equivalent
        first_result = next(iter(results.values()))['result']
        for name, res in results.items():
            if name != list(results.keys())[0]:
                print(f"\n{name} equivalent to original: {first_result.equals(res['result'])}")

    benchmark_final_versions()
    return (
        benchmark_final_versions,
        is_title_vectorized,
        is_title_vectorized_v2,
        split_and_reclassify_final,
        split_and_reclassify_final_v2,
    )


app._unparsable_cell(
    r"""
    (import marimo as mo
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
