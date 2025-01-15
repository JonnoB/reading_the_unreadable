import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __():
    from cleanlab.object_detection.filter import find_label_issues
    from cleanlab.object_detection.rank import get_label_quality_scores, compute_badloc_box_scores
    from cleanlab.object_detection.summary import visualize
    from archived.objdet_dataset_helpers import create_image_list
    from helper_functions import scale_bbox

    import os
    import pandas as pd
    import numpy as np
    import json

    image_folder = '/media/jonno/ncse/converted/all_files_png_72/English_Womans_Journal_issue_PDF_files'

    file_name_to_id_map = pd.read_parquet('data/file_name_to_id_map.parquet')

    image_list = create_image_list(file_name_to_id_map.loc[file_name_to_id_map['publication_id'] ==24], image_folder)

    dimensions_dict = {item['id']: (item['width'], item['height'])for item in image_list}
    return (
        compute_badloc_box_scores,
        create_image_list,
        dimensions_dict,
        file_name_to_id_map,
        find_label_issues,
        get_label_quality_scores,
        image_folder,
        image_list,
        json,
        np,
        os,
        pd,
        scale_bbox,
        visualize,
    )


@app.cell
def __(dimensions_dict, pd, scale_bbox):
    meta_data_df = pd.read_parquet('data/ncse_data_metafile.parquet')
    meta_data_df['image_name'] = meta_data_df['periodical_abbrev'] + '-' + meta_data_df['issue_date'] +"_page_" + meta_data_df['page_number'].astype(str) + '.png'
    meta_data_df = meta_data_df.loc[meta_data_df['periodical_abbrev']=='EWJ']

    meta_data_df['width_adjusted'] = meta_data_df['page_id'].map(lambda x: dimensions_dict.get(x)[0] if x in dimensions_dict else None)
    meta_data_df['height_adjusted'] = meta_data_df['page_id'].map(lambda x: dimensions_dict.get(x)[1] if x in dimensions_dict else None)

    meta_data_df.sort_values('image_name', inplace=True)

    _grouped = meta_data_df.groupby(['width', 'height', 'width_adjusted', 'height_adjusted'])

    # Process each group and concatenate results
    scaled_dfs = []
    for (orig_w, orig_h, new_w, new_h), group in _grouped:
        scaled_df = scale_bbox(group, 
                              original_size=(orig_w, orig_h),
                              new_size=(new_w, new_h))
        scaled_dfs.append(scaled_df)

    meta_data_df = pd.concat(scaled_dfs)

    return (
        group,
        meta_data_df,
        new_h,
        new_w,
        orig_h,
        orig_w,
        scaled_df,
        scaled_dfs,
    )


@app.cell
def __(meta_data_df):
    meta_data_df#.groupby(['article_type_id', 'publication_id']).size()
    return


@app.cell
def __(meta_data_df, os, pd):
    cleanlab_data_folder = "data/cross_validate_for_cleanlab/EWJ"

    xval_label_data = pd.concat([pd.read_csv(os.path.join(cleanlab_data_folder, file)) for file in os.listdir(cleanlab_data_folder)])

    xval_label_data = xval_label_data.loc[xval_label_data['image_name'].isin(meta_data_df['image_name'].unique())]

    xval_label_data.sort_values('image_name', inplace=True)
    return cleanlab_data_folder, xval_label_data


@app.cell
def __(xval_label_data):
    xval_label_data
    return


@app.cell
def __(np):
    def format_dataframe_predictions(df, num_classes, group_by_image=True):
        """
        Convert DataFrame predictions to match detectron2 prediction format

        Parameters:
        df: DataFrame with columns ['image_name', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2']
        num_classes: number of classes in the model
        group_by_image: whether to group results by image

        Returns:
        If group_by_image=True: List of lists, where each inner list contains predictions for one image
        If group_by_image=False: Single list of predictions for one image
        """

        def process_single_image(image_df):
            # Initialize empty list for each class
            res = [[] for _ in range(num_classes)]

            # Group predictions by class_id
            for _, row in image_df.iterrows():
                class_id = int(row['class_id'])
                # Create box coordinates with confidence [x1, y1, x2, y2, confidence]
                box_cord = [
                    float(row['x1']), 
                    float(row['y1']), 
                    float(row['x2']), 
                    float(row['y2']), 
                    float(row['confidence'])
                ]
                res[class_id].append(box_cord)

            # Convert to numpy arrays with proper shape
            res2 = []
            for i in res:
                if len(i) == 0:
                    q = np.array(i, dtype=np.float32).reshape((0, 5))
                else:
                    q = np.array(i, dtype=np.float32)
                res2.append(q)

            return res2

        if group_by_image:
            # Group by image_name and process each image
            results = []
            for image_name, group_df in df.groupby('image_name'):
                results.append(process_single_image(group_df))
            return results
        else:
            # Process single image
            return process_single_image(df)
    return (format_dataframe_predictions,)


@app.cell
def __(format_dataframe_predictions, xval_label_data):
    predictions = format_dataframe_predictions(xval_label_data, 3)
    return (predictions,)


@app.cell
def __(np):

    def create_label_set(df, bbox_col='bbox', label_col='label', image_col='image_name'):
        """
        Convert DataFrame to list of dictionaries with bboxes, labels, and image names.

        Parameters:
        df (pandas.DataFrame): Input DataFrame
        bbox_col (str): Name of column containing bbox dictionaries
        label_col (str): Name of column containing labels
        image_col (str): Name of column containing image names

        Returns:
        list: List of dictionaries with 'bboxes', 'labels', and 'image_name' keys
        """
        grouped = df.groupby(image_col)

        result = []

        # Process each group (image)
        for image_name, group in grouped:
            # Extract bboxes and convert to float
            bboxes = np.array([
                [float(box['x0']), float(box['y0']), float(box['x1']), float(box['y1'])] 
                for box in group[bbox_col]
            ], dtype=np.float32)  # Explicitly set dtype to float32

            # Extract labels and ensure they're integers
            labels = group[label_col].astype(int).to_numpy() -1 # Yolo is zero indexed

            # Create dictionary for this image
            image_dict = {
                'bboxes': bboxes,
                'labels': labels,
                'image_name': str(image_name)  # Ensure image_name is string
            }

            result.append(image_dict)

        return result

    def create_label_set(df, label_col='label', image_col='image_name'):
        """
        Convert DataFrame to list of dictionaries with bboxes, labels, and image names.

        Parameters:
        df (pandas.DataFrame): Input DataFrame with scaled_x0, scaled_x1, scaled_y0, scaled_y1 columns
        label_col (str): Name of column containing labels
        image_col (str): Name of column containing image names

        Returns:
        list: List of dictionaries with 'bboxes', 'labels', and 'image_name' keys
        """
        grouped = df.groupby(image_col)

        result = []

        # Process each group (image)
        for image_name, group in grouped:
            # Extract bboxes using the scaled columns
            bboxes = np.array([
                [float(row['scaled_x0']), float(row['scaled_y0']), 
                 float(row['scaled_x1']), float(row['scaled_y1'])] 
                for _, row in group.iterrows()
            ], dtype=np.float32)  # Explicitly set dtype to float32

            # Extract labels and ensure they're integers
            labels = group[label_col].astype(int).to_numpy() - 1  # Yolo is zero indexed

            # Create dictionary for this image
            image_dict = {
                'bboxes': bboxes,
                'labels': labels,
                'image_name': str(image_name)  # Ensure image_name is string
            }

            result.append(image_dict)

        return result
    return (create_label_set,)


@app.cell
def __(create_label_set, meta_data_df):
    labels = create_label_set(meta_data_df, label_col='article_type_id', image_col='image_name')
    return (labels,)


@app.cell
def __(find_label_issues, get_label_quality_scores, labels, predictions):
    # To get boolean vector of label issues for all images
    has_label_issue = find_label_issues(labels, predictions)

    # To get label quality scores for all images
    label_quality_scores = get_label_quality_scores(labels, predictions)
    return has_label_issue, label_quality_scores


@app.cell
def __(has_label_issue, label_quality_scores, meta_data_df):
    test2 = meta_data_df[['image_name', 'periodical_abbrev']].copy().drop_duplicates()
    test2.reset_index(inplace = True, drop = True)

    test2['label_issues'] = has_label_issue

    test2['label_quality_scores'] = label_quality_scores

    #test2['badloc'] = compute_badloc_box_scores(labels=labels,predictions= predictions)
    return (test2,)


@app.cell
def __(test2):
    test2
    return


@app.cell
def __(image_folder, labels, os, test2, visualize):
    target_image = 'EWJ-1864-05-02_page_14.png'


    image_id = test2.loc[test2['image_name']== target_image].index[0]
    print(image_id)

    visualize(image = os.path.join(image_folder, test2.loc[image_id, 'image_name']), 
                           label= labels[image_id],
                     #     prediction = predictions[image_id]
             )
    return image_id, target_image


if __name__ == "__main__":
    app.run()
