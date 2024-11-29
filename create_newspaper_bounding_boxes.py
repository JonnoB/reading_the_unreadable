import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __():
    import os 
    from helper_functions import files_to_df_func, scale_bbox
    import pandas as pd
    from tqdm import tqdm
    import shutil
    from pdf2image import convert_from_path
    import json
    from PIL import Image
    from sklearn.model_selection import KFold

    import shutil
    import yaml


    data_folder = 'data'
    #The folder where the pdfs have been converted to PNGs of various dpi.
    #To create better bounding boxes 72 dpi will be used.
    #converted_folder = '/media/jonno/ncse/converted/all_files_png_72/English_Womans_Journal_issue_PDF_files'
    converted_folder = '/media/jonno/ncse/converted/all_files_png_72'

    transcript_files = os.path.join(data_folder, 'transcripts/transcription_files')
    test_pdf_path = os.path.join(data_folder, 'ncse_test_pdf_date_names')

    test_pdf_path_correct_names = os.path.join(data_folder, 'ncse_test_pdf')
    test_jpg_path = os.path.join(data_folder, 'ncse_test_jpg')


    periodical_folders = ['English_Womans_Journal_issue_PDF_files', 'Leader_issue_PDF_files', 'Monthly_Repository_issue_PDF_files',
                         'Northern_Star_issue_PDF_files', 'Publishers_Circular_issue_PDF_files', 'Tomahawk_issue_PDF_files']
    return (
        Image,
        KFold,
        convert_from_path,
        converted_folder,
        data_folder,
        files_to_df_func,
        json,
        os,
        pd,
        periodical_folders,
        scale_bbox,
        shutil,
        test_jpg_path,
        test_pdf_path,
        test_pdf_path_correct_names,
        tqdm,
        transcript_files,
        yaml,
    )


@app.cell
def __(mo):
    mo.md("""## Create a meta data file at article level""")
    return


@app.cell
def __(ast, data_folder, os, pd, tqdm):
    _file_name = os.path.join(data_folder, 'ncse_data_metafile.parquet')

    # We need to add the page width and height to be able to scale to bounding boxes properly
    _page_data = pd.read_parquet(os.path.join(data_folder,'periodicals_page.parquet'))

    issue_data = pd.read_parquet(os.path.join(data_folder,'periodicals_issue.parquet'))
    issue_data['periodical_abbrev'] = issue_data['pdf'].str.split('-').str[1]
    issue_data['filename'] = issue_data['periodical_abbrev'] + '_' + issue_data['issue_date'].astype(str) + '.pdf'


    if not os.path.isfile(_file_name):
        test_file_meta_data = []
        file_folder ='new_parquet' #'ncse_text_chunks' # 'new_parquet'
        for file in tqdm(os.listdir(os.path.join(data_folder, file_folder))):

            _temp = pd.read_parquet(os.path.join(data_folder, file_folder, file))
            #load all necessary columns
            _temp = _temp.loc[:,['id', 'article_type_id', 'issue_id', 'page_id', 
                                 'issue_date', 'page_number', 'bounding_box']]

            #temp.drop(columns='content_html', inplace=True)
            test_file_meta_data.append(_temp)
            #delete as I feel there is some memory leak going on
            del _temp

        test_file_meta_data  = pd.concat(test_file_meta_data, ignore_index=True)

        test_file_meta_data['issue_date'] = test_file_meta_data['issue_date'].astype(str)

        test_file_meta_data = test_file_meta_data.merge(_page_data[['id', 'height', 'width']].set_index('id'), 
                                                        left_on = 'page_id', right_index= True)

        #add in the periodical id so that the file names can be re-constructed
        test_file_meta_data = test_file_meta_data.merge(issue_data[['id', 'periodical_abbrev', 'publication_id']].set_index('id'),
                                                       left_on = 'issue_id', right_index= True)

        #test_file_meta_data['bounding_box'] = test_file_meta_data.apply(lambda row: eval(row['bounding_box']) if isinstance(row['bounding_box'], str) else row['bounding_box'], axis = 1)
        test_file_meta_data['bounding_box'] = test_file_meta_data.apply(
            lambda row: ast.literal_eval(row['bounding_box']) 
            if isinstance(row['bounding_box'], str) 
            else row['bounding_box'], 
            axis=1
        )

        # Then convert dict values to int and create list
        test_file_meta_data['bounding_box_list'] = test_file_meta_data['bounding_box'].apply(
            lambda x: [int(v) for v in x.values()]
        )

        # Area = (x1 - x0) * (y1 - y0)
        test_file_meta_data['area'] = test_file_meta_data['bounding_box_list'].apply(lambda x: (x[1] - x[0]) * (x[3] - x[2]))

        test_file_meta_data.to_parquet(_file_name)
    else:
        test_file_meta_data = pd.read_parquet(_file_name)
    return file, file_folder, issue_data, test_file_meta_data


@app.cell
def __(mo):
    mo.md(r"""#NOTE ONLY DOES EWJ FOR SPEED!!!""")
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""## Create a dataframe for the stored page images""")
    return


@app.cell
def __():
    return


@app.cell
def __(
    converted_folder,
    os,
    pd,
    periodical_folders,
    test_file_meta_data,
    tqdm,
):
    _target_file = 'data/file_name_to_id_map.parquet'

    if not os.path.exists(_target_file):

        image_file_list = []
        for _folder in tqdm(periodical_folders):
            image_file_list = image_file_list + os.listdir(os.path.join(converted_folder, _folder))

        stored_image_files_df = pd.DataFrame({'filename':image_file_list})

        stored_image_files_df['periodical_abbrev'] = stored_image_files_df['filename'].str.split('-').str[0]
        stored_image_files_df['issue_date'] = stored_image_files_df['filename'].str.extract(r'-(\d{4}-\d{2}-\d{2})')
        stored_image_files_df['page_number'] = stored_image_files_df['filename'].str.extract(r'page_(\d+)').astype(int)

        _merge_columns = ['periodical_abbrev', 'issue_date', 'page_number']

        # First, calculate area sum grouped by the relevant columns
        _area_sums = test_file_meta_data.groupby(['page_id'])['area'].sum().reset_index()

        # Then, perform the drop duplicates and merge operation without the area column
        _base_data = test_file_meta_data[['issue_id', 'page_id', 'publication_id', 'width', 'height'] + _merge_columns].drop_duplicates()

        # Merge the area sums back
        _base_data = _base_data.merge(_area_sums, on=[ 'page_id'])

        # Finally, perform your original merge with stored_image_files_df
        file_name_to_id_map = _base_data.merge(
            stored_image_files_df.set_index(_merge_columns),
            left_on=_merge_columns, 
            right_index=True
        )
        file_name_to_id_map.rename(columns = {'area':'bbox_total_area'},inplace=True)
        file_name_to_id_map.to_parquet(_target_file)
        file_name_to_id_map['percent_cover'] = file_name_to_id_map['bbox_total_area']/(file_name_to_id_map['width'] * file_name_to_id_map['height']) 

    else:
        file_name_to_id_map = pd.read_parquet(_target_file)
    return file_name_to_id_map, image_file_list, stored_image_files_df


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map['percent_cover'].describe()
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map
    return


@app.cell
def __(file_name_to_id_map):
    import seaborn as sns
    sns.kdeplot(data = file_name_to_id_map, x = 'bbox_total_area', hue = 'publication_id')
    return (sns,)


@app.cell
def __(mo):
    mo.md(r"""There are some duplicated pages, I need to decide whether to drop them or include them and assume they will be removed when I see that the bounding boxes are all totally wrong. The overlap is 1000 pages aka 1% of the total number.""")
    return


@app.cell
def __(issue_data):
    _temp = issue_data.loc[issue_data['filename'].isin(issue_data.loc[issue_data['filename'].duplicated(), 'filename'])]
    print(_temp['number_of_pages'].sum())

    _temp
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Create the image filepath to page id map

        This allows me to create a json file that contains both images and bounding boxes
        """
    )
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map.loc[file_name_to_id_map['publication_id']==24]
    return


@app.cell
def __():
    return


@app.cell
def __(test_file_meta_data):
    test_file_meta_data.loc[test_file_meta_data['id']==503326]
    return


@app.cell
def __(Image, KFold, json, os, scale_bbox):
    def create_image_list(image_df, local_image_dir, coco_image_dir = ''):
        """Create list of image dictionaries
        image_df: dataframe. A pandas dataframe containing image meta data
        local_image_dir: str. The path to the images on the computer used to open images and get addition info
        coco_image_dir: str the path in the coco dataset, defaults to nothing so that json and images are all at same level.
        """
        image_list = []
        for idx, row in image_df.iterrows():
            image_path = os.path.join(local_image_dir, row['filename'])
            with Image.open(image_path) as img:
                width, height = img.size

            image_list.append({
                "id": row['page_id'],
                "file_name": os.path.join(os.path.basename(coco_image_dir), row['filename']),
                "width": width,
                "height": height,
                "license": 1,
            })
        return image_list

    def to_coco_format(bbox):
        """
        Convert [x1, x2, y1, y2] to COCO format [x, y, width, height]
        where (x, y) is the top-left corner
        """
        x1, x2, y1, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return [x1, y1, width, height]

    def create_annotation_list(bbox_df, image_list):
        """Create list of annotation dictionaries"""

        image_dict = {item['id']: item for item in image_list}
        dimensions_dict = {item['id']: (item['width'], item['height'])for item in image_list}

        temp_list = []

        for idx, row in bbox_df.iterrows():
            try:
                #scale bounding boxes to current image size using reference from original scan
                bounding_box_new = scale_bbox(row['bounding_box_list'], 
                                            original_size=(row['width'], row['height']), 
                                            new_size=dimensions_dict[row['page_id']])
                bounding_box_new = to_coco_format(bounding_box_new)

                temp_list.append({
                    "id": row['id'],
                    "image_id": row['page_id'],
                    "category_id": row['article_type_id'],
                    "bbox": bounding_box_new,
                    "area": bounding_box_new[2] * bounding_box_new[3],
                    "iscrowd": 0
                })

            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue

        return temp_list

    def turn_into_coco(meta_data_df, page_df, local_image_dir, coco_image_dir = ''):

        images_list = create_image_list(page_df, 
                                  local_image_dir,
                                   coco_image_dir    )

        annotations_list = create_annotation_list(meta_data_df, images_list )


        categories_list = [
                {
                    "id": 1,
                    "name": "article",
                    "supercategory": "text"
                },        {
                    "id": 2,
                    "name": "advert",
                    "supercategory": "text"
                },        {
                    "id": 3,
                    "name": "image",
                    "supercategory": "image"
                }
            ]

        return {'images': images_list, 'annotations': annotations_list, 'categories': categories_list}



    def create_cross_validation_coco(meta_data_df, page_df, local_image_dir, output_dir, n_folds=5, coco_image_dir='', random_state=42):
        """
        Create and save x-fold cross-validation COCO format JSON files

        Parameters:
        -----------
        meta_data_df : DataFrame
            DataFrame containing annotation metadata
        page_df : DataFrame
            DataFrame containing page/image information
        local_image_dir : str
            Path to local image directory
        output_dir : str
            Directory where JSON files will be saved
        n_folds : int
            Number of folds for cross-validation
        coco_image_dir : str
            Path for images in COCO dataset
        random_state : int
            Random seed for reproducibility
        """

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get unique page IDs
        unique_pages = page_df['page_id'].unique()

        # Initialize K-fold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # Perform k-fold split on page IDs
        for fold, (train_idx, val_idx) in enumerate(kf.split(unique_pages)):
            # Get train and validation page IDs
            train_pages = unique_pages[train_idx]
            val_pages = unique_pages[val_idx]

            # Filter DataFrames for train set
            train_page_df = page_df[page_df['page_id'].isin(train_pages)]
            train_meta_df = meta_data_df[meta_data_df['page_id'].isin(train_pages)]

            # Filter DataFrames for validation set
            val_page_df = page_df[page_df['page_id'].isin(val_pages)]
            val_meta_df = meta_data_df[meta_data_df['page_id'].isin(val_pages)]

            # Create COCO format for train set
            train_coco = turn_into_coco(
                train_meta_df,
                train_page_df,
                local_image_dir,
                coco_image_dir
            )

            # Create COCO format for validation set
            val_coco = turn_into_coco(
                val_meta_df,
                val_page_df,
                local_image_dir,
                coco_image_dir
            )

            # Save train JSON
            train_path = os.path.join(output_dir, f'train_fold_{fold}.json')
            with open(train_path, 'w') as f:
                json.dump(train_coco, f)

            # Save validation JSON
            val_path = os.path.join(output_dir, f'val_fold_{fold}.json')
            with open(val_path, 'w') as f:
                json.dump(val_coco, f)

            print(f"Fold {fold + 1}/{n_folds} completed")
            print(f"Train set size: {len(train_pages)} pages")
            print(f"Validation set size: {len(val_pages)} pages")
            print("------------------------")
    return (
        create_annotation_list,
        create_cross_validation_coco,
        create_image_list,
        to_coco_format,
        turn_into_coco,
    )


@app.cell
def __(
    converted_folder,
    file_name_to_id_map,
    os,
    periodical_folders,
    test_file_meta_data,
    turn_into_coco,
):
    EWJ_journal = turn_into_coco(meta_data_df = test_file_meta_data.loc[test_file_meta_data['publication_id']==24], 
                          page_df= file_name_to_id_map.loc[file_name_to_id_map['publication_id']==24], 
                          local_image_dir = os.path.join(converted_folder, periodical_folders[0]),
                        coco_image_dir=   os.path.join(converted_folder, periodical_folders[0])   )
    return (EWJ_journal,)


@app.cell
def __(EWJ_journal, json):
    with open('data/annotations.json', 'w') as f:
        json.dump(EWJ_journal, f)
    return (f,)


@app.cell
def __(mo):
    mo.md(
        """
        create_cross_validation_coco(meta_data_df = test_file_meta_data.loc[test_file_meta_data['publication_id']==24], 
                              page_df= file_name_to_id_map.loc[file_name_to_id_map['publication_id']==24], 
                              local_image_dir = os.path.join(converted_folder, periodical_folders[0]),
                            output_dir = os.path.join(converted_folder, 'EWJ_coco'), 
                            n_folds=5,    
                            coco_image_dir=   os.path.join(converted_folder, periodical_folders[0]))
        """
    )
    return


@app.cell
def __(Image, KFold, os, shutil, yaml):
    def convert_bbox_to_yolo(bbox, img_width, img_height):
        """
        Convert [x1, x2, y1, y2] format directly to YOLO format [x_center, y_center, width, height]
        All values in YOLO format are normalized to [0, 1]
        """
        x1, x2, y1, y2 = bbox

        # Calculate width and height
        width = x2 - x1
        height = y2 - y1

        # Calculate center coordinates
        x_center = x1 + width/2
        y_center = y1 + height/2

        # Normalize
        x_center = x_center / img_width
        y_center = y_center / img_height
        width = width / img_width
        height = height / img_height

        return [x_center, y_center, width, height]

    def create_yolo_annotation(image_path, annotations, img_width, img_height):
        """Create YOLO format annotation file content"""
        yolo_annotations = []

        for ann in annotations:
            category_id = ann['article_type_id'] - 1  # YOLO uses 0-based indexing
            bbox = convert_bbox_to_yolo(ann['bounding_box_list'], img_width, img_height)
            yolo_annotations.append(f"{category_id} {' '.join([str(x) for x in bbox])}")

        return '\n'.join(yolo_annotations)

    def create_cross_validation_yolo(meta_data_df, page_df, local_image_dir, output_dir, n_folds=5, random_state=42):
        """
        Create and save x-fold cross-validation YOLO format files
        """
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)

        # Get unique page IDs
        unique_pages = page_df['page_id'].unique()

        # Initialize K-fold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # Create data.yaml file
        yaml_content = {
            'path': output_dir,
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,  # number of classes
            'names': ['article', 'advert', 'image']
        }

        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            yaml.dump(yaml_content, f)

        # Perform k-fold split
        for fold, (train_idx, val_idx) in enumerate(kf.split(unique_pages)):
            # Create fold-specific directories
            fold_dir = os.path.join(output_dir, f'fold_{fold}')
            for split in ['train', 'val']:
                os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
                os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)

            # Get train and validation page IDs
            train_pages = unique_pages[train_idx]
            val_pages = unique_pages[val_idx]

            # Process train and validation sets
            for split, pages in [('train', train_pages), ('val', val_pages)]:
                split_page_df = page_df[page_df['page_id'].isin(pages)]

                for _, row in split_page_df.iterrows():
                    # Get image information
                    img_path = os.path.join(local_image_dir, row['filename'])
                    with Image.open(img_path) as img:
                        width, height = img.size

                    # Get annotations for this image
                    img_annotations = meta_data_df[meta_data_df['page_id'] == row['page_id']].to_dict('records')

                    # Convert annotations to YOLO format
                    yolo_content = create_yolo_annotation(img_path, img_annotations, width, height)

                    # Create paths for new image and label files
                    new_img_path = os.path.join(fold_dir, 'images', split, row['filename'])
                    label_filename = os.path.splitext(row['filename'])[0] + '.txt'
                    label_path = os.path.join(fold_dir, 'labels', split, label_filename)

                    # Copy image to new location
                    shutil.copy2(img_path, new_img_path)

                    # Save YOLO format annotations
                    with open(label_path, 'w') as f:
                        f.write(yolo_content)

            # Create fold-specific data.yaml
            fold_yaml_content = {
                'path': fold_dir,
                'train': 'images/train',
                'val': 'images/val',
                'nc': 3,  # number of classes
                'names': ['article', 'advert', 'image']
            }

            with open(os.path.join(fold_dir, 'data.yaml'), 'w') as f:
                yaml.dump(fold_yaml_content, f)

            print(f"Fold {fold + 1}/{n_folds} completed")
            print(f"Train set size: {len(train_pages)} pages")
            print(f"Validation set size: {len(val_pages)} pages")
            print("------------------------")
    return (
        convert_bbox_to_yolo,
        create_cross_validation_yolo,
        create_yolo_annotation,
    )


@app.cell
def __(mo):
    mo.md(
        """
        create_cross_validation_yolo(meta_data_df = test_file_meta_data.loc[test_file_meta_data['publication_id']==24], 
                              page_df= file_name_to_id_map.loc[file_name_to_id_map['publication_id']==24], 
                              local_image_dir = os.path.join(converted_folder, periodical_folders[0]),
                            output_dir = os.path.join(converted_folder, 'EWJ_yolo'), 
                            n_folds=5)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
