import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __():
    import os 
    from helper_functions import files_to_df_func, scale_bbox, check_bboxes_valid, calculate_coverage_and_overlap
    import pandas as pd
    from tqdm import tqdm
    import shutil
    from pdf2image import convert_from_path
    import json
    from PIL import Image
    from sklearn.model_selection import KFold
    import seaborn as sns
    import shutil
    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
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

    #one of these is missing! need to check which one
    ground_truth_page_id = [ 97484, 108039, 111597, 124384,  93238,  91046,  90939,  91483,  92066,  92372,
      92551, 164855, 124476, 110454, 120252, 122756, 123314, 123497, 128022,  79515,
     163094, 138047, 140098, 146451, 150954, 160259, 152432, 155298, 160323, 160759, 160813]
    return (
        Image,
        KFold,
        calculate_coverage_and_overlap,
        check_bboxes_valid,
        convert_from_path,
        converted_folder,
        data_folder,
        files_to_df_func,
        ground_truth_page_id,
        json,
        np,
        os,
        pd,
        periodical_folders,
        plt,
        scale_bbox,
        shutil,
        sns,
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
def __(data_folder, os, pd):
    periodical_df = pd.read_parquet(os.path.join(data_folder,'periodicals_publication.parquet'))
    return (periodical_df,)


@app.cell
def __(periodical_df):
    periodical_df
    return


@app.cell
def __(ast, check_bboxes_valid, data_folder, os, pd, periodical_df, tqdm):
    _file_name = os.path.join(data_folder, 'ncse_data_metafile.parquet')

    # We need to add the page width and height to be able to scale to bounding boxes properly
    _page_data = pd.read_parquet(os.path.join(data_folder,'periodicals_page.parquet'))

    issue_data = pd.read_parquet(os.path.join(data_folder,'periodicals_issue.parquet'))
    issue_data['periodical_abbrev'] = issue_data['pdf'].str.split('-').str[1]
    issue_data['filename'] = issue_data['periodical_abbrev'] + '_' + issue_data['issue_date'].astype(str) + '.pdf'


    if not os.path.isfile(_file_name):
        _df = []
        file_folder ='new_parquet' #'ncse_text_chunks' # 'new_parquet'
        for file in tqdm(os.listdir(os.path.join(data_folder, file_folder))):

            _temp = pd.read_parquet(os.path.join(data_folder, file_folder, file))
            #load all necessary columns
            _temp = _temp.loc[:,['id', 'article_type_id', 'issue_id', 'page_id', 
                                 'issue_date', 'page_number', 'bounding_box']]

            #temp.drop(columns='content_html', inplace=True)
            _df.append(_temp)
            #delete as I feel there is some memory leak going on
            del _temp

        _df  = pd.concat(_df, ignore_index=True)

        _df['issue_date'] = _df['issue_date'].astype(str)

        _df = _df.merge(_page_data[['id', 'height', 'width']].set_index('id'), 
                                                        left_on = 'page_id', right_index= True)

        #add in the periodical id so that the file names can be re-constructed
        _df = _df.merge(issue_data[['id', 'periodical_abbrev', 'publication_id']].set_index('id'),
                                                       left_on = 'issue_id', right_index= True)
        _df = _df.merge(periodical_df[['id', 'abbreviation']].set_index('id'), left_on='publication_id', right_index=True)

        _df['bounding_box'] = _df.apply(
            lambda row: ast.literal_eval(row['bounding_box']) 
            if isinstance(row['bounding_box'], str) 
            else row['bounding_box'], 
            axis=1
        )

        bbox_df = pd.json_normalize(_df['bounding_box'])
        _df = pd.concat([_df, bbox_df], axis=1)

        _df['x0'] = _df['x0'].astype(int)
        _df['x1'] = _df['x1'].astype(int)
        _df['y0'] = _df['y0'].astype(int)
        _df['y1'] = _df['y1'].astype(int)

        # Area = (x1 - x0) * (y1 - y0)
        _df['page_area'] = _df['height'] * _df['width']
        _df['bounding_box_area'] = (_df['x1'] -_df['x0']) * (_df['y1'] -_df['y0']) 
        _df['page_fract'] = _df['bounding_box_area'] / _df['page_area'] # This is useful for checking nothing stupid happening
        #... because lots of stupid is happening

        _df['valid_bbox'] = check_bboxes_valid(_df, 'width', 'height')

        _df.to_parquet(_file_name)
        bbox_data_df = _df
    else:
        bbox_data_df = pd.read_parquet(_file_name)
    return bbox_data_df, bbox_df, file, file_folder, issue_data


@app.cell
def __(bbox_data_df):
    bbox_data_df.groupby(['article_type_id', 'abbreviation']).size().reset_index()
    return


@app.cell
def __(bbox_data_df):
    bbox_data_df.groupby('page_id').agg(print_left=('x0', 'min'), print_top=('y0', 'min') , print_right=('x1', 'max'), print_bottom=('y1', 'max'))
    return


@app.cell
def __(mo):
    mo.md(r"""## Create a dataframe for the stored page images""")
    return


@app.cell
def __(os, pd):
    converted_page_info = pd.read_parquet('data/page_size_info.parquet')

    converted_page_info['filename'] = converted_page_info['output_file'].apply(os.path.basename)
    return (converted_page_info,)


@app.cell
def __(converted_page_info):
    converted_page_info
    return


@app.cell
def __(mo):
    mo.md(r"""#""")
    return


@app.cell
def __(
    bbox_data_df,
    calculate_coverage_and_overlap,
    converted_page_info,
    ground_truth_page_id,
    os,
    pd,
    periodical_df,
):
    _target_file = 'data/file_name_to_id_map.parquet'

    if not os.path.exists('data/bbox_overlap.parquet'):
        print('calculating bounding box percent coverage for each image')  
        _coverage_results = calculate_coverage_and_overlap(bbox_data_df)
        _coverage_results.to_parquet('data/bbox_overlap.parquet')
    else:
        _coverage_results = pd.read_parquet('data/bbox_overlap.parquet')


    if not os.path.exists(_target_file):

    #    _image_file_list = []
    #    for _folder in tqdm(periodical_folders):
    #        _image_file_list = _image_file_list + os.listdir(os.path.join(converted_folder, _folder))

        stored_image_files_df = converted_page_info.copy()#pd.DataFrame({'filename':_image_file_list})

        stored_image_files_df['periodical_abbrev'] = stored_image_files_df['filename'].str.split('-').str[0]
        stored_image_files_df['issue_date'] = stored_image_files_df['filename'].str.extract(r'-(\d{4}-\d{2}-\d{2})')
        stored_image_files_df['page_number'] = stored_image_files_df['filename'].str.extract(r'page_(\d+)').astype(int)
    #
        _merge_columns = ['periodical_abbrev', 'issue_date', 'page_number']

        # First, calculate area sum grouped by the relevant columns
        _area_sums = bbox_data_df.groupby(['page_id'])['bounding_box_area'].sum().reset_index()
        _valid_bbox = bbox_data_df.groupby(['page_id'])['valid_bbox'].all().reset_index()
        _overlap_fract = bbox_data_df.groupby(['page_id'])['overlap_fract'].max()

        # Then, perform the drop duplicates and merge operation without the area column
        _base_data = bbox_data_df[['issue_id', 'page_id', 'publication_id', 'width', 'height'] + _merge_columns].drop_duplicates()

        # Merge the area sums back
        _base_data = _base_data.merge(_area_sums, on=[ 'page_id'])
        _base_data = _base_data.merge(_valid_bbox, on=[ 'page_id'])
        _base_data = _base_data.merge(_overlap_fract, on=[ 'page_id'])

        # Finally, perform your original merge with stored_image_files_df
        file_name_to_id_map = _base_data.merge(
            stored_image_files_df.set_index(_merge_columns),
            left_on=_merge_columns, 
            right_index=True
        )

        #calculating percent cover of image                                        
        file_name_to_id_map = file_name_to_id_map.merge(_coverage_results, on = 'page_id')

        file_name_to_id_map = file_name_to_id_map.merge(periodical_df[['id', 'abbreviation']].set_index('id'), left_on='publication_id', right_index=True)

        file_name_to_id_map['gt_set'] = file_name_to_id_map['page_id'].isin(ground_truth_page_id)

        file_name_to_id_map.to_parquet(_target_file)
    else:
        file_name_to_id_map = pd.read_parquet(_target_file)
    return file_name_to_id_map, stored_image_files_df


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map
    return


@app.cell
def __():
    1275/1193
    return


@app.cell
def __(mo):
    mo.md(
        """
        # Understanding Overlap and invalid boxes

        There are a a substantial number of bounding boxes with substantial overlap as well as images where the bounding boxes extend beyond the image.

        This suggests that there may be systematic errors in the way that the bounding boxes are sized relative to the reported original image size.

        I need to try to understand what scaling or mislocation errors there are whether they are systematic or not and how to correct them.

        The below figure shows that there is a substantial difference between the total image cover of the bounding boxes and the total area of the bounding boxes as a percent of the image. Some images have over 600% coverage
        """
    )
    return


@app.cell
def __(file_name_to_id_map, sns):
    sns.kdeplot(data = file_name_to_id_map, x = 'total_coverage_percent', hue = 'abbreviation')
    return


@app.cell
def __(file_name_to_id_map, plt, sns):
    sns.kdeplot(data = file_name_to_id_map, x = 'text_overlap_percent', hue = 'abbreviation')
    plt.ylim(0,10)
    plt.show()
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map.loc[(file_name_to_id_map['abbreviation']=='MRUC') & (file_name_to_id_map['text_overlap_percent']>0.9)]
    return


@app.cell
def __(bbox_data_df):
    bbox_data_df.loc[bbox_data_df['page_id']==96401]
    return


@app.cell
def __(file_name_to_id_map):
    (file_name_to_id_map.groupby('abbreviation')[['text_overlap_percent']].apply(lambda x: (x > 0.05).astype(int).mean()))
    return


@app.cell
def __(file_name_to_id_map):
    (file_name_to_id_map.groupby('abbreviation')[['total_coverage_percent']].apply(lambda x: (x > 0.95).astype(int).mean()))
    return


@app.cell
def __(file_name_to_id_map):
    # Original overlap calculation
    check_overlap = (file_name_to_id_map.groupby(['periodical_abbrev', 'publication_id'])
                    ['overlap_fract'].apply(lambda x: (x > 0.1).astype(int).mean())).reset_index()

    # Add count calculation
    counts = file_name_to_id_map.groupby(['periodical_abbrev', 'publication_id']).size().reset_index(name='count')

    # Merge the two
    check_overlap = check_overlap.merge(counts, on=['periodical_abbrev', 'publication_id'])

    check_overlap.loc[check_overlap['publication_id']==20]
    return check_overlap, counts


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Invalid boxes

        What we can see is that the vast majority of invalid boxes are concentrated in the Northern Star periodical, and within that the scan groups NS2, NS3, and NS4.
        """
    )
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map.groupby(['abbreviation', 'publication_id'])['valid_bbox'].mean()
    return


@app.cell
def __(file_name_to_id_map):
    check_validity = file_name_to_id_map.groupby(['periodical_abbrev', 'publication_id']).agg({
        'valid_bbox': ['mean', 'count']
    }).reset_index()

    check_validity.loc[check_validity['publication_id']==27]
    return (check_validity,)


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
def __(bbox_data_df):
    bbox_data_df.loc[bbox_data_df['id']==503326]
    return


@app.cell
def __(mo):
    mo.md(
        """
        create_cross_validation_yolo(meta_data_df = bbox_data_df.loc[bbox_data_df['publication_id']==24], 
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
