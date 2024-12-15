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


        How many images have low levels of overlap and high levels of covereage?

        Images with high levels of overlap and boxes that have high levels of covereage are good targets for fixing.

        If a model shows a drop in overlap and an equal measure of coverage. This means it is working better than the original dataset.

        Train two models.

        - Model 1 uses all the data
        - Model 2 uses only low overlap, high coverage data.

        Predict on test set, see if metrics improve.
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
    import seaborn as sns
    import random
    converted_folder = '/media/jonno/ncse/converted/all_files_png_72'

    from helper_functions_plotting import plot_multiple_images_with_boxes, plot_single_image_with_boxes


    bbox_data_df = pd.read_parquet(os.path.join('data', 'ncse_data_metafile.parquet'))
    #page_conversion_df = pd.read_parquet(os.path.join(converted_folder, 'page_size_info.parquet'))
    #page_conversion_df['filename'] = page_conversion_df['output_file'].apply(os.path.basename)

    file_name_to_id_map = pd.read_parquet('data/file_name_to_id_map.parquet')#.merge(page_conversion_df, on = 'filename')

    file_name_to_id_map['page_area'] = file_name_to_id_map['width'] * file_name_to_id_map['width']

    file_name_to_id_map['page_coverage_percent'] = file_name_to_id_map['total_covered_pixels']/file_name_to_id_map['page_area'] 

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
        patches,
        pd,
        periodical_folders,
        plot_multiple_images_with_boxes,
        plot_single_image_with_boxes,
        plt,
        random,
        scale_bbox,
        sns,
    )


@app.cell
def __(bbox_data_df):
    bbox_data_df.loc[bbox_data_df['page_id'].isin([97044])]
    return


@app.cell
def __():
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map.loc[file_name_to_id_map['page_id']==97044]
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map.loc[file_name_to_id_map['abbreviation']=='NS'].groupby('valid_bbox').first()

    #file_name_to_id_map.loc[file_name_to_id_map['periodical_abbrev']== 'NS3']
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map
    return


@app.cell
def __(file_name_to_id_map, sns):
    sns.kdeplot(data = file_name_to_id_map.loc[file_name_to_id_map['abbreviation']=='L'], x = 'page_width_pt')
    return


@app.cell
def __(file_name_to_id_map, sns):
    sns.kdeplot(data = file_name_to_id_map.loc[(file_name_to_id_map['abbreviation']=='L')  ], x = 'page_width_pt', hue = 'periodical_abbrev')
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map.loc[file_name_to_id_map['abbreviation']=='EWJ']
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map.groupby('abbreviation')['page_coverage_percent'].describe()
    return


@app.cell
def __(file_name_to_id_map):
    (file_name_to_id_map.groupby('abbreviation')['valid_bbox'].apply(lambda x: (x).astype(int).mean()))
    return


@app.cell
def __(file_name_to_id_map):
    (file_name_to_id_map.groupby('abbreviation')['page_coverage_percent'].apply(lambda x: ((x > 0.95) & (x<1.01)).astype(int).mean()))
    return


@app.cell
def __(file_name_to_id_map):
    (file_name_to_id_map.groupby('abbreviation')['valid_bbox'].apply(lambda x: (x).astype(int).mean()))
    return


@app.cell
def __(file_name_to_id_map):
    (file_name_to_id_map.groupby('abbreviation').apply(
        lambda x: ((x['valid_bbox']) & (x['text_overlap_percent'] < 0.1)).astype(int).mean()
    ))
    return


@app.cell
def __(bbox_data_df, file_name_to_id_map):
    page_id = 100886


    #167499 NS3 (1307*1.805, 2197*1.805)
    #168959 NS4
    165363

    bboxes = bbox_data_df.loc[bbox_data_df['page_id']==page_id]#.iloc[[4,1]] #2,7

    pub_id = bboxes['publication_id'].unique()[0]


    size_df = file_name_to_id_map.loc[file_name_to_id_map['page_id']==page_id].filter(regex = 'width|height|coverage')
    print(size_df)

    bboxes
    return bboxes, page_id, pub_id, size_df


@app.cell
def __():
    (1307*1.805, 2197*1.805)
    return


@app.cell
def __(bbox_data_df, file_name_to_id_map, plot_single_image_with_boxes):
    # Get a single page_id
    _page_id = 164588 # or use any specific page_id

    # Get the image path and boxes for this page
    _image_path = file_name_to_id_map[file_name_to_id_map['page_id'] == _page_id]['output_file'].iloc[0]
    _page_boxes = bbox_data_df[bbox_data_df['page_id'] == _page_id]

    # Plot the single image with boxes
    plot_single_image_with_boxes(
        image_path=_image_path,
        boxes_df=_page_boxes,
        scale_factor_x=1.064,
        scale_factor_y=1.064,
        title=f'Page ID: {_page_id}'
    )
    return


@app.cell
def __(bbox_data_df, file_name_to_id_map, plot_single_image_with_boxes):

    # Get a single page_id
    _page_id = 96348 # or use any specific page_id

    # Get the image path and boxes for this page
    _image_path = file_name_to_id_map[file_name_to_id_map['page_id'] == _page_id]['output_file'].iloc[0]
    _page_boxes = bbox_data_df[bbox_data_df['page_id'] == _page_id]

    # Plot the single image with boxes
    plot_single_image_with_boxes(
        image_path=_image_path,
        boxes_df=_page_boxes,
        scale_factor_x=1.064,
        scale_factor_y=1.064,
        title=f'Page ID: {_page_id}'
    )

    return


@app.cell
def __(bbox_data_df, file_name_to_id_map, plot_multiple_images_with_boxes):
    file_name_to_id_map2 = file_name_to_id_map.loc[file_name_to_id_map['periodical_abbrev']=='LDR']

    plot_multiple_images_with_boxes(
        file_name_to_id_map2,
        bbox_data_df,
        file_name_to_id_map2['page_id'].sample(6).to_list(),  # List of 6 page IDs
        figsize=(10, 15),  # Larger figure size to accommodate multiple plots
        box_color='red',
        box_linewidth=2,
        padding_color='white',
        scale_factor_x = 1.064,
        scale_factor_y = 1.064,
        save_path=None
    )

    return (file_name_to_id_map2,)


@app.cell
def __(
    bbox_data_df,
    file_name_to_id_map,
    np,
    plot_multiple_images_with_boxes,
):
    bbox_data_df2 = bbox_data_df.loc[(bbox_data_df['page_fract']>0.5) & 
    bbox_data_df['page_id'].isin(file_name_to_id_map.loc[(file_name_to_id_map['text_overlap_percent'] >0.15), 'page_id'])]
    file_name_to_id_map['page_id'].sample(6).tolist()

    plot_multiple_images_with_boxes(
        file_name_to_id_map,
        bbox_data_df2,
        np.random.choice(bbox_data_df2['page_id'].unique(), size=6, replace=False),  # List of 6 page IDs
        figsize=(10, 15),  # Larger figure size to accommodate multiple plots
        box_color='red',
        box_linewidth=2,
        padding_color='white',
        scale_factor_x = 1.064,
        scale_factor_y = 1.064,
        save_path=None
    )


    return (bbox_data_df2,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Problem pages
        Leader
         - 103122
         - 137464
         - -137504 looks like a dup
        """
    )
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map[file_name_to_id_map['page_id'] == 137464]
    return


@app.cell
def __():
    (801/1536, 1051/2016 )
    return


@app.cell
def __(file_name_to_id_map):
    file_name_to_id_map['periodical_abbrev'].unique()
    return


@app.cell
def __(bbox_data_df, file_name_to_id_map, plot_single_image_with_boxes):

    # Get a single page_id
    _page_id = 103122 # or use any specific page_id

    # Get the image path and boxes for this page
    _image_path = file_name_to_id_map[file_name_to_id_map['page_id'] == _page_id]['output_file'].iloc[0]
    _page_boxes = bbox_data_df[bbox_data_df['page_id'] == _page_id]

    # Plot the single image with boxes
    plot_single_image_with_boxes(
        image_path=_image_path,
        boxes_df=_page_boxes,
        scale_factor_x=0.89,
        scale_factor_y=0.89,
        box_color='red',
        box_linewidth=1,
        padding_color='white',
        padding=20,
        title=f'Page ID: {_page_id}'
    )

    return


@app.cell
def __(bbox_data_df, file_name_to_id_map, plot_single_image_with_boxes):

    # Get a single page_id
    _page_id = 83320 # or use any specific page_id

    # Get the image path and boxes for this page
    _image_path = file_name_to_id_map[file_name_to_id_map['page_id'] == _page_id]['output_file'].iloc[0]
    _page_boxes = bbox_data_df[bbox_data_df['page_id'] == _page_id]

    # Plot the single image with boxes
    plot_single_image_with_boxes(
        image_path=_image_path,
        boxes_df=_page_boxes,
        scale_factor_x=1.064,
        scale_factor_y=1.064,
        box_color='red',
        box_linewidth=1,
        padding_color='white',
        padding=20,
        title=f'Page ID: {_page_id}'
    )

    return


if __name__ == "__main__":
    app.run()
