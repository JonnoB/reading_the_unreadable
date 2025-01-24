import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    from bbox_functions import plot_boxes_on_image
    import os 


    data = os.path.join('data', "download_jobs/EWJ.parquet")

    data_path = os.path.join('data', "download_jobs", "ncse")

    all_data_files = [file for file in os.listdir(data_path) if '.parquet' in file ]

    periodical_mapping = pd.DataFrame({'periodical':['TEC', 'FTEC', 'TTW', 'ATTW', 'ETTW', 'FTTW', 'EWJ', 'FEWJ',
           'EMRP', 'FMRP', 'MRP', 'SMRP', 'CLD', 'FLDR', 'LDR', 'NSS', 'NS2',
           'NS8', 'NS4', 'NS3', 'NS5', 'NS7', 'NS6', 'NS9', 'SNSS'], 
    'periodical_code':['TEC', 'TEC', 'TTW','TTW','TTW','TTW', 'EWJ', 'EWJ','MRP', 'MRP','MRP','MRP', 'CLD', 'CLD','CLD', 'NS', 'NS','NS','NS','NS','NS','NS','NS','NS','NS', ]}
    )

    path_mapping = {
        'CLD': '/media/jonno/ncse/converted/all_files_png_120/Leader_issue_PDF_files',
        'EWJ': '/media/jonno/ncse/converted/all_files_png_120/English_Womans_Journal_issue_PDF_files',
        'MRP': '/media/jonno/ncse/converted/all_files_png_120/Monthly_Repository_issue_PDF_files',
        'TTW': '/media/jonno/ncse/converted/all_files_png_120/Tomahawk_issue_PDF_files',
        'TEC': '/media/jonno/ncse/converted/all_files_png_120/Publishers_Circular_issue_PDF_files',
        'NS': '/media/jonno/ncse/converted/all_files_png_200/Northern_Star_issue_PDF_files'
    }

    raw_bbox_path = "data/periodical_bboxes/post_process_raw"
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
        os,
        path_mapping,
        pd,
        periodical_mapping,
        plot_boxes_on_image,
        raw_bbox_path,
        raw_bboxes_df,
    )


@app.cell
def _(all_data_files, data_path, os, pd):
    test2 = []

    for _file in all_data_files:

        test2.append(pd.read_parquet(os.path.join(data_path, _file)))

    test2 = pd.concat(test2, ignore_index=True)
    return (test2,)


@app.cell
def _(test2):
    test2.groupby('class').size()
    return


@app.cell
def _(test2):
    test2['page_id'].unique()
    return


@app.cell
def _(
    bboxes_df,
    os,
    path_mapping,
    periodical_mapping,
    plot_boxes_on_image,
    test2,
):
    target_page =test2['page_id'].unique()[17]

    periodical_code = periodical_mapping.loc[periodical_mapping['periodical']==target_page.split("-")[0], 'periodical_code'].iloc[0]
    #  Identify image folder using the periodical ID
    image_path = path_mapping[periodical_code]

    plot_boxes_on_image(bboxes_df[bboxes_df['page_id']==target_page], 
                        image_path = os.path.join(image_path, target_page+'.png'), show_reading_order=True)
    return image_path, periodical_code, target_page


@app.cell
def _(image_path, os, plot_boxes_on_image, raw_bboxes_df, target_page):

    plot_boxes_on_image(raw_bboxes_df[raw_bboxes_df['page_id']==target_page], 
                        image_path = os.path.join(image_path, target_page+'.png'), show_reading_order=True)

    return


@app.cell
def _(raw_bboxes_df, target_page):
    raw_bboxes_df[raw_bboxes_df['page_id']==target_page]
    return


@app.cell
def _(raw_bboxes_df, test2):
    raw_bboxes_df[raw_bboxes_df['page_id'].isin(test2['page_id'].unique())].groupby('class').size()
    return


@app.cell
def _(bboxes_df, test2):
    bboxes_df[bboxes_df['page_id'].isin(test2['page_id'].unique())].groupby('class').size()
    return


if __name__ == "__main__":
    app.run()
