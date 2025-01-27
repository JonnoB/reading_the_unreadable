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
    from function_modules.bbox_functions import calculate_coverage_and_overlap, plot_boxes_on_image

    from pathlib import Path

    # Change working directory to project root
    os.chdir(Path(__file__).parent)

    bbox_folder = 'data/periodical_bboxes/post_process'

    bbox_df = pd.concat([pd.read_parquet(os.path.join(bbox_folder, file)) for file in os.listdir(bbox_folder)])


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
        calculate_coverage_and_overlap,
        os,
        path_mapping,
        pd,
        periodical_mapping,
        plot_boxes_on_image,
    )


@app.cell
def _(bbox_df, calculate_coverage_and_overlap):
    post_process_coverage_df = calculate_coverage_and_overlap(bbox_df.loc[
                                                              bbox_df['filename'].isin(bbox_df['filename'].unique()[0:20])])

    # post_process_coverage_df = calculate_coverage_and_overlap(bbox_df)
    return (post_process_coverage_df,)


@app.cell
def _(post_process_coverage_df):
    post_process_coverage_df
    return


@app.cell
def _(bbox_df, os, path_mapping, periodical_mapping, plot_boxes_on_image):
    target_page = 'NS2-1838-01-13_page_8'


    _temp_bbox = bbox_df.loc[bbox_df['page_id']== target_page]

    periodical_code = periodical_mapping.loc[periodical_mapping['periodical']==target_page.split("-")[0], 'periodical_code'].iloc[0]

    plot_boxes_on_image(_temp_bbox, os.path.join(path_mapping[periodical_code],target_page+'.png' ))
    return periodical_code, target_page


@app.cell
def _(os, path_mapping, periodical_code):
    os.path.join(path_mapping[periodical_code])
    return


@app.cell
def _(os, path_mapping, periodical_code):
    os.path.isfile( os.path.join(path_mapping[periodical_code]))
    return


if __name__ == "__main__":
    app.run()
