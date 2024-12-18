import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        """
        # Preparing the NCSE test set

        The original NCSE test set images were only used for transcription and were not part of the overall analysis. Now that the images are being used it is clear they need some further processing. 

        This note book ensures that there is a clear link between the original pages used to create the test set and the bounding boxes, cropped images, and ground truth text used in this work.
        """
    )
    return


@app.cell
def __():
    import os 
    from helper_functions import files_to_df_func
    import pandas as pd
    from tqdm import tqdm
    import shutil
    from pdf2image import convert_from_path
    import json

    data_folder = 'data'

    transcript_files = os.path.join(data_folder, 'transcripts/transcription_files')
    test_pdf_path = os.path.join(data_folder, 'ncse_test_pdf_date_names')

    test_pdf_path_correct_names = os.path.join(data_folder, 'ncse_test_pdf')
    test_jpg_path = os.path.join(data_folder, 'ncse_test_jpg')
    return (
        convert_from_path,
        data_folder,
        files_to_df_func,
        json,
        os,
        pd,
        shutil,
        test_jpg_path,
        test_pdf_path,
        test_pdf_path_correct_names,
        tqdm,
        transcript_files,
    )


@app.cell
def __(files_to_df_func, transcript_files):
    ncse_test_transcripts = files_to_df_func(transcript_files)
    ncse_test_transcripts['artid'] = ncse_test_transcripts['artid'].astype(int)

    #ncse_test_transcripts['issue_date'] = ncse_test_transcripts['issue_date'].astype(str)
    return (ncse_test_transcripts,)


@app.cell
def __(data_folder, ncse_test_transcripts, os, pd, tqdm):
    test_file_meta_data = []
    file_folder ='new_parquet' #'ncse_text_chunks' # 'new_parquet'
    for file in tqdm(os.listdir(os.path.join(data_folder, file_folder))):

        temp = pd.read_parquet(os.path.join(data_folder, file_folder, file))
        temp = temp.loc[temp['id'].isin(ncse_test_transcripts['artid']),['id', 'article_type_id', 'issue_id', 
                                                                         'page_id', 'issue_date', 'page_number', 'bounding_box']]
        #temp.drop(columns='content_html', inplace=True)
        test_file_meta_data.append(temp)
        del temp

    test_file_meta_data  = pd.concat(test_file_meta_data, ignore_index=True)

    test_file_meta_data['issue_date'] = test_file_meta_data['issue_date'].astype(str)


    # We need to add the page width and height to be able to scale to bounding boxes properly
    page_data = pd.read_parquet('data/periodicals_page.parquet')

    test_file_meta_data = test_file_meta_data.merge(page_data[['id', 'height', 'width']].set_index('id'), left_on = 'page_id', right_index= True)

    test_file_meta_data.to_csv('data/ncse_test_data_metafile.csv')

    del page_data
    return file, file_folder, page_data, temp, test_file_meta_data


@app.cell
def __(test_file_meta_data):
    test_file_meta_data
    return


@app.cell
def __(os, pd, test_pdf_path):
    #cell renames from the original file names which were just an abbreviation for the periodical and date of publication, to something more useful.
    test_files = pd.DataFrame({'filename': os.listdir(test_pdf_path)})
    test_files[['PUB_abrev', 'issue_date']] = test_files['filename'].str.split('-', n = 1, expand=True)
    test_files['issue_date'] = test_files['issue_date'].str.removesuffix('.pdf').astype(str) #can be converted to actual date if necessary
    return (test_files,)


@app.cell
def __(test_file_meta_data):
    test_file_meta_data
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Rename file

        The files are named only after the publication and date, this makes it difficult to locate the correct meta data, the below renames with the page id and other details for both computational and human readable search
        """
    )
    return


@app.cell
def __(test_file_meta_data, test_files):
    new_file_names = test_file_meta_data[['issue_date', 'page_id', 'page_number']].drop_duplicates().merge(test_files, on = 'issue_date')

    new_file_names['new_filename'] = new_file_names.apply(lambda x: f"{x['PUB_abrev']}_pageid_{x['page_id']}_pagenum_{x['page_number']}_{x['issue_date']}.pdf", axis = 1)

    new_file_names
    return (new_file_names,)


@app.cell
def __(
    new_file_names,
    os,
    shutil,
    test_pdf_path,
    test_pdf_path_correct_names,
):
    def rename_files(df, source_folder, destination_folder):
        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Iterate through the DataFrame
        for index, row in df.iterrows():
            old_name = row['filename']
            new_name = row['new_filename']

            # Create full file paths
            old_path = os.path.join(source_folder, old_name)
            new_path = os.path.join(destination_folder, new_name)

            try:
                # Copy file to new location with new name
                shutil.copy2(old_path, new_path)
                print(f"Successfully renamed '{old_name}' to '{new_name}'")
            except Exception as e:
                print(f"Error processing {old_name}: {str(e)}")




    # Call the function
    rename_files(new_file_names, test_pdf_path, test_pdf_path_correct_names)
    return (rename_files,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Create jpgs

        Using the renamed files I will now create some files
        """
    )
    return


@app.cell
def __(convert_from_path, os, test_jpg_path, test_pdf_path_correct_names):
    def convert_pdf_to_jpg(input_folder, output_folder, dpi=200):
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get all PDF files in the input folder
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]

        for pdf_file in pdf_files:
            # Full path to PDF file
            pdf_path = os.path.join(input_folder, pdf_file)

            # Create output filename (change extension from .pdf to .jpg)
            output_filename = os.path.splitext(pdf_file)[0] + '.jpg'
            output_path = os.path.join(output_folder, output_filename)

            # Convert PDF to image
            images = convert_from_path(pdf_path, dpi=dpi)

            # Since we know it's single page, just take the first image
            images[0].save(output_path, 'JPEG')



    convert_pdf_to_jpg(test_pdf_path_correct_names, test_jpg_path, dpi=120)
    return (convert_pdf_to_jpg,)


@app.cell
def __(test_jpg_path):
    test_jpg_path
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Overlay bounding boxes on top of images

        I need to check how well the bounding boxes match the images. the below plots the bounding boxes on the images and saves them as a folder
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### Subset the bounding box json to only the test set and resave

        This is to ensure that the json for the test set is as small as possible
        """
    )
    return


@app.cell
def __(json):
    def filter_json_dict_by_column(json_dict, df, column_name):
        valid_keys = set(df[column_name].astype(str))
        return {k: v for k, v in json_dict.items() if k in valid_keys}

    with open('data/bounding_boxes.json') as json_data:
        bounding_boxes = json.load(json_data)
    return bounding_boxes, filter_json_dict_by_column, json_data


@app.cell
def __(bounding_boxes, filter_json_dict_by_column, test_file_meta_data):
    test_bounding_boxes = filter_json_dict_by_column(bounding_boxes, test_file_meta_data, 'page_id')
    return (test_bounding_boxes,)


@app.cell
def __(json, test_bounding_boxes):
    with open('data/test_ncse_bounding_boxes.json', 'w') as _file:
        json.dump(test_bounding_boxes, _file, indent = 4)
    return


@app.cell
def __(test_file_meta_data):
    test_file_meta_data
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
