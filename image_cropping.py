import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import os
    from helper_functions import create_page_dict, scale_bbox

    from pdf2image import convert_from_path
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import io
    import json

    image_drive = '/media/jonno/ncse'
    return (
        Image,
        ImageDraw,
        convert_from_path,
        create_page_dict,
        image_drive,
        io,
        json,
        mo,
        os,
        pd,
        plt,
        scale_bbox,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # General things

        The images are a total 48gb zipped  and 60Gb unzipped.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("""# Create dataframe with bounding box""")
    return


@app.cell
def __(pd):
    page_data = pd.read_parquet('data/periodicals_page.parquet')
    return (page_data,)


@app.cell
def __(os, page_data, pd):
    all_bounding_boxes = []
    for _file in os.listdir("data/new_parquet"):
        _temp = pd.read_parquet(os.path.join("data/new_parquet",_file))
        all_bounding_boxes.append(_temp.loc[:, ['id', 'bounding_box', 'article_type_id', 'issue_id', 'page_id', 'publication_id', 'page_number', 'pdf']])

    all_bounding_boxes = pd.concat(all_bounding_boxes, ignore_index=True)

    all_bounding_boxes['bounding_box'] = all_bounding_boxes['bounding_box'].apply(lambda box: {k: int(v) for k, v in box.items()})

    all_bounding_boxes['file_name']  = all_bounding_boxes['pdf'].str.extract('(\w{3}-\d{4}-\d{2}-\d{2})')

    all_bounding_boxes['file_name'] = all_bounding_boxes['file_name'] +'.pdf'

    all_bounding_boxes['date'] = pd.to_datetime(all_bounding_boxes['file_name'].str.extract(r'-(\d{4}-\d{2}-\d{2})\.pdf')[0], format='%Y-%m-%d')

    #merge in the height and width data
    all_bounding_boxes = all_bounding_boxes.merge(
        page_data[['id', 'height', 'width']].set_index('id'),
        left_on='page_id',
        right_index=True
    )
    return (all_bounding_boxes,)


@app.cell
def __():
    return


@app.cell
def __(all_bounding_boxes):
    len(all_bounding_boxes['issue_id'].unique())
    return


@app.cell
def __(all_bounding_boxes):
    all_bounding_boxes
    return


@app.cell
def __(mo):
    mo.md(r"""# Create bounding box dictionary""")
    return


@app.cell
def __(all_bounding_boxes, create_page_dict, json):
    # Create the dictionary
    page_dict = create_page_dict(all_bounding_boxes)

    output_file = 'data/page_dict.json'
    with open(output_file, 'w') as f:
        json.dump(page_dict, f, indent=4)
    return f, output_file, page_dict


@app.cell
def __(page_dict, pd):
    lengths = pd.DataFrame({'len':[len(value) for key, value in page_dict.items()]})

    lengths.describe()
    return (lengths,)


@app.cell
def __(page_dict):
    page_dict.keys()
    return


@app.cell
def __(all_bounding_boxes):
    page_issue_df = all_bounding_boxes.groupby(['page_id', 'page_number','issue_id', 'pdf']).size().reset_index(name = 'counts')
    return (page_issue_df,)


@app.cell
def __(pd):
    #Add the folder paths to the periodicals dataframe, this will allow us to construct paths to the PDF's

    periodicals = pd.read_parquet('data/periodicals_publication.parquet')

    periodicals['folder_path'] = [
     'Northern_Star_issue_PDF_files',
    'Leader_issue_PDF_files',
    'Tomahawk_issue_PDF_files',
    'Publishers_Circular_issue_PDF_files',
        'English_Womans_Journal_issue_PDF_files',
    'Monthly_Repository_issue_PDF_files']


    periodicals
    return (periodicals,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Create subset

        There is too much data. I am creating a subset of only the overlapping periods to allow a compare and contrast
        """
    )
    return


@app.cell
def __(all_bounding_boxes, periodicals):
    mask_1850_1852 = all_bounding_boxes['date'].between('1850-01-01', '1852-12-31')

    # Create a mask for dates between 1858-1860 (inclusive)
    mask_1858_1860 = all_bounding_boxes['date'].between('1858-01-01', '1860-12-31')

    # Combine the masks using the OR operator
    combined_mask = mask_1850_1852 | mask_1858_1860

    # Combine the masks using the OR operator
    combined_mask = mask_1858_1860


    # Apply the combined mask to the dataframe
    subset_df = all_bounding_boxes[combined_mask]

    subset_df = subset_df.merge(periodicals[['id','folder_path']].set_index('id'), 
                                left_on='publication_id', right_index = True)

    print(f"Rows in dataset:{len(subset_df['page_id'].unique())}")
    subset_df

    subset_df.to_parquet('data/example_set_1858-1860.parquet')
    return combined_mask, mask_1850_1852, mask_1858_1860, subset_df


@app.cell
def __(subset_df):
    subset_df
    return


@app.cell
def __(subset_df):
    target_pages_isues = subset_df.copy().loc[:, 
    ['issue_id', 'page_id', 'page_number', 'file_name', 'folder_path', 'width', 'height']].drop_duplicates().reset_index(drop=True)


    print(f"Number of issues to extract {len(target_pages_isues[['issue_id']].drop_duplicates())}, number of pages {len(target_pages_isues[['page_id']].drop_duplicates())},")
    return (target_pages_isues,)


@app.cell
def __(mo):
    mo.md(r"""# Show Single page""")
    return


@app.cell
def __(convert_from_path, image_drive, os, target_pages_isues):
    check_row_df = target_pages_isues.loc[300, :]

    _file = os.path.join(image_drive, check_row_df['folder_path'], check_row_df['file_name'])

    all_pages = convert_from_path(_file, dpi = 300)
    return all_pages, check_row_df


@app.cell
def __(
    Image,
    ImageDraw,
    all_pages,
    check_row_df,
    io,
    page_dict,
    plt,
    scale_bbox,
):
    _page = all_pages[check_row_df['page_number']-1].copy()
    _draw = ImageDraw.Draw(_page)

    # Your bounding box dictionary
    _bounding_boxes = page_dict[check_row_df['page_id']]


    # Draw rectangles for each bounding box
    for _box_id, _coords in _bounding_boxes.items():
        _x0, _y0, _x1, _y1 = scale_bbox([_coords["x0"], _coords["y0"], _coords["x1"], _coords["y1"]],
                                         (check_row_df['width'], check_row_df['height']), _page.size)
        _draw.rectangle([_x0, _y0, _x1, _y1], outline="red", width=2)

    _buf = io.BytesIO()
    _page.save(_buf, format='PNG')
    _buf.seek(0)

    # Display the image in the notebook
    plt.figure(figsize=(15, 20))
    plt.imshow(Image.open(_buf))
    plt.axis('off')
    plt.show()

    _bounding_boxes
    return


if __name__ == "__main__":
    app.run()
