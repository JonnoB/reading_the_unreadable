import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    from docling.document_converter import DocumentConverter
    import os
    import tqdm
    ncse_test_docs = 'data/ncse_test_jpg'
    return DocumentConverter, ncse_test_docs, os, tqdm


@app.cell
def __(DocumentConverter, ncse_test_docs, os):
    source = "https://arxiv.org/pdf/2408.09869"  # PDF path or URL
    source = os.path.join(ncse_test_docs, 'EWJ-1858-09-01_page_1.jpg')
    converter = DocumentConverter()
    result = converter.convert(source)
    print(result.document.export_to_markdown())  # output: "### Docling Technical Report[...]"
    return converter, result, source


@app.cell
def __(converter, os, tqdm):

    BLN_folder = 'data/BLN600/Images_jpg'

    save_folder = 'data/BLN_results/docling'

    os.makedirs(save_folder, exist_ok = True)

    for file in tqdm(os.listdir(BLN_folder)):

        _result = converter.convert(os.path.join(BLN_folder,file))

        text = _result.document.export_to_text()
    return BLN_folder, file, save_folder, text


@app.cell
def __(result):
    result.document.export_to_text()
    return


@app.cell
def __(source):
    from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
    from PIL import Image

    artifacts_path = '.venv/lib/python3.12/site-packages/docling_ibm_models'
    # Initialize the predictor
    predictor = LayoutPredictor(artifacts_path)

    # Load your image
    image = Image.open(source)

    # Get predictions
    predictions = predictor.predict(image)

    # Extract bounding boxes
    bounding_boxes = []
    for pred in predictions:
        bbox = {
            'x0': pred['x0'],
            'y0': pred['y0'], 
            'x1': pred['x1'],
            'y1': pred['y1'],
            'label': pred['label'],
            'confidence': pred['confidence']
        }
        bounding_boxes.append(bbox)
    return (
        Image,
        LayoutPredictor,
        artifacts_path,
        bbox,
        bounding_boxes,
        image,
        pred,
        predictions,
        predictor,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
