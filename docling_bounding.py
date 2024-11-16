import marimo

__generated_with = "0.9.18"
app = marimo.App(width="medium")


@app.cell
def __():
    _layout_model_path = "model_artifacts/layout/beehive_v0.0.5_pt"
    _table_model_path = "model_artifacts/tableformer"
    return


@app.cell
def __():
    import logging
    import json
    import random
    from pathlib import Path
    from PIL import ImageDraw
    import matplotlib.pyplot as plt
    import pypdfium2 as pdfium
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    # Setup logging
    #logging.basicConfig(level=logging.DEBUG)
    #_log = logging.getLogger(__name__)

    # Configure input/output paths
    input_doc_path = Path("data/ncse_test_pdf/TTW_pageid_160259_pagenum_12_1867-12-21.pdf")
    output_dir = Path("bbox_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, let's verify we can open the PDF directly
    try:
        pdf = pdfium.PdfDocument(str(input_doc_path.absolute()))
        print(f"Direct PDF open successful. Page count: {len(pdf)}")
        pdf.close()
    except Exception as e:
        print(f"Error opening PDF directly: {e}")

    # Configure pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.images_scale = 1.0  # Increased for better quality

    # Setup document converter
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )
    return (
        DocumentConverter,
        ImageDraw,
        InputFormat,
        Path,
        PdfFormatOption,
        PdfPipelineOptions,
        PyPdfiumDocumentBackend,
        doc_converter,
        input_doc_path,
        json,
        logging,
        output_dir,
        pdf,
        pdfium,
        pipeline_options,
        plt,
        random,
    )


@app.cell
def __(
    ImageDraw,
    doc_converter,
    input_doc_path,
    json,
    output_dir,
    pdfium,
    pipeline_options,
    plt,
    random,
):
    try:
        # Convert document
        conv_result = doc_converter.convert(input_doc_path)

        print(f"Number of pages: {len(conv_result.pages)}")
        print(f"Backend type: {type(conv_result.input._backend)}")

        # Initialize PDF document directly
        pdf_doc = pdfium.PdfDocument(str(input_doc_path.absolute()))

        # Process first page
        if len(conv_result.pages) > 0:
            page = conv_result.pages[0]
            print(f"\nPage backend before: {type(page._backend)}")

            try:
                # Initialize page using the direct PDF document
                pdf_page = pdf_doc[0]
                page._backend._ppage = pdf_page

                if hasattr(page._backend, 'is_valid'):
                    print(f"Page backend valid: {page._backend.is_valid()}")

                    if page._backend.is_valid():
                        # Get text cells
                        cells = list(page._backend.get_text_cells())
                        print(f"Found {len(cells)} cells")

                        # Get page image
                        try:
                            image = pdf_page.render(
                                scale=pipeline_options.images_scale,
                                rotation=0
                            ).to_pil()

                            if image:
                                draw = ImageDraw.Draw(image)

                                # Draw boxes
                                for cell in cells:
                                    bbox = cell.bbox.as_tuple()
                                    # Scale the bbox coordinates
                                    scaled_bbox = [
                                        coord * pipeline_options.images_scale 
                                        for coord in bbox
                                    ]
                                    draw.rectangle(
                                        scaled_bbox,
                                        outline=(random.randint(30, 140),
                                                random.randint(30, 140),
                                                random.randint(30, 140)),
                                        width=2
                                    )
                                    # Display using matplotlib
                                plt.figure(figsize=(15, 20))
                                plt.imshow(image)
                                plt.axis('off')
                                plt.title('Page 1')
                                plt.show()

                                # Save image
                                output_path = output_dir / "page_0_viz.png"
                                image.save(output_path)
                                print(f"Saved visualization to: {output_path}")

                                # Save bounding box data
                                bbox_data = [
                                    {
                                        "text": cell.text,
                                        "bbox": cell.bbox.as_tuple(),
                                        "page": 0
                                    }
                                    for cell in cells
                                ]
                                json_path = output_dir / "page_0_boxes.json"
                                with json_path.open("w", encoding='utf-8') as f:
                                    json.dump(bbox_data, f, indent=2, ensure_ascii=False)
                                print(f"Saved bounding boxes to: {json_path}")

                                # Print some statistics
                                print(f"\nStatistics:")
                                print(f"Number of text cells: {len(cells)}")
                                print(f"Image size: {image.size}")

                            else:
                                print("Failed to create page image")
                        except Exception as img_error:
                            print(f"Error processing page image: {img_error}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("Page backend is not valid")
                else:
                    print("Page backend doesn't have is_valid method")

            except Exception as page_error:
                print(f"Error processing page: {page_error}")
                import traceback
                traceback.print_exc()
        else:
            print("No pages found in document")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        try:
            if 'pdf_doc' in locals():
                pdf_doc.close()
                print("Closed PDF document")
            if 'conv_result' in locals() and hasattr(conv_result.input._backend, 'unload'):
                conv_result.input._backend.unload()
                print("Unloaded backend")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

    print("\nProcessing complete")
    return (
        bbox,
        bbox_data,
        cell,
        cells,
        conv_result,
        draw,
        f,
        image,
        json_path,
        output_path,
        page,
        pdf_doc,
        pdf_page,
        scaled_bbox,
        traceback,
    )


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
