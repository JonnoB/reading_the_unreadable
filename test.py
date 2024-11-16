import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from pathlib import Path
    import logging
    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    import random

    # Setup logging
    #logging.basicConfig(level=logging.DEBUG)
    #_log = logging.getLogger(__name__)

    input_doc_path = Path("data/ncse_test_pdf/NS2-1843-04-01.pdf")
    output_dir = Path("data/docling_layout")
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, explicitly download the models
    artifacts_path = StandardPdfPipeline.download_models_hf(force=False)  # Changed to False to avoid redownloading
    print(f"Models downloaded to: {artifacts_path}")

    # Verify the layout model path exists
    layout_model_path = artifacts_path / "model_artifacts/layout/beehive_v0.0.5_pt"
    print(f"Checking if layout model exists at: {layout_model_path}")
    if not layout_model_path.exists():
        raise FileNotFoundError(f"Layout model path does not exist: {layout_model_path}")

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.images_scale = 1.0
    pipeline_options.artifacts_path = artifacts_path
    pipeline_options.generate_page_images = True  # Make sure we get the page images

    # Setup document converter
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
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
        StandardPdfPipeline,
        artifacts_path,
        doc_converter,
        input_doc_path,
        layout_model_path,
        logging,
        output_dir,
        pipeline_options,
        plt,
        random,
    )


@app.cell
def __(DocumentConverter, ImageDraw, Path, plt, random):
    def process_pdfs(input_dir: Path, 
                    output_dir: Path, 
                    doc_converter: DocumentConverter,
                    show_plots: bool = True) -> None:
        """
        Process all PDFs in a directory and save layout visualizations.
        
        Args:
            input_dir: Path to directory containing PDFs
            output_dir: Path to save visualizations
            doc_converter: Initialized DocumentConverter instance
            show_plots: Whether to display plots interactively (default: True)
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all PDF files in input directory
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        for pdf_file in pdf_files:
            try:
                print(f"Processing: {pdf_file.name}")
                
                # Convert document
                conv_result = doc_converter.convert(pdf_file)
                
                # Process each page
                for page_idx, page in enumerate(conv_result.pages):
                    if page.predictions.layout and page.image:
                        # Create a copy of the image to draw on
                        image_with_boxes = page.image.copy()
                        draw = ImageDraw.Draw(image_with_boxes)
                        
                        # Draw boxes for each cluster with different colors
                        for cluster in page.predictions.layout.clusters:
                            color = (random.randint(0, 255), 
                                    random.randint(0, 255), 
                                    random.randint(0, 255))
                            
                            bbox = cluster.bbox.as_tuple()
                            draw.rectangle(bbox, outline=color, width=2)
                            draw.text((bbox[0], bbox[1] - 10), 
                                    str(cluster.label), 
                                    fill=color)
                        
                        if show_plots:
                            plt.figure(figsize=(15, 20))
                            plt.imshow(image_with_boxes)
                            plt.axis('off')
                            plt.title(f'{pdf_file.name} - Page {page_idx + 1}')
                            plt.show()
                        
                        # Save with original filename but different extension
                        output_path = output_dir / f"{pdf_file.stem}_p{page_idx + 1}.png"
                        image_with_boxes.save(output_path)
                        
                print(f"Completed processing: {pdf_file.name}")
                
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue
            finally:
                if show_plots:
                    plt.close('all')
    return (process_pdfs,)


@app.cell
def __(Path, doc_converter, output_dir, process_pdfs):
    process_pdfs(Path('data/ncse_test_pdf'), 
                    output_dir, 
                    doc_converter,
                    False)
    return


if __name__ == "__main__":
    app.run()
