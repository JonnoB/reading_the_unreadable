from pdf2image import convert_from_path
from typing import List, Dict, Union, Optional
import fitz
import os
from PIL import Image
from pdf2image import convert_from_path
from markdown_it import MarkdownIt
from mdit_plain.renderer import RendererPlain

parser = MarkdownIt(renderer_cls=RendererPlain)


def get_page_images_info(pdf_path: str) -> List[Dict[str, Union[int, float, str]]]:
    """
    Get metadata about images embedded in PDF pages.
    """
    doc = fitz.open(pdf_path)
    page_images_info = []

    for page_num, page in enumerate(doc):
        page_rect = page.rect
        page_width_pt = page_rect.width
        page_height_pt = page_rect.height

        # Get images from the page
        image_list = page.get_images()

        if not image_list:
            print(f"Warning: No images found on page {page_num + 1}")
            continue

        for img_index, img in enumerate(image_list):
            try:
                # Get image info without extracting the full image
                image_width = img[2]  # width is at index 2
                image_height = img[3]  # height is at index 3

                # Calculate effective DPI
                width_dpi = (image_width / page_width_pt) * 72
                height_dpi = (image_height / page_height_pt) * 72

                # Calculate aspect ratios
                image_aspect_ratio = image_width / image_height
                page_aspect_ratio = page_width_pt / page_height_pt

                page_images_info.append(
                    {
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "pdf_width": image_width,
                        "pdf_height": image_height,
                        "page_width_pt": page_width_pt,
                        "page_height_pt": page_height_pt,
                        "calculated_width_dpi": round(width_dpi, 2),
                        "calculated_height_dpi": round(height_dpi, 2),
                        "image_aspect_ratio": round(image_aspect_ratio, 3),
                        "page_aspect_ratio": round(page_aspect_ratio, 3),
                        "image_format": "assume png",
                    }
                )

            except Exception as e:
                print(
                    f"Error processing image {img_index} on page {page_num + 1}: {str(e)}"
                )
                continue

    doc.close()

    if not page_images_info:
        print("Warning: No images found in the entire PDF")

    return page_images_info


def convert_pdf_to_image(
    pdf_path: str,
    output_folder: str = "output_images",
    dpi: int = 96,
    image_format: str = "PNG",
    use_greyscale: bool = True,
    quality: int = 85,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Converts each page of a PDF file into an image and saves the images to an output folder.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to be converted.
    output_folder : str, optional
        Directory where the output images will be saved. Defaults to 'output_images'.
    dpi : int, optional
        Resolution for the converted images, in dots per inch. Higher values increase image quality.
        Defaults to 96.
    image_format : str, optional
        Image format for the output files. Supported formats are 'PNG' and 'JPEG'.
        Defaults to 'PNG'.
    use_greyscale : bool, optional
        Whether to save as a 1 channel greyscale image. This reduces file size by about 66%.
        Defaults to True.
    quality : int, optional
        The quality setting for JPEG compression (1-100). Higher values mean better quality but larger files.
        Only applies when image_format is 'JPEG'. Defaults to 85.

    Returns
    -------
    list of dict
        A list containing dictionaries with information about each converted page:
        - 'original_file': Path to the source PDF
        - 'output_file': Path to the generated image file
        - 'page_number': Page number in the PDF
        - 'original_width': Width of the original PDF page
        - 'original_height': Height of the original PDF page
        - 'final_width': Width of the converted image
        - 'final_height': Height of the converted image

    Raises
    ------
    ValueError
        If an unsupported image format is specified (only 'PNG' and 'JPEG' are supported).

    Notes
    -----
    - The function creates the output directory if it doesn't exist.
    - Output files are named as '{original_filename}_page_{page_number}.{extension}'.
    - For JPEG format, quality and optimization parameters are applied.
    - For PNG format, maximum compression is applied.
    - When use_greyscale is True, images are converted to binary black and white using a threshold of 200.

    Examples
    --------
    >>> result = convert_pdf_to_image('example.pdf',
    ...                              output_folder='images',
    ...                              dpi=200,
    ...                              image_format='JPEG',
    ...                              use_greyscale=True,
    ...                              quality=85)
    >>> print(result[0]['output_file'])
    'images/example_page_1.jpg'
    """
    os.makedirs(output_folder, exist_ok=True)
    original_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    # Get PDF information using PyMuPDF
    page_metadata = get_page_images_info(pdf_path)
    if not page_metadata:
        raise ValueError("No images found in PDF file")

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)

    # Validate format and get extension
    format_map = {"PNG": "png", "JPEG": "jpg"}
    file_extension = format_map.get(image_format.upper())
    if not file_extension:
        raise ValueError("Unsupported format. Use 'PNG' or 'JPEG'.")

    page_info = []
    for i, image in enumerate(images):
        # Find corresponding metadata
        meta = next((m for m in page_metadata if m["page_num"] == i + 1), None)
        if not meta:
            print(f"Warning: No metadata found for page {i + 1}")
            continue

        output_file = os.path.join(
            output_folder, f"{original_filename}_page_{i + 1}.{file_extension}"
        )

        # Convert to greyscale if requested
        if use_greyscale:
            image = image.convert("L")
            threshold = 200
            image = image.point(lambda x: 0 if x < threshold else 255, "1")

        # Save with appropriate format settings
        if image_format.upper() == "JPEG":
            image.save(
                output_file, image_format.upper(), quality=quality, optimize=True
            )
        else:
            image.save(output_file, image_format.upper(), optimize=True, compression=9)

        # Get the dimensions of the output image
        output_width, output_height = image.size

        # Add output file path and dimensions to metadata
        meta["output_file"] = output_file
        meta["original_file"] = pdf_path
        meta["target_dpi"] = dpi
        meta["output_width"] = output_width
        meta["output_height"] = output_height

        page_info.append(meta)

    if not page_info:
        raise ValueError("No pages were successfully processed")

    return page_info
