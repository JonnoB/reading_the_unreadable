""" """

import os
from PIL import Image
import pytesseract
from tqdm import tqdm


def extract_text_from_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supported image extensions
    valid_extensions = (".png", ".jpg", ".jpeg")

    # Process each file in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(valid_extensions):
            # Full path to the image file
            image_path = os.path.join(input_folder, filename)

            try:
                # Open and process the image
                image = Image.open(image_path)

                # Extract text from the image
                text = pytesseract.image_to_string(image)

                # Create output filename (same name but with .txt extension)
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_folder, f"{base_name}.txt")

                # Save the extracted text to a file
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print("\nText extraction completed!")


extract_text_from_images(
    "data/converted/ncse_cropped_images", "data/model_performance/NCSE_tesseract"
)

extract_text_from_images(
    "data/BLN600/Images_jpg", "data/model_performance/BLN_tesseract"
)

example_image = Image.open("data/example_for_paper/NS2-1843-04-01_page_4_excerpt.png")
text = pytesseract.image_to_string(example_image)

with open(
    "data/example_for_paper/example_results/tesseract.txt", "w", encoding="utf-8"
) as f:
    f.write(text)

text
