from efficient_ocr import EffOCR
import os
from pathlib import Path
from tqdm import tqdm

model = EffOCR(
    config={
        "Recognizer": {
            "char": {
                "model_backend": "onnx",
                "model_dir": "./models",
                "hf_repo_id": "dell-research-harvard/effocr_en/char_recognizer",
            },
            "word": {
                "model_backend": "onnx",
                "model_dir": "./models",
                "hf_repo_id": "dell-research-harvard/effocr_en/word_recognizer",
            },
        },
        "Localizer": {
            "model_dir": "./models",
            "hf_repo_id": "dell-research-harvard/effocr_en",
            "model_backend": "onnx",
        },
        "Line": {
            "model_dir": "./models",
            "hf_repo_id": "dell-research-harvard/effocr_en",
            "model_backend": "onnx",
        },
    }
)


def process_images_to_text(
    model,
    source_folder,
    destination_folder,
    supported_extensions=(".jpg", ".jpeg", ".png"),
):
    """
    Process all images in source folder and save inference results as text files.
    Skips files that have already been processed.

    Args:
        model: The inference model to use
        source_folder (str): Path to folder containing source images
        destination_folder (str): Path to folder where text files will be saved
        supported_extensions (tuple): Tuple of supported image file extensions
    """
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get list of already processed files
    existing_txt_files = {
        Path(f).stem for f in os.listdir(destination_folder) if f.endswith(".txt")
    }

    # Get list of all image files that need processing
    image_files = [
        f
        for f in os.listdir(source_folder)
        if f.lower().endswith(supported_extensions)
        and Path(f).stem not in existing_txt_files
    ]

    # Counter for errors
    errors = 0

    # Process files with progress bar
    for filename in tqdm(image_files, desc="Processing images"):
        try:
            # Construct full path for input image
            image_path = os.path.join(source_folder, filename)

            # Perform inference
            result = model.infer(image_path)

            # Create text file name (same name as image but with .txt extension)
            text_filename = Path(filename).stem + ".txt"
            text_path = os.path.join(destination_folder, text_filename)

            # Save the inference result to text file
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(result[0].text)

        except Exception as e:
            errors += 1
            print(f"\nError processing {filename}: {str(e)}")

    # Final summary
    total_processed = len(image_files) - errors
    already_existing = len(existing_txt_files)

    print("\nProcessing complete!")
    print(f"Already existing files: {already_existing}")
    print(f"Newly processed files: {total_processed}")
    print(f"Errors encountered: {errors}")
    print(f"Total files in destination: {already_existing + total_processed}")


# Example usage:
# source_folder = "Images_jpg"
# destination_folder = "Text_output"
# process_images_to_text(model, source_folder, destination_folder)

process_images_to_text(
    model, "Images_jpg", "BLN_effocr", supported_extensions=(".jpg", ".png")
)


process_images_to_text(
    model,
    "ncse_cropped_images",
    "NCSE_effocr",
    supported_extensions=(".jpg", ".jpeg", ".png"),
)
