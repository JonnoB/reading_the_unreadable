"""
This script loops through the periodical folders and performs object detection on all the image files inside.
Saves a parquet file of each periodical.

The size of the image will be adjusted to that closest to the target which is also a multiple of the stride

"""

from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd


filepath = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
)
model = YOLOv10(filepath)

output_folder = "periodical_bboxes/docyolo"

multiplier = 2

image_size_base = 1056
batch_size_base = 64


os.makedirs(output_folder, exist_ok=True)

folder_mapping = {
    "all_files_png_120": [
        "English_Womans_Journal_issue_PDF_files",
        "Leader_issue_PDF_files",
        "Monthly_Repository_issue_PDF_files",
        "Publishers_Circular_issue_PDF_files",
        "Tomahawk_issue_PDF_files",
    ],
    "all_files_png_200": ["Northern_Star_issue_PDF_files"],
}

folder_mapping = {
    "all_files_png_120": [
        "Monthly_Repository_issue_PDF_files",
        "Tomahawk_issue_PDF_files",
    ]
}

# Define settings for each folder
folder_settings = {
    "all_files_png_120": {"image_size": image_size_base, "batch_size": batch_size_base},
    "all_files_png_200": {
        "image_size": image_size_base * multiplier,
        "batch_size": int(batch_size_base / (multiplier**2)),
    },
}

for folder, periodicals in folder_mapping.items():
    # Get the settings for the current folder
    current_settings = folder_settings[folder]
    image_size = current_settings["image_size"]
    batch_size = current_settings["batch_size"]

    for periodical in periodicals:
        input_dir = f"image_files/{folder}/{periodical}"
        bbox_output = f"{periodical}_{image_size}.parquet"
        output_path = os.path.join(output_folder, bbox_output)

        # Skip if output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {periodical} in {folder} - output file already exists")
            continue
        else:
            print(f"Processing {periodical}")
        # Get all image paths
        image_paths = list(Path(input_dir).glob("*.png"))
        all_detections = []

        # Process images in batches
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]

            try:
                # Predict on batch of images
                det_res = model.predict(
                    [str(p) for p in batch_paths],
                    imgsz=image_size,
                    conf=0.2,
                    device="cuda:0",
                    verbose=False,
                )

                # Process each result in the batch
                for img_path, result in zip(batch_paths, det_res):
                    filename = img_path.name

                    # Get image dimensions from the result
                    img_height = result.orig_shape[0]  # Original image height
                    img_width = result.orig_shape[1]  # Original image width

                    # Extract bounding box information
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = box.cls[0].item()
                        cls_name = result.names[int(cls)]

                        detection_info = {
                            "filename": filename,
                            "class": cls_name,
                            "confidence": conf,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "page_height": img_height,
                            "page_width": img_width,
                        }
                        all_detections.append(detection_info)

            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

        # Save all detection results to a CSV file
        df = pd.DataFrame(all_detections)
        df.to_parquet(output_path, index=False)
        print(f"Results saved to {bbox_output}")
