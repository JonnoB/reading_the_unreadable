"""
This script transfers a folder of compressed NCSE image files to a lightning.ai studio to run DocLayout-Yolo on.
To use this script you need to have your lightnint user_id and API key in the .env file.
See lightning documentation  for details.

The asyncio cannot be run from within marimo as it causes a nested loop issue
This can be got around but it is easier just to do this as a script


Adjust the details below such as , 'name', 'teamspace', 'user', and 'total_root', to you use case.

"""

import os
from lightning_sdk import Studio

from pathlib import Path

# Change working directory to project root
os.chdir(Path(__file__).parent.parent)
image_folder = os.environ["image_folder"]

studio = Studio(name="doclayout-yolo", teamspace="Language-model", user="ucabbou")
total_root = "data/ncse_test_jpg/"
total_root = os.path.join(image_folder, "converted/all_files_png_120")

files = os.listdir(total_root)
zip_files = [f for f in files if f.endswith(".zip")]


print(f"zip files identified {zip_files}")
for zip_file in zip_files:
    remote_path = os.path.join("image_files", os.path.basename(total_root), zip_file)
    local_path = os.path.join(total_root, zip_file)
    studio.upload_file(local_path, remote_path)
    print(f"Uploaded: {zip_file}")
