""" 
The asyncio cannot be run from within marimo as it causes a nested loop issue
This can be got around but it is easier just to do this as a script

Leader start time 0723 2024/12/21
"""

import os
from lightning_sdk import Studio

studio = Studio(name="doclayout-yolo", teamspace="Language-model", user="ucabbou")
total_root = "data/ncse_test_jpg/"
total_root = "/media/jonno/ncse/converted/all_files_png_120"

files = os.listdir(total_root)
zip_files = [f for f in files if f.endswith('.zip')]
zip_files = ['English_Womans_Journal_issue_PDF_files.zip', 'Tomahawk_issue_PDF_files.zip']

print(f'zip files identified {zip_files}')
for zip_file in zip_files:
    remote_path = os.path.join('image_files', os.path.basename(total_root), zip_file)
    local_path = os.path.join(total_root, zip_file)
    studio.upload_file(local_path, remote_path)
    print(f"Uploaded: {zip_file}")