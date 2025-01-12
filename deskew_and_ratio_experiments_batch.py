import pandas as pd
from send_to_lm_functions import process_issues_to_jobs
import os
from mistralai import Mistral

# API setup
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)


# Prompt dictionary
prompt_dict = {
    'plain text': """The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only. Do not add any commentary.""",
    'figure': 'Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary',
    'table': 'Please extract the table from the image taken from a 19th century English newspaper. Use markdown, do not add any commentary'
}

ncse_image_path = 'data/converted/cropped_images'

BLN_image_path = 'data/BLN600/Images_jpg'

#
# Creating the 'bounding box' dataframe here, this probably isn't the best idea but I can change if necessary
#

#Doesn't actually include any bbox information as this is not'necessary when the image is not cropped
BLN600_bbox = pd.DataFrame({'filename': os.listdir( BLN_image_path)})
BLN600_bbox['page_id'] = BLN600_bbox['filename'].str.replace('.jpg', '') + "_page_1"
BLN600_bbox['box_page_id'] = 'B0C1R0' #This is just so the data is parsed properly when it is returned
BLN600_bbox['issue'] = 'test_' + ((BLN600_bbox.index // 100) + 1).astype(str)
BLN600_bbox['class'] = 'plain text'

# read bbox csv
ncse_bbox = pd.read_csv('data/ncse_testset_bboxes.csv')

# create the filename properly
ncse_bbox['filename'] = ncse_bbox['page_id'] + "_" + ncse_bbox['box_page_id'] + ".png"

# Process datasets
datasets = [
   # (ncse_image_path, ncse_bbox, 'NCSE'),
    (BLN_image_path, BLN600_bbox, 'BLN600')
]

experiments = list(zip(
    [True, True, True, False, False, False],
    [1, 1.5, 2, 1, 1.5, 2]
))

for image_path, bbox_df, dataset_name in datasets:
    print(f"Processing dataset: {dataset_name}")
    
    # Making a loop like this is obviously uncessarily slow as the image loading and deskewing and cropping is done
    # multiple times, however, this doesn't need to be fast for this test.
    for deskew, max_ratio in experiments:

        file_name = f"{dataset_name}_deskew_{str(deskew)}_max_ratio_{str(max_ratio)}"
        # Create output filename based on the experiment name
        output_file = f'data/processed_jobs/{file_name}.csv'

        # This ensures when the results are downloaded they are saved with the filename that matches the experiment
        bbox_df['issue'] = f'{file_name}_' + ((bbox_df.index // 100) + 1).astype(str)
        # Process the data
        process_issues_to_jobs(
            bbox_df = bbox_df,
            images_folder = image_path,
            prompt_dict = prompt_dict,
            client = client,
            output_file = output_file,
            deskew = deskew,
            crop_image = False,
            max_ratio = max_ratio
        )
        
        print(f"Completed processing {file_name}")
    print(f'Dataset "{dataset_name}" completed')

print("All files processed!")