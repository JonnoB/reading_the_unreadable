from helper_functions import convert_pdf_to_image
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import traceback


##############
##
## This script is run to convert the pdfs into smaller png files which can be used directly for machine learning.
## The PNGs are converted to 1 bit files to minimise the size.
## THe script is run twice once with dpi = 200 and once with dpi = 72. The 200 dpi can be used by Pixtral to 
## extract the text, whilst the smaller 72 dpi can be used as training data to improve the text box detection on the
## pre-made bounding boxes.
##
############

image_dpi = 72
save_folder = f'/media/jonno/ncse/converted/all_files_png_{image_dpi}'
source_folder = os.path.join('/media/jonno/ncse')

subfolder_names = ['English_Womans_Journal_issue_PDF_files', 'Leader_issue_PDF_files', 'Monthly_Repository_issue_PDF_files',
 'Northern_Star_issue_PDF_files', 'Publishers_Circular_issue_PDF_files','Tomahawk_issue_PDF_files']


# Define log file path
log_file = os.path.join(save_folder, 'conversion_log.csv')

# Initialize or load existing log
if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
else:
    log_df = pd.DataFrame(columns=['timestamp', 'subfolder', 'file', 'status', 'error_message'])


def update_log(subfolder, file, status, error_message=''):
    global log_df
    new_row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'subfolder': subfolder,
        'file': file,
        'status': status,
        'error_message': error_message
    }
    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    log_df.to_csv(log_file, index=False)

for subfolder in subfolder_names:
    print(f'Processing the {subfolder} folder')
    subfolder_path = os.path.join(source_folder, subfolder)    
    file_names = os.listdir(subfolder_path)
    save_subfolder = os.path.join(save_folder, subfolder)
    os.makedirs(save_subfolder, exist_ok=True)

    for file in tqdm(file_names):
        # Skip if file was already successfully processed
        if not log_df.empty and len(log_df[(log_df['subfolder'] == subfolder) & 
                                         (log_df['file'] == file) & 
                                         (log_df['status'] == 'success')]) > 0:
            continue

        try:
            convert_pdf_to_image(os.path.join(subfolder_path, file), 
                               output_folder=save_subfolder, 
                               dpi=image_dpi, 
                               image_format='PNG', 
                               use_greyscale=True)
            update_log(subfolder, file, 'success')
        except Exception as e:
            error_message = str(e) + '\n' + traceback.format_exc()
            update_log(subfolder, file, 'failed', error_message)