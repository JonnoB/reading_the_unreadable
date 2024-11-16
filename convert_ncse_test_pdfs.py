from helper_functions import convert_pdf_to_image
import os
from tqdm import tqdm


save_folder = 'data/converted/ncse_test_png_200'
source_folder = os.path.join('data','ncse_test_pdf')
file_names = os.listdir(source_folder)

os.makedirs(save_folder, exist_ok=True)


for file in tqdm(file_names):

    convert_pdf_to_image(os.path.join(source_folder, file), output_folder= save_folder, dpi=200, image_format='PNG', use_greyscale=True)