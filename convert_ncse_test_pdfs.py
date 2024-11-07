from helper_functions import convert_pdf_to_image
import os


file_names = os.listdir('data/ncse_test_pdf')


for file in file_names:
    convert_pdf_to_image(os.path.join('data','ncse_test_pdf' ,file), output_folder='data/ncse_test_jpg', dpi=300, image_format='JPEG')