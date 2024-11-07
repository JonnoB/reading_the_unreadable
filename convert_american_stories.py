##
## Convert the json files from the american stories pipeline to txt files
##


import json
import os
from pathlib import Path

# Specify your input and output directories
input_dir = "data/BLN_results/american_stories_json"  # Replace with your json folder path
output_dir = "data/BLN_results/american_stories_txt"  # Replace with desired output folder path

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all json files in the input directory
for json_file in Path(input_dir).glob("*.json"):
    try:
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get the raw text from the first bbox (assuming structure is consistent)
        raw_text = data['bboxes'][0]['raw_text']
        
        # Create output filename (same name but with .txt extension)
        output_filename = Path(output_dir) / (json_file.stem + ".txt")
        
        # Write the raw text to a new text file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(raw_text)
            
        print(f"Processed: {json_file.name}")
            
    except Exception as e:
        print(f"Error processing {json_file.name}: {str(e)}")

print("Conversion complete!")