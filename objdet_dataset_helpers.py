import os 
from helper_functions import scale_bbox
import pandas as pd
from tqdm import tqdm
import shutil
import json
from PIL import Image
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


import shutil
import yaml

def create_image_list(image_df, local_image_dir, coco_image_dir = ''):
    """Create list of image dictionaries
    image_df: dataframe. A pandas dataframe containing image meta data
    local_image_dir: str. The path to the images on the computer used to open images and get addition info
    coco_image_dir: str the path in the coco dataset, defaults to nothing so that json and images are all at same level.
    """
    image_list = []
    for idx, row in image_df.iterrows():
        image_path = os.path.join(local_image_dir, row['filename'])
        with Image.open(image_path) as img:
            width, height = img.size

        image_list.append({
            "id": row['page_id'],
            "file_name": os.path.join(os.path.basename(coco_image_dir), row['filename']),
            "width": width,
            "height": height,
            "license": 1,
        })
    return image_list

def to_coco_format(bbox):
    """
    Convert [x1, x2, y1, y2] to COCO format [x, y, width, height]
    where (x, y) is the top-left corner
    """
    x1, x2, y1, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return [x1, y1, width, height]

def create_annotation_list(bbox_df, image_list):
    """Create list of annotation dictionaries"""

    dimensions_dict = {item['id']: (item['width'], item['height'])for item in image_list}

    temp_list = []

    for idx, row in bbox_df.iterrows():
        try:
            #scale bounding boxes to current image size using reference from original scan
            bounding_box_new = scale_bbox(row['bounding_box_list'], 
                                        original_size=(row['width'], row['height']), 
                                        new_size=dimensions_dict[row['page_id']])
            bounding_box_new = to_coco_format(bounding_box_new)

            temp_list.append({
                "id": row['id'],
                "image_id": row['page_id'],
                "category_id": row['article_type_id'],
                "bbox": bounding_box_new,
                "area": bounding_box_new[2] * bounding_box_new[3],
                "iscrowd": 0
            })

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue

    return temp_list

def turn_into_coco(meta_data_df, page_df, local_image_dir, coco_image_dir = ''):

    images_list = create_image_list(page_df, 
                              local_image_dir,
                               coco_image_dir    )

    annotations_list = create_annotation_list(meta_data_df, images_list )


    categories_list = [
            {
                "id": 1,
                "name": "article",
                "supercategory": "text"
            },        {
                "id": 2,
                "name": "advert",
                "supercategory": "text"
            },        {
                "id": 3,
                "name": "image",
                "supercategory": "image"
            }
        ]

    return {'images': images_list, 'annotations': annotations_list, 'categories': categories_list}

    

def create_cross_validation_coco(meta_data_df, page_df, local_image_dir, output_dir, n_folds=5, coco_image_dir='', random_state=42):
    """
    Create and save x-fold cross-validation COCO format JSON files
    
    Parameters:
    -----------
    meta_data_df : DataFrame
        DataFrame containing annotation metadata
    page_df : DataFrame
        DataFrame containing page/image information
    local_image_dir : str
        Path to local image directory
    output_dir : str
        Directory where JSON files will be saved
    n_folds : int
        Number of folds for cross-validation
    coco_image_dir : str
        Path for images in COCO dataset
    random_state : int
        Random seed for reproducibility
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique page IDs
    unique_pages = page_df['page_id'].unique()
    
    # Initialize K-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Perform k-fold split on page IDs
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_pages)):
        # Get train and validation page IDs
        train_pages = unique_pages[train_idx]
        val_pages = unique_pages[val_idx]
        
        # Filter DataFrames for train set
        train_page_df = page_df[page_df['page_id'].isin(train_pages)]
        train_meta_df = meta_data_df[meta_data_df['page_id'].isin(train_pages)]
        
        # Filter DataFrames for validation set
        val_page_df = page_df[page_df['page_id'].isin(val_pages)]
        val_meta_df = meta_data_df[meta_data_df['page_id'].isin(val_pages)]
        
        # Create COCO format for train set
        train_coco = turn_into_coco(
            train_meta_df,
            train_page_df,
            local_image_dir,
            coco_image_dir
        )
        
        # Create COCO format for validation set
        val_coco = turn_into_coco(
            val_meta_df,
            val_page_df,
            local_image_dir,
            coco_image_dir
        )
        
        # Save train JSON
        train_path = os.path.join(output_dir, f'train_fold_{fold}.json')
        with open(train_path, 'w') as f:
            json.dump(train_coco, f)
            
        # Save validation JSON
        val_path = os.path.join(output_dir, f'val_fold_{fold}.json')
        with open(val_path, 'w') as f:
            json.dump(val_coco, f)
        
        print(f"Fold {fold + 1}/{n_folds} completed")
        print(f"Train set size: {len(train_pages)} pages")
        print(f"Validation set size: {len(val_pages)} pages")
        print("------------------------")




##
##
## YOLO VERSION
##

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert [x1, x2, y1, y2] format directly to YOLO format [x_center, y_center, width, height]
    All values in YOLO format are normalized to [0, 1]
    """
    x1, x2, y1, y2 = bbox
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Calculate center coordinates
    x_center = x1 + width/2
    y_center = y1 + height/2
    
    # Normalize
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height
    
    return [x_center, y_center, width, height]

def create_yolo_annotation(image_path, annotations, img_width, img_height):
    """Create YOLO format annotation file content"""
    yolo_annotations = []
    
    for ann in annotations:
        category_id = ann['article_type_id'] - 1  # YOLO uses 0-based indexing
        bbox = convert_bbox_to_yolo(ann['bounding_box_list'], img_width, img_height)
        yolo_annotations.append(f"{category_id} {' '.join([str(x) for x in bbox])}")
    
    return '\n'.join(yolo_annotations)

def create_cross_validation_yolo(meta_data_df, page_df, local_image_dir, output_dir, n_folds=5, random_state=42):
    """
    Create and save x-fold cross-validation YOLO format files
    """
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique page IDs
    unique_pages = page_df['page_id'].unique()
    
    # Initialize K-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Create data.yaml file
    yaml_content = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,  # number of classes
        'names': ['article', 'advert', 'image']
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)
    
    # Perform k-fold split
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_pages)):
        # Create fold-specific directories
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        for split in ['train', 'val']:
            os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)
        
        # Get train and validation page IDs
        train_pages = unique_pages[train_idx]
        val_pages = unique_pages[val_idx]
        
        # Process train and validation sets
        for split, pages in [('train', train_pages), ('val', val_pages)]:
            split_page_df = page_df[page_df['page_id'].isin(pages)]
            
            for _, row in split_page_df.iterrows():
                # Get image information
                img_path = os.path.join(local_image_dir, row['filename'])
                with Image.open(img_path) as img:
                    width, height = img.size
                
                # Get annotations for this image
                img_annotations = meta_data_df[meta_data_df['page_id'] == row['page_id']].to_dict('records')
                
                # Convert annotations to YOLO format
                yolo_content = create_yolo_annotation(img_path, img_annotations, width, height)
                
                # Create paths for new image and label files
                new_img_path = os.path.join(fold_dir, 'images', split, row['filename'])
                label_filename = os.path.splitext(row['filename'])[0] + '.txt'
                label_path = os.path.join(fold_dir, 'labels', split, label_filename)
                
                # Copy image to new location
                shutil.copy2(img_path, new_img_path)
                
                # Save YOLO format annotations
                with open(label_path, 'w') as f:
                    f.write(yolo_content)
        
        # Create fold-specific data.yaml
        fold_yaml_content = {
            'path': fold_dir,
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,  # number of classes
            'names': ['article', 'advert', 'image']
        }
        
        with open(os.path.join(fold_dir, 'data.yaml'), 'w') as f:
            yaml.dump(fold_yaml_content, f)
        
        print(f"Fold {fold + 1}/{n_folds} completed")
        print(f"Train set size: {len(train_pages)} pages")
        print(f"Validation set size: {len(val_pages)} pages")
        print("------------------------")


##
##      plot for the different types
##
## Good to have a straight forward way to plot the pages and see what is happening.


def plot_yolo_box(image_path, label_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width, _ = image.shape
    
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Read YOLO format labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    # Plot each bounding box
    for line in lines:
        # Parse YOLO format (class_id, x_center, y_center, box_width, box_height)
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())
        
        # Convert YOLO coordinates to pixel coordinates
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        
        # Calculate top-left corner
        x_min = x_center - (box_width / 2)
        y_min = y_center - (box_height / 2)
        
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.show()

def plot_coco_box(image_path, annotation_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Read COCO format annotations
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    # Plot each bounding box
    for ann in annotations['annotations']:
        # Get bbox coordinates (x_min, y_min, width, height)
        x_min, y_min, box_width, box_height = ann['bbox']
        
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        # Optionally, add category name if available
        if 'categories' in annotations:
            category_id = ann['category_id']
            category_name = next(
                (cat['name'] for cat in annotations['categories'] if cat['id'] == category_id),
                str(category_id)
            )
            ax.text(x_min, y_min-5, category_name, color='r')
    
    plt.axis('off')
    plt.show()