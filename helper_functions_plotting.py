"""
The functions in this helper module are to help visualise the images. 
This is primarily for debugging and understanding the dataset

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.patches as patches


def calculate_padding(
    img_width: int,
    img_height: int,
    boxes_df: pd.DataFrame,
    coord_cols: list = ['x0', 'x1', 'y0', 'y1'],
    padding: int = 20
) -> tuple:
    """
    Calculate required padding for an image based on bounding box coordinates.
    
    Parameters:
    - img_width: int, width of the original image
    - img_height: int, height of the original image
    - boxes_df: DataFrame containing bounding box coordinates
    - coord_cols: list of column names for coordinates [x0, x1, y0, y1]
    - padding: int, additional padding to add around boxes
    
    Returns:
    - tuple containing:
        - pad_left: padding needed on left side
        - pad_top: padding needed on top
        - pad_right: padding needed on right side
        - pad_bottom: padding needed on bottom
        - new_width: width of padded image
        - new_height: height of padded image
    """
    # Calculate minimum and maximum coordinates
    min_x = min(boxes_df[coord_cols[0]].min() - padding, 0)
    min_y = min(boxes_df[coord_cols[2]].min() - padding, 0)
    max_x = max(boxes_df[coord_cols[1]].max() + padding, img_width)
    max_y = max(boxes_df[coord_cols[3]].max() + padding, img_height)
    
    # Calculate padding for each side
    pad_left = abs(min(min_x, 0))
    pad_top = abs(min(min_y, 0))
    pad_right = max(max_x - img_width, 0)
    pad_bottom = max(max_y - img_height, 0)
    
    # Calculate new dimensions
    new_width = int(img_width + pad_left + pad_right)
    new_height = int(img_height + pad_top + pad_bottom)
    
    return (
        pad_left,
        pad_top,
        pad_right,
        pad_bottom,
        new_width,
        new_height
    )

def scale_bbox_coordinates(
    boxes_df: pd.DataFrame, 
    scale_factor_x: float = 1.0,
    scale_factor_y: float = 1.0,
    ) -> pd.DataFrame:
    """
    Scale bounding box coordinates by separate x and y scaling factors.
    
    Parameters:
    - boxes_df: DataFrame with columns ['x0', 'x1', 'y0', 'y1']
    - scale_factor_x: float, scaling factor for x coordinates
    - scale_factor_y: float, scaling factor for y coordinates
    
    Returns:
    - DataFrame with additional columns ['scaled_x0', 'scaled_x1', 'scaled_y0', 'scaled_y1']
    """
    # Create a copy to avoid modifying the original DataFrame
    scaled_df = boxes_df.copy()
    
    # Scale x coordinates
    scaled_df['scaled_x0'] = boxes_df['x0'] * scale_factor_x
    scaled_df['scaled_x1'] = boxes_df['x1'] * scale_factor_x
    
    # Scale y coordinates
    scaled_df['scaled_y0'] = boxes_df['y0'] * scale_factor_y
    scaled_df['scaled_y1'] = boxes_df['y1'] * scale_factor_y
    
    return scaled_df

# Example usage:
# boxes_df_scaled = scale_bbox_coordinates(boxes_df, scale_factor_x=0.8, scale_factor_y=1.2)


def _plot_single_image_with_boxes_core(
    ax,
    image_path: str,
    boxes_df: pd.DataFrame,
    scale_factor_x: float = 1.0,
    scale_factor_y: float = 1.0,
    box_color='red',
    box_linewidth=2,
    padding_color='white',
    padding: int = 20,
    title: str = None
):
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Scale the boxes
    scaled_boxes = scale_bbox_coordinates(
        boxes_df,
        scale_factor_x=scale_factor_x,
        scale_factor_y=scale_factor_y
    )
    
    # Use scaled coordinates
    coord_cols = ['scaled_x0', 'scaled_x1', 'scaled_y0', 'scaled_y1']
    
    # Calculate padding
    pad_left, pad_top, pad_right, pad_bottom, new_width, new_height = calculate_padding(
        img_width=img_width,
        img_height=img_height,
        boxes_df=scaled_boxes,
        coord_cols=coord_cols,
        padding=padding
    )
    
    # Create new image with padding
    new_img = Image.new('L', (new_width, new_height), padding_color)
    new_img.paste(img, (int(pad_left), int(pad_top)))
    
    # Display image
    ax.imshow(new_img, cmap='gray')
    
    # Draw bounding boxes
    for _, box in scaled_boxes.iterrows():
        rect = patches.Rectangle(
            (box[coord_cols[0]] + pad_left, box[coord_cols[2]] + pad_top),
            box[coord_cols[1]] - box[coord_cols[0]],
            box[coord_cols[3]] - box[coord_cols[2]],
            linewidth=box_linewidth,
            edgecolor=box_color,
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # Remove axes and add title
    ax.axis('off')
    if title:
        ax.set_title(title)

def plot_single_image_with_boxes(

    image_path: str,
    boxes_df: pd.DataFrame,
    scale_factor_x: float = 1.0,
    scale_factor_y: float = 1.0,
    figsize=(10, 10),
    box_color='red',
    box_linewidth=2,
    padding_color='white',
    padding: int = 20,
    title: str = None,
    save_path: str = None
):
    """
    Plot a single image with its corresponding bounding boxes.

    Parameters:
    -----------
    image_path : str
        Path to the image file to be plotted
    boxes_df : pd.DataFrame
        DataFrame containing bounding box coordinates with columns ['x0', 'x1', 'y0', 'y1']
    scale_factor_x : float, optional (default=1.0)
        Scaling factor for x coordinates of bounding boxes
    scale_factor_y : float, optional (default=1.0)
        Scaling factor for y coordinates of bounding boxes
    figsize : tuple, optional (default=(10, 10))
        Size of the figure (width, height) in inches
    box_color : str, optional (default='red')
        Color of the bounding box lines
    box_linewidth : int, optional (default=2)
        Width of the bounding box lines
    padding_color : str, optional (default='white')
        Color of the padding around the image
    padding : int, optional (default=20)
        Amount of padding to add around the boxes in pixels
    title : str, optional (default=None)
        Title to display above the image
    save_path : str, optional (default=None)
        If provided, saves the plot to this path instead of displaying

    Returns:
    --------
    None
        Displays the plot or saves it to file
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use the core plotting function
    _plot_single_image_with_boxes_core(
        ax=ax,
        image_path=image_path,
        boxes_df=boxes_df,
        scale_factor_x=scale_factor_x,
        scale_factor_y=scale_factor_y,
        box_color=box_color,
        box_linewidth=box_linewidth,
        padding_color=padding_color,
        padding=padding,
        title=title
    )
    
    # Handle saving/showing
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def plot_multiple_images_with_boxes(
    image_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    page_ids: list,
    scale_factor_x: float = 1.0,
    scale_factor_y: float = 1.0,
    figsize=(20, 30),
    box_color='red',
    box_linewidth=2,
    padding_color='white',
    padding: int = 20,
    save_path=None
):
    
    """
    Plot multiple images with their corresponding bounding boxes in a grid layout.

    Parameters:
    -----------
    image_df : pd.DataFrame
        DataFrame containing image information with columns ['page_id', 'output_file']
    boxes_df : pd.DataFrame
        DataFrame containing bounding box coordinates with columns 
        ['page_id', 'x0', 'x1', 'y0', 'y1']
    page_ids : list
        List of page IDs to plot
    scale_factor_x : float, optional (default=1.0)
        Scaling factor for x coordinates of bounding boxes
    scale_factor_y : float, optional (default=1.0)
        Scaling factor for y coordinates of bounding boxes
    figsize : tuple, optional (default=(20, 30))
        Size of the figure (width, height) in inches
    box_color : str, optional (default='red')
        Color of the bounding box lines
    box_linewidth : int, optional (default=2)
        Width of the bounding box lines
    padding_color : str, optional (default='white')
        Color of the padding around the images
    padding : int, optional (default=20)
        Amount of padding to add around the boxes in pixels
    save_path : str, optional (default=None)
        If provided, saves the plot to this path instead of displaying

    Returns:
    --------
    None
        Displays the plot or saves it to file

    Notes:
    ------
    The images will be arranged in a grid with 2 columns and enough rows to 
    accommodate all images. Empty subplots will be hidden if the number of 
    images is odd.
    """
    # Create figure with subplots
    rows = (len(page_ids) + 1) // 2  # Calculate rows needed
    fig, axes = plt.subplots(2, rows, figsize=figsize)
    axes = axes.ravel()
    
    # Plot each image
    for idx, (ax, page_id) in enumerate(zip(axes, page_ids)):
        # Get the image path and boxes
        image_path = image_df[image_df['page_id'] == page_id]['output_file'].iloc[0]
        page_boxes = boxes_df[boxes_df['page_id'] == page_id].copy()
        overlap_percent = round(image_df[image_df['page_id'] == page_id]['text_overlap_percent'].iloc[0]*100)
        # Plot single image
        _plot_single_image_with_boxes_core(
            ax=ax,
            image_path=image_path,
            boxes_df=page_boxes,
            scale_factor_x=scale_factor_x,
            scale_factor_y=scale_factor_y,
            box_color=box_color,
            box_linewidth=box_linewidth,
            padding_color=padding_color,
            padding=padding,
            title=f'Page ID: {page_id}/Overlap: {overlap_percent}'
        )
    
    # Hide any unused subplots
    for idx in range(len(page_ids), len(axes)):
        axes[idx].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

    