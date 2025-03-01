"""
Pipeline stages for document processing workflow.

This module contains standalone functions for each stage of the newspaper
document processing pipeline, from bounding box prediction through article creation.
Each function handles one specific processing step and can be used independently
or orchestrated through the NewspaperPipeline class.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
from mistralai import Mistral

# Import necessary functions from other modules
from function_modules.bbox_functions import postprocess_bbox
from function_modules.send_to_lm_functions import (
    process_issues_to_jobs,
    download_processed_jobs,
    process_json_files
)
from function_modules.analysis_functions import (
    remove_line_breaks,
    process_documents
)

# Configure logging
logger = logging.getLogger(__name__)


def predict_bounding_boxes(
    periodical: str,
    image_folder: str,
    output_path: str,
    model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench",
    model_file: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
    image_size: int = 1056,
    batch_size: int = 64,
    conf_threshold: float = 0.2,
    device: Optional[str] = None
) -> str:
    """
    Run layout detection to predict bounding boxes.
    
    Args:
        periodical: Name of the periodical to process
        image_folder: Path to folder containing the images
        output_path: Path to save the resulting parquet file
        model_repo: HuggingFace model repository ID
        model_file: Model filename within the repository
        image_size: Size to resize images to
        batch_size: Processing batch size
        conf_threshold: Confidence threshold for detections
        device: Device to run model on (None for auto-detection)
        
    Returns:
        Path to the output parquet file
    """
    try:
        # Import here to avoid loading GPU libraries unnecessarily
        from huggingface_hub import hf_hub_download
        from doclayout_yolo import YOLOv10
        
        # Skip if already done
        if os.path.exists(output_path):
            logger.info(f"Bounding box prediction already completed for {periodical}. Skipping.")
            return output_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download model if needed
        logger.info(f"Downloading DocLayout-YOLO model from {model_repo}")
        filepath = hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
        )
        
        # Initialize model
        model = YOLOv10(filepath)
        
        # Determine device
        if device is None:
            device = "cuda:0" if is_gpu_available() else "cpu"
        
        # Get image paths
        image_paths = list(Path(image_folder).glob("*.png"))
        logger.info(f"Found {len(image_paths)} images to process")
        
        all_detections = []
        
        # Process images in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Processing {periodical}"):
            batch_paths = image_paths[i : i + batch_size]
            
            try:
                # Predict on batch of images
                det_res = model.predict(
                    [str(p) for p in batch_paths],
                    imgsz=image_size,
                    conf=conf_threshold,
                    device=device,
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
                logger.error(f"Error processing batch: {str(e)}")
                continue
        
        # Save detection results
        df = pd.DataFrame(all_detections)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved bounding box predictions to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error predicting bounding boxes: {str(e)}")
        raise


def postprocess_bounding_boxes(
    input_path: str,
    output_path: str,
    fill_columns: bool = True,
    width_multiplier: float = 1.5,
    remove_abandon: bool = True
) -> str:
    """
    Post-process the raw bounding boxes.
    
    Args:
        input_path: Path to raw bounding boxes parquet file
        output_path: Path to save processed bounding boxes
        fill_columns: Whether to fill columns
        width_multiplier: Width multiplier for margin adjustment
        remove_abandon: Whether to remove abandoned boxes
        
    Returns:
        Path to the processed bounding box file
    """
    try:
        # Skip if already done
        if os.path.exists(output_path):
            logger.info(f"Bounding box post-processing already completed. Skipping.")
            return output_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load raw bounding boxes
        logger.info(f"Post-processing bounding boxes from {input_path}")
        bbox_df = pd.read_parquet(input_path)
        
        # Add page_id
        bbox_df["page_id"] = bbox_df["filename"].str.replace(".png", "")
        
        # Normalize class names
        bbox_df["class"] = np.where(
            bbox_df["class"] == "plain text", "text", bbox_df["class"]
        )
        
        # Post-process
        bbox_df = postprocess_bbox(
            bbox_df,
            10,
            width_multiplier=width_multiplier,
            remove_abandon=remove_abandon,
            fill_columns=fill_columns,
        )
        
        # Save processed results
        bbox_df.to_parquet(output_path)
        logger.info(f"Saved post-processed bounding boxes to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error post-processing bounding boxes: {str(e)}")
        raise


def prepare_batch_job(
    bbox_path: str,
    image_folder: str,
    output_file: str,
    prompt_dict: Dict[str, str],
    api_key: Optional[str] = None,
    deskew: bool = False,
    max_ratio: float = 1.5
) -> str:
    """
    Prepare and submit batch jobs for OCR processing.
    
    Args:
        bbox_path: Path to processed bounding boxes parquet file
        image_folder: Path to folder containing images
        output_file: Path to save batch job details
        prompt_dict: Dictionary mapping element types to prompts
        api_key: Mistral API key (defaults to environment variable)
        deskew: Whether to deskew images
        max_ratio: Maximum aspect ratio for images
        
    Returns:
        Path to the batch job file
    """
    try:
        # Skip if already done
        if os.path.exists(output_file):
            logger.info(f"Batch job already prepared. Skipping.")
            return output_file
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Get API key
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            err_msg = "No Mistral API key provided. Set MISTRAL_API_KEY environment variable or provide api_key parameter."
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        # Initialize client
        client = Mistral(api_key=api_key)
        
        # Load bounding box dataframe
        logger.info(f"Preparing batch job from {bbox_path}")
        bbox_df = pd.read_parquet(bbox_path)
        
        # Process the data
        logger.info(f"Submitting OCR batch job")
        process_issues_to_jobs(
            bbox_df=bbox_df,
            images_folder=image_folder,
            prompt_dict=prompt_dict,
            client=client,
            output_file=output_file,
            deskew=deskew,
            max_ratio=max_ratio,
        )
        
        logger.info(f"Batch job prepared and submitted. Details saved to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error preparing batch job: {str(e)}")
        raise


def download_batch_results(
    jobs_file: str,
    output_dir: str,
    log_file: str,
    api_key: Optional[str] = None
) -> str:
    """
    Download results from a completed batch job.
    
    Args:
        jobs_file: Path to batch job file
        output_dir: Directory to save downloaded results
        log_file: Path to save download log
        api_key: Mistral API key (defaults to environment variable)
        
    Returns:
        Path to the folder containing downloaded results
    """
    try:
        # Check if files already exist
        if os.path.exists(log_file) and os.path.exists(output_dir) and os.listdir(output_dir):
            logger.info(f"Batch results already downloaded. Skipping.")
            return output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Get API key
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            err_msg = "No Mistral API key provided. Set MISTRAL_API_KEY environment variable or provide api_key parameter."
            logger.error(err_msg)
            raise ValueError(err_msg)
            
        # Initialize client
        client = Mistral(api_key=api_key)
        
        # Download the results
        logger.info(f"Downloading batch results from {jobs_file}")
        results = download_processed_jobs(
            client=client,
            jobs_file=jobs_file,
            output_dir=output_dir,
            log_file=log_file,
        )
        
        logger.info(f"Downloaded batch results to {output_dir}")
        
        return output_dir
    
    except Exception as e:
        logger.error(f"Error downloading batch results: {str(e)}")
        raise


def process_results(
    json_folder: str,
    raw_output: str,
    post_processed_output: str
) -> Dict[str, str]:
    """
    Process downloaded batch results to create dataframes.
    
    Args:
        json_folder: Path to folder containing downloaded JSON files
        raw_output: Path to save raw dataframe
        post_processed_output: Path to save post-processed dataframe
        
    Returns:
        Dictionary with paths to the generated dataframes
    """
    try:
        # Track processing results
        results = {}
        
        # Ensure output directories exist
        for path in [raw_output, post_processed_output]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Step 1: Convert JSONs to raw dataframe
        if not os.path.exists(raw_output):
            logger.info(f"Converting JSONs to dataframe")
            df = process_json_files(
                json_folder=json_folder,
                output_path=raw_output,
                num_workers=None  # Will use CPU count - 1
            )
            logger.info(f"Raw dataframe created and saved to {raw_output}")
        else:
            logger.info(f"Raw dataframe already exists. Skipping.")
        
        results["raw_dataframe"] = raw_output
        
        # Step 2: Post-process the dataframe
        if not os.path.exists(post_processed_output):
            logger.info(f"Post-processing dataframe")
            returned_docs = pd.read_parquet(raw_output)
            
            # Clean up content
            returned_docs["content"] = (
                returned_docs["content"].str.strip("`").str.replace("tsv", "", n=1)
            )
            
            # Fix misclassifications (title -> text for long content)
            returned_docs.loc[
                (returned_docs["completion_tokens"] > 50)
                & (returned_docs["class"] == "title"),
                "class",
            ] = "text"
            
            # Clean line breaks except for tables
            returned_docs.loc[returned_docs["class"] != "table", "content"] = (
                remove_line_breaks(
                    returned_docs.loc[returned_docs["class"] != "table", "content"]
                )
            )
            
            # Save post-processed dataframe
            returned_docs.to_parquet(post_processed_output)
            logger.info(f"Post-processed dataframe saved to {post_processed_output}")
        else:
            logger.info(f"Post-processed dataframe already exists. Skipping.")
        
        results["post_processed_dataframe"] = post_processed_output
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing results: {str(e)}")
        raise


def is_batch_complete(periodical: str, batch_path: str, interactive: bool = True) -> bool:
    """
    Check if a batch job is complete.
    
    Args:
        periodical: Name of the periodical
        batch_path: Path to the batch job file
        interactive: Whether to ask the user interactively
        
    Returns:
        Boolean indicating if batch is complete
    """
    if interactive:
        user_input = input(f"Is the batch job for {periodical} complete? (yes/no): ")
        return user_input.lower() in ["yes", "y"]
    else:
        # In a real implementation, this would check batch status via API
        # For now we'll return False as a placeholder
        return False


def is_gpu_available() -> bool:
    """Check if GPU is available for processing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False