#!/usr/bin/env python3

import os
import glob
import json
from function_modules.pipeline import NewspaperPipeline
from typing import Optional, Tuple

def find_latest_pipeline_state() -> Tuple[Optional[str], Optional[dict]]:
    """
    Find the most recent pipeline state file.
    
    Returns:
        Tuple of (pipeline_id, state_data) or (None, None) if no state exists
    """
    state_dir = os.path.join('data', 'output', 'pipeline_state')
    if not os.path.exists(state_dir):
        return None, None
        
    state_files = glob.glob(os.path.join(state_dir, 'pipeline_*.json'))
    if not state_files:
        return None, None
        
    # Get the most recent state file
    latest_state = max(state_files, key=os.path.getmtime)
    pipeline_id = os.path.splitext(os.path.basename(latest_state))[0]
    
    try:
        with open(latest_state, 'r') as f:
            state_data = json.load(f)
        return pipeline_id, state_data
    except Exception:
        return None, None

def get_or_create_pipeline() -> Tuple[NewspaperPipeline, bool]:
    """
    Get existing pipeline or create a new one.
    
    Returns:
        Tuple of (pipeline, is_new_pipeline)
    """
    pipeline_id, state_data = find_latest_pipeline_state()
    
    # Create a new pipeline instance
    pipeline = NewspaperPipeline()
    
    if pipeline_id and state_data:
        # Override the new state with the loaded state
        pipeline.state = state_data
        return pipeline, False
        
    return pipeline, True

def process_folder(folder_path: str, api_key: Optional[str] = None, max_workers: int = 4) -> None:
    """
    Process all newspaper images in the specified folder.
    
    Args:
        folder_path: Path to the folder containing newspaper images
        api_key: Optional Mistral API key (defaults to environment variable)
        max_workers: Number of parallel workers for batch processing (default: 4)
    """
    # Get or create pipeline
    pipeline, is_new_pipeline = get_or_create_pipeline()
    
    # Use the folder name as the periodical name
    periodical = os.path.basename(folder_path)
    
    # Update the max_workers in the pipeline config
    pipeline.config["max_workers"] = max_workers
    
    if not is_new_pipeline:
        print(f"Resuming pipeline {pipeline.state['pipeline_id']}")
        try:
            # Try to resume the pipeline
            result = pipeline.resume_pipeline(periodical, api_key)
            print(f"Pipeline resumed and completed with status: {result['status']}")
            return
        except Exception as e:
            print(f"Failed to resume pipeline: {str(e)}")
            print("Starting new pipeline run...")
            # Create a fresh pipeline for the new run
            pipeline = NewspaperPipeline()
            # Set max_workers in the fresh pipeline
            pipeline.config["max_workers"] = max_workers
    
    # Start a new pipeline run
    pipeline.run_pipeline(
        periodical=periodical,
        image_folder=folder_path,
        api_key=api_key,
        wait_for_batch=False
    )
    
    print(f"Pipeline started for images in {folder_path}")
    print("The script has terminated. Run again later to continue processing from the last checkpoint.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process newspaper images using the pipeline")
    parser.add_argument("folder", help="Path to the folder containing newspaper images")
    parser.add_argument("--api-key", help="Mistral API key (optional, defaults to environment variable)")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Number of parallel workers for batch processing (default: 4)")
    
    args = parser.parse_args()
    
    # Ensure the folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        exit(1)
        
    process_folder(args.folder, args.api_key, args.max_workers)
