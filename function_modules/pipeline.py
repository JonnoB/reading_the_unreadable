"""
Pipeline for end-to-end document processing.

This module provides a comprehensive pipeline that orchestrates all steps
of the document processing workflow, from bounding box prediction through 
batching, OCR, and article creation.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, TypedDict
from datetime import datetime

class PipelineConfig(TypedDict):
    base_dir: str
    input_dir: str
    image_dir: str
    output_dir: str
    model_repo: str
    model_file: str
    bbox_raw_dir: str
    bbox_processed_dir: str
    processed_jobs_dir: str
    download_jobs_dir: str
    default_image_size: int
    batch_size: int
    conf_threshold: float
    fill_columns: bool
    deskew: bool
    max_ratio: float
    prompts: Dict[str, str]

class PipelineState(TypedDict):
    pipeline_id: str
    current_stage: str
    completed_stages: List[str]
    periodicals: Dict[str, Any]
    timestamp: str

# Import pipeline stages module
import function_modules.pipeline_stages as pipeline_stages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewspaperPipeline:
    config: PipelineConfig
    state: PipelineState
    """
    Pipeline for processing historical newspaper documents.
    
    This class orchestrates the workflow from layout analysis through OCR and article creation,
    handling the distinct stages of the process including the long-running batch operations.
    
    It focuses on state management and workflow coordination, delegating actual processing
    to the specialized functions in pipeline_stages.py.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to a JSON configuration file (optional)
        """
        # Set default paths
        self.config = {
            "base_dir": os.getcwd(),
            "input_dir": "data/input",
            "image_dir": "data/images",
            "output_dir": "data/output",
            "model_repo": "juliozhao/DocLayout-YOLO-DocStructBench",
            "model_file": "doclayout_yolo_docstructbench_imgsz1024.pt",
            "bbox_raw_dir": "data/periodical_bboxes/raw",
            "bbox_processed_dir": "data/periodical_bboxes/post_process",
            "processed_jobs_dir": "data/processed_jobs",
            "download_jobs_dir": "data/download_jobs",
            "default_image_size": 1056,
            "batch_size": 64,
            "conf_threshold": 0.2,
            "fill_columns": True,
            "deskew": False,
            "max_ratio": 1.5,
            "prompts": {
                "text": "The text in the image is from a 19th century English newspaper, please transcribe the text including linebreaks. Do not use markdown use plain text only. Do not add any commentary.",
                "figure": "Please describe the graphic taken from a 19th century English newspaper. Do not add additional commentary",
                "table": "Please extract the table from the image taken from a 19th century English newspaper as a tab separated values (tsv) text file. Do not add any commentary"
            }
        }
        
        # Load configuration from file if provided
        if config_path:
            self._load_config(config_path)
            
        # Create necessary directories
        self._create_directories()
        
        # Track pipeline state
        self.state = {
            "pipeline_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "current_stage": "initialized",
            "completed_stages": [],
            "periodicals": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Save initial state
        self._save_state()
        
        logger.info(f"Pipeline initialized with ID: {self.state['pipeline_id']}")
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _create_directories(self) -> None:
        """Create all necessary directories for the pipeline."""
        dirs_to_create: List[str] = [
            self.config["output_dir"],
            self.config["bbox_raw_dir"],
            self.config["bbox_processed_dir"],
            self.config["processed_jobs_dir"],
            self.config["download_jobs_dir"],
            os.path.join(self.config["download_jobs_dir"], "dataframes"),
            os.path.join(self.config["download_jobs_dir"], "dataframes", "raw"),
            os.path.join(self.config["download_jobs_dir"], "dataframes", "post_processed")
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            
        logger.info("Created necessary directories")
    
    def _save_state(self) -> None:
        """Save the current pipeline state to a file."""
        state_dir = os.path.join(self.config["output_dir"], "pipeline_state")
        os.makedirs(state_dir, exist_ok=True)
        
        state_path = os.path.join(state_dir, f"{self.state['pipeline_id']}.json")
        self.state["timestamp"] = datetime.now().isoformat()
        
        with open(state_path, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        logger.debug(f"Saved pipeline state to {state_path}")
    
    def load_state(self, pipeline_id: str) -> bool:
        """
        Load a previously saved pipeline state.
        
        Args:
            pipeline_id: ID of the pipeline state to load
            
        Returns:
            True if state was loaded successfully, False otherwise
        """
        state_dir = os.path.join(self.config["output_dir"], "pipeline_state")
        state_path = os.path.join(state_dir, f"{pipeline_id}.json")
        
        if not os.path.exists(state_path):
            logger.error(f"State file not found: {state_path}")
            return False
        
        try:
            with open(state_path, 'r') as f:
                self.state = json.load(f)
            logger.info(f"Loaded pipeline state: {pipeline_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            return False
    
    def _update_stage(self, stage: str) -> None:
        """
        Update the current pipeline stage and completed stages list.
        
        Args:
            stage: New stage name to set as current
        
        Note:
            Also adds the previous stage to completed_stages if not already present
        """
        if self.state["current_stage"] not in self.state["completed_stages"]:
            self.state["completed_stages"].append(self.state["current_stage"])
        
        self.state["current_stage"] = stage
        self._save_state()
        logger.info(f"Pipeline stage updated to: {stage}")
    
    def predict_bounding_boxes(self, 
                              periodical: str,
                              image_folder: str,
                              image_size: Optional[int] = None,
                              batch_size: Optional[int] = None) -> str:
        """
        Run layout detection to predict bounding boxes.
        
        Args:
            periodical: Name of the periodical to process
            image_folder: Path to folder containing the images
            image_size: Size to resize images to (defaults to config value)
            batch_size: Processing batch size (defaults to config value)
            
        Returns:
            Path to the output parquet file
        """
        self._update_stage("predicting_boxes")
        
        # Prepare output path
        img_size: int = image_size or self.config["default_image_size"]
        output_file: str = f"{periodical}_{img_size}.parquet"
        output_path: str = os.path.join(self.config["bbox_raw_dir"], output_file)
        
        # Delegate to the pipeline stage function
        result_path = pipeline_stages.predict_bounding_boxes(
            periodical=periodical,
            image_folder=image_folder,
            output_path=output_path,
            model_repo=self.config["model_repo"],
            model_file=self.config["model_file"],
            image_size=img_size,
            batch_size=batch_size or self.config["batch_size"],
            conf_threshold=self.config["conf_threshold"]
        )
        
        # Update pipeline state
        if periodical not in self.state["periodicals"]:
            self.state["periodicals"][periodical] = {}
        self.state["periodicals"][periodical]["bbox_raw_path"] = result_path
        self._save_state()
        
        return result_path
    
    def postprocess_bounding_boxes(self, periodical: str, fill_columns: Optional[bool] = None) -> str:
        """
        Post-process the raw bounding boxes.
        
        Args:
            periodical: Name of the periodical to process
            fill_columns: Whether to fill columns (defaults to config value)
            
        Returns:
            Path to the processed bounding box file
        """
        self._update_stage("postprocessing_boxes")
        
        # Verify prerequisites
        if periodical not in self.state["periodicals"] or "bbox_raw_path" not in self.state["periodicals"][periodical]:
            raise ValueError(f"No raw bounding boxes found for {periodical}. Run predict_bounding_boxes first.")
        
        # Prepare paths
        raw_path: str = self.state["periodicals"][periodical]["bbox_raw_path"]
        fill_cols: bool = fill_columns if fill_columns is not None else self.config["fill_columns"]
        output_folder: str = self.config["bbox_processed_dir"] + "_fill" if fill_cols else self.config["bbox_processed_dir"]
        os.makedirs(output_folder, exist_ok=True)
        output_file: str = os.path.basename(raw_path)
        output_path: str = os.path.join(output_folder, output_file)
        
        # Delegate to the pipeline stage function
        result_path = pipeline_stages.postprocess_bounding_boxes(
            input_path=raw_path,
            output_path=output_path,
            fill_columns=fill_cols,
            width_multiplier=1.5,
            remove_abandon=True
        )
        
        # Update pipeline state
        self.state["periodicals"][periodical]["bbox_processed_path"] = result_path
        self._save_state()
        
        return result_path
    
    def prepare_batch_job(self, 
                         periodical: str, 
                         image_folder: str,
                         api_key: Optional[str] = None) -> str:
        """
        Prepare and submit batch jobs for OCR processing.
        
        Args:
            periodical: Name of the periodical to process
            image_folder: Path to folder containing images
            api_key: Mistral API key (defaults to environment variable)
            
        Returns:
            Path to the batch job file
        """
        self._update_stage("preparing_batch")
        
        # Verify prerequisites
        if periodical not in self.state["periodicals"] or "bbox_processed_path" not in self.state["periodicals"][periodical]:
            raise ValueError(f"No processed bounding boxes found for {periodical}. Run postprocess_bounding_boxes first.")
        
        # Prepare paths
        bbox_path: str = self.state["periodicals"][periodical]["bbox_processed_path"]
        output_folder: str = self.config["processed_jobs_dir"]
        os.makedirs(output_folder, exist_ok=True)
        output_file: str = os.path.join(output_folder, f"{periodical}.csv")
        
        # Delegate to the pipeline stage function
        result_path = pipeline_stages.prepare_batch_job(
            bbox_path=bbox_path,
            image_folder=image_folder,
            output_file=output_file,
            prompt_dict=self.config["prompts"],
            api_key=api_key,
            deskew=self.config["deskew"],
            max_ratio=self.config["max_ratio"]
        )
        
        # Update pipeline state
        self.state["periodicals"][periodical]["batch_job_path"] = result_path
        self._save_state()
        
        return result_path
    
    def download_batch_results(self, periodical: str, api_key: Optional[str] = None) -> str:
        """
        Download results from a completed batch job.
        
        Args:
            periodical: Name of the periodical to download
            api_key: Mistral API key (defaults to environment variable)
            
        Returns:
            Path to the folder containing downloaded results
        """
        self._update_stage("downloading_results")
        
        # Verify prerequisites
        if periodical not in self.state["periodicals"] or "batch_job_path" not in self.state["periodicals"][periodical]:
            raise ValueError(f"No batch job found for {periodical}. Run prepare_batch_job first.")
        
        # Prepare paths
        jobs_file: str = self.state["periodicals"][periodical]["batch_job_path"]
        download_folder: str = self.config["download_jobs_dir"]
        json_folder: str = os.path.join(download_folder, periodical)
        os.makedirs(json_folder, exist_ok=True)
        log_file: str = os.path.join(download_folder, f"{periodical}.csv")
        
        # Delegate to the pipeline stage function
        result_folder = pipeline_stages.download_batch_results(
            jobs_file=jobs_file,
            output_dir=json_folder,
            log_file=log_file,
            api_key=api_key
        )
        
        # Update pipeline state
        self.state["periodicals"][periodical]["download_folder"] = result_folder
        self.state["periodicals"][periodical]["download_log"] = log_file
        self._save_state()
        
        return result_folder
    
    def process_results(self, periodical: str) -> Dict[str, str]:
        """
        Process downloaded batch results to create dataframes.
        
        Args:
            periodical: Name of the periodical to process
            
        Returns:
            Dictionary with paths to the generated dataframes
        """
        self._update_stage("processing_results")
        
        # Verify prerequisites
        if periodical not in self.state["periodicals"] or "download_folder" not in self.state["periodicals"][periodical]:
            raise ValueError(f"No downloaded results found for {periodical}. Run download_batch_results first.")
        
        # Prepare paths
        json_folder: str = self.state["periodicals"][periodical]["download_folder"]
        dataframe_folder: str = os.path.join(self.config["download_jobs_dir"], "dataframes")
        raw_output: str = os.path.join(dataframe_folder, "raw", f"{periodical}.parquet")
        post_processed_output: str = os.path.join(dataframe_folder, "post_processed", f"{periodical}.parquet")
        
        # Delegate to the pipeline stage function
        result_paths = pipeline_stages.process_results(
            json_folder=json_folder,
            raw_output=raw_output,
            post_processed_output=post_processed_output
        )
        
        # Update pipeline state
        self.state["periodicals"][periodical].update({
            "raw_dataframe": result_paths["raw_dataframe"],
            "post_processed_dataframe": result_paths["post_processed_dataframe"]
        })
        self._save_state()
        
        return result_paths
    
    def run_pipeline(self, 
                    periodical: str, 
                    image_folder: str,
                    api_key: Optional[str] = None,
                    wait_for_batch: bool = False,
                    check_interval: int = 3600) -> Dict[str, Any]:
        """
        Run the complete pipeline for a periodical.
        
        Args:
            periodical: Name of the periodical to process
            image_folder: Path to folder containing images
            api_key: Mistral API key (defaults to environment variable)
            wait_for_batch: Whether to wait for batch completion
            check_interval: Seconds between batch status checks if waiting
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Starting pipeline for {periodical}")
        
        # Run each pipeline stage in sequence
        bbox_path = self.predict_bounding_boxes(periodical, image_folder)
        processed_bbox_path = self.postprocess_bounding_boxes(periodical)
        batch_job_path = self.prepare_batch_job(periodical, image_folder, api_key)
        
        # If not waiting for batch, stop here
        if not wait_for_batch:
            logger.info(f"Batch job submitted for {periodical}, but not waiting for completion.")
            logger.info(f"To resume pipeline, run download_batch_results and process_results when batch is complete.")
            return {
                "status": "batch_submitted",
                "periodical": periodical,
                "batch_job_path": batch_job_path
            }
        
        # Wait for batch to complete
        logger.info(f"Waiting for batch job to complete. Checking every {check_interval} seconds.")
        batch_complete = False
        while not batch_complete:
            # Check if batch is complete
            batch_complete = pipeline_stages.is_batch_complete(periodical, batch_job_path, interactive=True)
            if not batch_complete:
                logger.info(f"Batch job still in progress. Waiting {check_interval} seconds.")
                time.sleep(check_interval)
        
        logger.info("Batch job reported as complete.")
        
        # Complete the pipeline
        download_folder = self.download_batch_results(periodical, api_key)
        result_paths = self.process_results(periodical)
        
        logger.info(f"Pipeline completed for {periodical}")
        return {
            "status": "complete",
            "periodical": periodical,
            "result_paths": result_paths
        }
    
    def resume_pipeline(self, 
                       periodical: str, 
                       api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume pipeline processing after batch job completion.
        
        Args:
            periodical: Name of the periodical to process
            api_key: Mistral API key (defaults to environment variable)
            
        Returns:
            Dictionary with pipeline results
        """
        # Verify prerequisites
        if periodical not in self.state["periodicals"]:
            raise ValueError(f"No state found for {periodical}. Cannot resume pipeline.")
        
        # Determine current stage and resume appropriately
        periodical_state = self.state["periodicals"][periodical]
        
        # Logic for determining where to resume
        if "download_folder" not in periodical_state:
            logger.info(f"Resuming pipeline for {periodical} at download_batch_results stage")
            download_folder = self.download_batch_results(periodical, api_key)
            result_paths = self.process_results(periodical)
        elif "raw_dataframe" not in periodical_state:
            logger.info(f"Resuming pipeline for {periodical} at process_results stage")
            result_paths = self.process_results(periodical)
        else:
            logger.info(f"Pipeline already completed for {periodical}")
            result_paths = {
                "raw_dataframe": periodical_state["raw_dataframe"],
                "post_processed_dataframe": periodical_state["post_processed_dataframe"]
            }
        
        return {
            "status": "complete",
            "periodical": periodical,
            "result_paths": result_paths
        }


# Helper function to create a pipeline instance
def create_pipeline(config_path: Optional[str] = None) -> NewspaperPipeline:
    """
    Create a new pipeline instance.
    
    Args:
        config_path: Optional path to a JSON configuration file
        
    Returns:
        Initialized NewspaperPipeline instance
    """
    return NewspaperPipeline(config_path)