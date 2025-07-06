#!/usr/bin/env python3
"""
Video Processing Service
Author: Youssef Shitiwi
Description: Video processing service for sperm analysis
"""

import cv2
import numpy as np
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Any
import tempfile
import shutil
from datetime import datetime
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.models.tracker import SpermTracker
from training.models.casa_metrics import CASAMetrics, SpermParameters
from backend.models.schemas import AnalysisConfig, AnalysisStatus
from loguru import logger

class VideoProcessor:
    """
    Video processing service for sperm analysis.
    Handles video loading, processing, and result generation.
    """
    
    def __init__(self, 
                 model_path: str = "data/models/best/sperm_detection_best.pt",
                 output_dir: str = "data/results",
                 max_workers: int = 2):
        """
        Initialize video processor.
        
        Args:
            model_path: Path to trained YOLO model
            output_dir: Directory for output files
            max_workers: Maximum number of concurrent processing tasks
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing queue and status tracking
        self.processing_queue: Dict[str, Dict] = {}
        self.processing_status: Dict[str, Dict] = {}
        
        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize tracker and metrics calculator
        self.tracker = None
        self.casa_metrics = None
        
        # Initialize services
        self._initialize_services()
        
        logger.info(f"VideoProcessor initialized with model: {self.model_path}")
    
    def _initialize_services(self):
        """Initialize AI services."""
        try:
            # Check if model exists
            if not self.model_path.exists():
                logger.warning(f"Model not found at {self.model_path}. Using default YOLOv8n.")
                self.model_path = "yolov8n.pt"  # Use default model
            
            # Initialize tracker (will be created per analysis to avoid conflicts)
            logger.info("AI services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI services: {e}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "format": Path(video_path).suffix.lower()
        }
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            info = self.get_video_info(video_path)
            
            # Check basic requirements
            if info["fps"] <= 0 or info["frame_count"] <= 0:
                return False
            
            if info["width"] <= 0 or info["height"] <= 0:
                return False
            
            # Check format
            supported_formats = ['.mp4', '.avi', '.mov']
            if info["format"] not in supported_formats:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            return False
    
    async def start_analysis(self, 
                           analysis_id: str,
                           video_path: str, 
                           config: AnalysisConfig,
                           progress_callback: Optional[Callable] = None) -> str:
        """
        Start video analysis asynchronously.
        
        Args:
            analysis_id: Unique analysis identifier
            video_path: Path to video file
            config: Analysis configuration
            progress_callback: Optional progress callback function
            
        Returns:
            Analysis ID
        """
        # Validate video
        if not self.validate_video(video_path):
            raise ValueError("Invalid video file")
        
        # Get video info
        video_info = self.get_video_info(video_path)
        
        # Create analysis entry
        analysis_data = {
            "analysis_id": analysis_id,
            "video_path": video_path,
            "config": config,
            "video_info": video_info,
            "progress_callback": progress_callback,
            "created_at": datetime.utcnow(),
            "status": AnalysisStatus.PENDING
        }
        
        # Add to queue
        self.processing_queue[analysis_id] = analysis_data
        
        # Update status
        self._update_status(analysis_id, AnalysisStatus.PENDING, "Analysis queued", 0.0)
        
        # Submit to executor
        future = self.executor.submit(self._process_video_sync, analysis_id)
        
        logger.info(f"Analysis {analysis_id} queued for processing")
        return analysis_id
    
    def _update_status(self, 
                      analysis_id: str, 
                      status: AnalysisStatus,
                      message: str,
                      progress: float,
                      current_frame: Optional[int] = None,
                      total_frames: Optional[int] = None,
                      processing_stage: Optional[str] = None,
                      error: Optional[str] = None):
        """Update analysis status."""
        now = datetime.utcnow()
        
        status_data = {
            "analysis_id": analysis_id,
            "status": status,
            "message": message,
            "progress": progress,
            "current_frame": current_frame,
            "total_frames": total_frames,
            "processing_stage": processing_stage,
            "error_message": error,
            "last_updated": now
        }
        
        if analysis_id not in self.processing_status:
            status_data["created_at"] = now
        
        if status == AnalysisStatus.PROCESSING and "started_at" not in self.processing_status.get(analysis_id, {}):
            status_data["started_at"] = now
        
        if status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
            status_data["completed_at"] = now
        
        self.processing_status[analysis_id] = {
            **self.processing_status.get(analysis_id, {}),
            **status_data
        }
    
    def _process_video_sync(self, analysis_id: str) -> Dict[str, Any]:
        """
        Process video synchronously (runs in thread pool).
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Processing results
        """
        try:
            analysis_data = self.processing_queue[analysis_id]
            video_path = analysis_data["video_path"]
            config = analysis_data["config"]
            video_info = analysis_data["video_info"]
            
            logger.info(f"Starting analysis {analysis_id}")
            
            # Update status
            self._update_status(
                analysis_id, 
                AnalysisStatus.PROCESSING, 
                "Initializing analysis", 
                5.0,
                processing_stage="initialization"
            )
            
            # Initialize tracker and metrics for this analysis
            tracker = SpermTracker(
                model_path=str(self.model_path),
                conf_threshold=config.confidence_threshold,
                iou_threshold=config.iou_threshold
            )
            
            casa_metrics = CASAMetrics(
                fps=config.fps,
                pixel_to_micron=config.pixel_to_micron,
                min_track_length=config.min_track_length
            )
            
            # Process video
            results = self._process_video_frames(
                analysis_id, video_path, tracker, casa_metrics, config, video_info
            )
            
            # Generate outputs
            self._generate_outputs(analysis_id, results, config)
            
            # Update final status
            self._update_status(
                analysis_id,
                AnalysisStatus.COMPLETED,
                "Analysis completed successfully",
                100.0,
                processing_stage="completed"
            )
            
            logger.info(f"Analysis {analysis_id} completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            self._update_status(
                analysis_id,
                AnalysisStatus.FAILED,
                f"Analysis failed: {str(e)}",
                0.0,
                error=str(e)
            )
            raise
    
    def _process_video_frames(self, 
                            analysis_id: str,
                            video_path: str, 
                            tracker: SpermTracker,
                            casa_metrics: CASAMetrics,
                            config: AnalysisConfig,
                            video_info: Dict) -> Dict[str, Any]:
        """
        Process video frames for sperm detection and tracking.
        
        Args:
            analysis_id: Analysis identifier
            video_path: Path to video file
            tracker: SpermTracker instance
            casa_metrics: CASAMetrics instance
            config: Analysis configuration
            video_info: Video information
            
        Returns:
            Processing results
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare visualization video if requested
        vis_writer = None
        if config.enable_visualization:
            output_video_path = self.output_dir / f"{analysis_id}_visualization.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vis_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                config.fps,
                (video_info["width"], video_info["height"])
            )
        
        frame_idx = 0
        all_detections = []
        all_tracks = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections, tracks = tracker.process_frame(frame)
                
                # Store results
                all_detections.extend(detections)
                all_tracks.extend(tracks)
                
                # Generate visualization if requested
                if vis_writer is not None:
                    vis_frame = tracker.visualize_tracking(
                        frame, detections, tracks, show_trajectories=True
                    )
                    vis_writer.write(vis_frame)
                
                # Update progress
                progress = (frame_idx / total_frames) * 80  # Reserve 20% for post-processing
                self._update_status(
                    analysis_id,
                    AnalysisStatus.PROCESSING,
                    f"Processing frame {frame_idx}/{total_frames}",
                    progress,
                    current_frame=frame_idx,
                    total_frames=total_frames,
                    processing_stage="frame_processing"
                )
                
                frame_idx += 1
                
                # Check if analysis was cancelled (optional feature)
                if self.processing_status.get(analysis_id, {}).get("status") == AnalysisStatus.CANCELLED:
                    break
            
        finally:
            cap.release()
            if vis_writer is not None:
                vis_writer.release()
        
        # Post-processing: Calculate CASA metrics
        self._update_status(
            analysis_id,
            AnalysisStatus.PROCESSING,
            "Calculating CASA metrics",
            85.0,
            processing_stage="casa_calculation"
        )
        
        # Get all trajectories
        trajectories = tracker.get_all_trajectories()
        
        # Analyze tracks
        sperm_parameters = casa_metrics.analyze_multiple_tracks(trajectories)
        
        # Calculate population statistics
        population_stats = casa_metrics.calculate_population_statistics(sperm_parameters)
        
        # Prepare results
        results = {
            "analysis_id": analysis_id,
            "video_info": video_info,
            "config": config,
            "total_detections": len(all_detections),
            "total_tracks": len(set(track.track_id for track in all_tracks)),
            "analyzed_sperm": len(sperm_parameters),
            "sperm_parameters": sperm_parameters,
            "population_statistics": population_stats,
            "trajectories": trajectories,
            "processing_time": time.time() - self.processing_status[analysis_id]["started_at"].timestamp()
        }
        
        return results
    
    def _generate_outputs(self, analysis_id: str, results: Dict[str, Any], config: AnalysisConfig):
        """
        Generate output files from analysis results.
        
        Args:
            analysis_id: Analysis identifier
            results: Analysis results
            config: Analysis configuration
        """
        self._update_status(
            analysis_id,
            AnalysisStatus.PROCESSING,
            "Generating output files",
            90.0,
            processing_stage="output_generation"
        )
        
        # Create output directory for this analysis
        analysis_output_dir = self.output_dir / analysis_id
        analysis_output_dir.mkdir(exist_ok=True)
        
        # Save results as JSON
        json_path = analysis_output_dir / "results.json"
        self._save_results_json(results, json_path)
        
        # Generate CSV export
        csv_path = analysis_output_dir / "sperm_analysis.csv"
        self._save_results_csv(results, csv_path, config.export_trajectories)
        
        # Save population statistics
        stats_path = analysis_output_dir / "population_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(results["population_statistics"], f, indent=2)
        
        # Save trajectories if requested
        if config.export_trajectories:
            trajectories_path = analysis_output_dir / "trajectories.json"
            with open(trajectories_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_trajectories = {}
                for track_id, trajectory in results["trajectories"].items():
                    serializable_trajectories[str(track_id)] = trajectory
                json.dump(serializable_trajectories, f, indent=2)
        
        # Update results with file paths
        results["output_files"] = {
            "json_results": str(json_path),
            "csv_export": str(csv_path),
            "statistics": str(stats_path),
            "trajectories": str(trajectories_path) if config.export_trajectories else None,
            "visualization": str(self.output_dir / f"{analysis_id}_visualization.mp4") if config.enable_visualization else None
        }
    
    def _save_results_json(self, results: Dict[str, Any], output_path: Path):
        """Save results as JSON file."""
        # Prepare serializable results
        serializable_results = {
            "analysis_id": results["analysis_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "video_info": results["video_info"],
            "config": results["config"].dict() if hasattr(results["config"], 'dict') else results["config"],
            "summary": {
                "total_detections": results["total_detections"],
                "total_tracks": results["total_tracks"],
                "analyzed_sperm": results["analyzed_sperm"],
                "processing_time": results["processing_time"]
            },
            "population_statistics": results["population_statistics"],
            "individual_sperm": []
        }
        
        # Add individual sperm data
        for sperm in results["sperm_parameters"]:
            if hasattr(sperm, '__dict__'):
                sperm_dict = sperm.__dict__
            else:
                sperm_dict = sperm
            serializable_results["individual_sperm"].append(sperm_dict)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _save_results_csv(self, results: Dict[str, Any], output_path: Path, include_trajectories: bool = False):
        """Save results as CSV file."""
        from training.models.casa_metrics import CASAMetrics
        
        # Create temporary CASA metrics instance for CSV export
        casa_metrics = CASAMetrics()
        casa_metrics.export_results_to_csv(
            results["sperm_parameters"],
            str(output_path),
            include_trajectories=include_trajectories
        )
    
    def get_analysis_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis status.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Status information or None if not found
        """
        return self.processing_status.get(analysis_id)
    
    def get_analysis_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis results.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Results or None if not available
        """
        status = self.get_analysis_status(analysis_id)
        if not status or status["status"] != AnalysisStatus.COMPLETED:
            return None
        
        # Load results from file
        results_path = self.output_dir / analysis_id / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def cancel_analysis(self, analysis_id: str) -> bool:
        """
        Cancel an ongoing analysis.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            True if cancelled, False if not possible
        """
        status = self.get_analysis_status(analysis_id)
        if not status:
            return False
        
        if status["status"] in [AnalysisStatus.PENDING, AnalysisStatus.PROCESSING]:
            self._update_status(
                analysis_id,
                AnalysisStatus.CANCELLED,
                "Analysis cancelled by user",
                0.0
            )
            return True
        
        return False
    
    def cleanup_analysis(self, analysis_id: str, keep_results: bool = True):
        """
        Clean up analysis data.
        
        Args:
            analysis_id: Analysis identifier
            keep_results: Whether to keep result files
        """
        # Remove from processing queue
        self.processing_queue.pop(analysis_id, None)
        
        if not keep_results:
            # Remove output directory
            analysis_output_dir = self.output_dir / analysis_id
            if analysis_output_dir.exists():
                shutil.rmtree(analysis_output_dir)
        
        # Remove from status tracking after some time (optional)
        # Could implement a cleanup scheduler
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall processing queue status."""
        pending = sum(1 for status in self.processing_status.values() 
                     if status["status"] == AnalysisStatus.PENDING)
        processing = sum(1 for status in self.processing_status.values() 
                        if status["status"] == AnalysisStatus.PROCESSING)
        completed = sum(1 for status in self.processing_status.values() 
                       if status["status"] == AnalysisStatus.COMPLETED)
        failed = sum(1 for status in self.processing_status.values() 
                    if status["status"] == AnalysisStatus.FAILED)
        
        return {
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "total": len(self.processing_status)
        }