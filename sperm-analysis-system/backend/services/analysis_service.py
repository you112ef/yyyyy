#!/usr/bin/env python3
"""
Analysis Service
Author: Youssef Shitiwi
Description: High-level analysis service for sperm analysis API
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import aiofiles
import tempfile
import shutil
from loguru import logger

from backend.services.video_processor import VideoProcessor
from backend.models.schemas import (
    AnalysisConfig, AnalysisResponse, AnalysisStatus, 
    StatusResponse, ResultsResponse, AnalysisResults
)

class AnalysisService:
    """
    High-level analysis service that coordinates video processing.
    """
    
    def __init__(self, video_processor: VideoProcessor):
        """
        Initialize analysis service.
        
        Args:
            video_processor: VideoProcessor instance
        """
        self.video_processor = video_processor
        self.upload_dir = Path("data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Store uploaded files temporarily
        self.uploaded_files: Dict[str, Dict] = {}
        
        logger.info("AnalysisService initialized")
    
    async def upload_video(self, file_content: bytes, filename: str) -> str:
        """
        Upload and save video file.
        
        Args:
            file_content: Video file content
            filename: Original filename
            
        Returns:
            Upload ID
        """
        # Generate upload ID
        upload_id = str(uuid.uuid4())
        
        # Determine file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in ['.mp4', '.avi', '.mov']:
            raise ValueError(f"Unsupported video format: {file_ext}")
        
        # Save file
        upload_path = self.upload_dir / f"{upload_id}{file_ext}"
        
        async with aiofiles.open(upload_path, 'wb') as f:
            await f.write(file_content)
        
        # Validate video
        if not self.video_processor.validate_video(str(upload_path)):
            upload_path.unlink()  # Delete invalid file
            raise ValueError("Invalid video file")
        
        # Get video info
        video_info = self.video_processor.get_video_info(str(upload_path))
        
        # Store upload information
        self.uploaded_files[upload_id] = {
            "upload_id": upload_id,
            "original_filename": filename,
            "file_path": str(upload_path),
            "file_size": len(file_content),
            "video_info": video_info,
            "uploaded_at": datetime.utcnow()
        }
        
        logger.info(f"Video uploaded: {filename} -> {upload_id}")
        return upload_id
    
    async def start_analysis(self, 
                           upload_id: str, 
                           config: AnalysisConfig,
                           analysis_name: Optional[str] = None) -> AnalysisResponse:
        """
        Start video analysis.
        
        Args:
            upload_id: Video upload ID
            config: Analysis configuration
            analysis_name: Optional analysis name
            
        Returns:
            Analysis response
        """
        # Check if upload exists
        if upload_id not in self.uploaded_files:
            raise ValueError(f"Upload not found: {upload_id}")
        
        upload_info = self.uploaded_files[upload_id]
        video_path = upload_info["file_path"]
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Estimate processing time (rough calculation)
        video_duration = upload_info["video_info"]["duration"]
        estimated_time = video_duration * 2  # Rough estimate: 2x video duration
        
        try:
            # Start analysis
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: asyncio.create_task(
                    self._start_analysis_async(analysis_id, video_path, config)
                )
            )
            
            response = AnalysisResponse(
                analysis_id=analysis_id,
                status=AnalysisStatus.PENDING,
                message="Analysis queued for processing",
                estimated_processing_time=estimated_time,
                created_at=datetime.utcnow()
            )
            
            logger.info(f"Analysis started: {analysis_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to start analysis: {e}")
            raise ValueError(f"Failed to start analysis: {str(e)}")
    
    async def _start_analysis_async(self, analysis_id: str, video_path: str, config: AnalysisConfig):
        """Start analysis asynchronously."""
        try:
            await self.video_processor.start_analysis(analysis_id, video_path, config)
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            # The video processor will handle status updates
    
    def get_analysis_status(self, analysis_id: str) -> Optional[StatusResponse]:
        """
        Get analysis status.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Status response or None if not found
        """
        status_data = self.video_processor.get_analysis_status(analysis_id)
        if not status_data:
            return None
        
        response = StatusResponse(
            analysis_id=analysis_id,
            status=status_data["status"],
            progress=status_data.get("progress", 0.0),
            message=status_data["message"],
            created_at=status_data["created_at"],
            started_at=status_data.get("started_at"),
            completed_at=status_data.get("completed_at"),
            error_message=status_data.get("error_message"),
            current_frame=status_data.get("current_frame"),
            total_frames=status_data.get("total_frames"),
            processing_stage=status_data.get("processing_stage")
        )
        
        return response
    
    def get_analysis_results(self, analysis_id: str) -> Optional[ResultsResponse]:
        """
        Get analysis results.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Results response or None if not available
        """
        status_data = self.video_processor.get_analysis_status(analysis_id)
        if not status_data:
            return None
        
        status = status_data["status"]
        results_data = None
        available_downloads = []
        
        if status == AnalysisStatus.COMPLETED:
            # Get results
            results_data = self.video_processor.get_analysis_results(analysis_id)
            
            # Check available download files
            output_dir = Path(self.video_processor.output_dir) / analysis_id
            if output_dir.exists():
                if (output_dir / "sperm_analysis.csv").exists():
                    available_downloads.append("csv")
                if (output_dir / "results.json").exists():
                    available_downloads.append("json")
                if (output_dir / "population_statistics.json").exists():
                    available_downloads.append("statistics")
                if (output_dir / "trajectories.json").exists():
                    available_downloads.append("trajectories")
                
                # Check for visualization video
                vis_video = Path(self.video_processor.output_dir) / f"{analysis_id}_visualization.mp4"
                if vis_video.exists():
                    available_downloads.append("visualization")
        
        # Convert results to AnalysisResults schema if available
        analysis_results = None
        if results_data:
            analysis_results = self._convert_to_analysis_results(results_data)
        
        response = ResultsResponse(
            analysis_id=analysis_id,
            status=status,
            results=analysis_results,
            available_downloads=available_downloads
        )
        
        return response
    
    def _convert_to_analysis_results(self, results_data: Dict) -> AnalysisResults:
        """Convert raw results to AnalysisResults schema."""
        # This is a simplified conversion - in a real implementation,
        # you would properly map all the fields according to your schema
        
        from backend.models.schemas import (
            AnalysisResults, SpermParameters, PopulationStatistics, AnalysisConfig
        )
        
        # Convert individual sperm data
        individual_sperm = []
        for sperm_data in results_data.get("individual_sperm", []):
            # Map the data to SpermParameters schema
            # This is simplified - you'd need to properly map all fields
            individual_sperm.append(sperm_data)
        
        # Create AnalysisResults object
        analysis_results = AnalysisResults(
            analysis_id=results_data["analysis_id"],
            analysis_name=results_data.get("analysis_name"),
            timestamp=datetime.fromisoformat(results_data["timestamp"]),
            video_filename=results_data["video_info"].get("original_filename", "unknown"),
            video_duration=results_data["video_info"]["duration"],
            total_frames=results_data["video_info"]["frame_count"],
            fps=results_data["video_info"]["fps"],
            config=AnalysisConfig(**results_data["config"]),
            individual_sperm=individual_sperm,
            population_statistics=results_data["population_statistics"],
            processing_time=results_data["summary"]["processing_time"],
            model_version="YOLOv8+DeepSORT v1.0",
            visualization_video=results_data.get("output_files", {}).get("visualization"),
            csv_export=results_data.get("output_files", {}).get("csv_export"),
            json_export=results_data.get("output_files", {}).get("json_results")
        )
        
        return analysis_results
    
    def get_download_path(self, analysis_id: str, format_type: str) -> Optional[Path]:
        """
        Get download path for specific format.
        
        Args:
            analysis_id: Analysis ID
            format_type: Format type (csv, json, statistics, trajectories, visualization)
            
        Returns:
            File path or None if not available
        """
        output_dir = Path(self.video_processor.output_dir) / analysis_id
        
        format_map = {
            "csv": "sperm_analysis.csv",
            "json": "results.json", 
            "statistics": "population_statistics.json",
            "trajectories": "trajectories.json"
        }
        
        if format_type == "visualization":
            vis_path = Path(self.video_processor.output_dir) / f"{analysis_id}_visualization.mp4"
            return vis_path if vis_path.exists() else None
        
        if format_type in format_map:
            file_path = output_dir / format_map[format_type]
            return file_path if file_path.exists() else None
        
        return None
    
    def cancel_analysis(self, analysis_id: str) -> bool:
        """
        Cancel an analysis.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            True if cancelled successfully
        """
        return self.video_processor.cancel_analysis(analysis_id)
    
    def cleanup_upload(self, upload_id: str):
        """
        Clean up uploaded file.
        
        Args:
            upload_id: Upload ID
        """
        if upload_id in self.uploaded_files:
            upload_info = self.uploaded_files[upload_id]
            file_path = Path(upload_info["file_path"])
            
            # Delete file
            if file_path.exists():
                file_path.unlink()
            
            # Remove from tracking
            del self.uploaded_files[upload_id]
            
            logger.info(f"Upload cleaned up: {upload_id}")
    
    def cleanup_analysis(self, analysis_id: str, keep_results: bool = True):
        """
        Clean up analysis data.
        
        Args:
            analysis_id: Analysis ID
            keep_results: Whether to keep result files
        """
        self.video_processor.cleanup_analysis(analysis_id, keep_results)
        logger.info(f"Analysis cleaned up: {analysis_id}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        queue_status = self.video_processor.get_queue_status()
        
        stats = {
            "total_uploads": len(self.uploaded_files),
            "processing_queue": queue_status,
            "disk_usage": self._calculate_disk_usage()
        }
        
        return stats
    
    def _calculate_disk_usage(self) -> Dict[str, float]:
        """Calculate disk usage for uploads and results."""
        upload_size = 0
        results_size = 0
        
        # Calculate upload directory size
        if self.upload_dir.exists():
            for file_path in self.upload_dir.rglob('*'):
                if file_path.is_file():
                    upload_size += file_path.stat().st_size
        
        # Calculate results directory size
        results_dir = Path(self.video_processor.output_dir)
        if results_dir.exists():
            for file_path in results_dir.rglob('*'):
                if file_path.is_file():
                    results_size += file_path.stat().st_size
        
        return {
            "uploads_mb": upload_size / (1024 * 1024),
            "results_mb": results_size / (1024 * 1024),
            "total_mb": (upload_size + results_size) / (1024 * 1024)
        }