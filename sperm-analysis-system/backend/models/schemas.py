#!/usr/bin/env python3
"""
Pydantic Models for API Schemas
Author: Youssef Shitiwi
Description: Data models and validation schemas for the API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

class AnalysisStatus(str, Enum):
    """Analysis status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MotilityClass(str, Enum):
    """Sperm motility classification."""
    PROGRESSIVE = "progressive"
    SLOW_PROGRESSIVE = "slow_progressive"
    NON_PROGRESSIVE = "non_progressive"
    IMMOTILE = "immotile"

class VideoFormat(str, Enum):
    """Supported video formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"

class ExportFormat(str, Enum):
    """Export format options."""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"

# Request Models
class AnalysisRequest(BaseModel):
    """Analysis request model."""
    analysis_name: Optional[str] = Field(None, description="Optional name for the analysis")
    fps: Optional[float] = Field(30.0, ge=1.0, le=120.0, description="Video frame rate")
    pixel_to_micron: Optional[float] = Field(1.0, gt=0.0, description="Pixel to micron conversion factor")
    confidence_threshold: Optional[float] = Field(0.3, ge=0.1, le=1.0, description="Detection confidence threshold")
    iou_threshold: Optional[float] = Field(0.5, ge=0.1, le=1.0, description="IoU threshold for NMS")
    min_track_length: Optional[int] = Field(10, ge=3, description="Minimum track length for analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_name": "Sample Analysis",
                "fps": 30.0,
                "pixel_to_micron": 0.5,
                "confidence_threshold": 0.3,
                "iou_threshold": 0.5,
                "min_track_length": 10
            }
        }

class AnalysisConfig(BaseModel):
    """Configuration for analysis parameters."""
    fps: float = Field(30.0, description="Video frame rate")
    pixel_to_micron: float = Field(1.0, description="Pixel to micron conversion")
    confidence_threshold: float = Field(0.3, description="Detection confidence threshold")
    iou_threshold: float = Field(0.5, description="IoU threshold")
    min_track_length: int = Field(10, description="Minimum track length")
    enable_visualization: bool = Field(True, description="Generate visualization video")
    export_trajectories: bool = Field(True, description="Include trajectory data")

# Response Models
class SpermParameters(BaseModel):
    """Individual sperm parameters."""
    track_id: int = Field(description="Unique track identifier")
    duration_frames: int = Field(description="Track duration in frames")
    duration_seconds: float = Field(description="Track duration in seconds")
    
    # Motion classification
    motility_class: MotilityClass = Field(description="Motility classification")
    is_motile: bool = Field(description="Whether sperm is motile")
    is_progressive: bool = Field(description="Progressive motility")
    
    # Velocity parameters (μm/s)
    vcl: float = Field(description="Curvilinear velocity (μm/s)")
    vsl: float = Field(description="Straight-line velocity (μm/s)")
    vap: float = Field(description="Average path velocity (μm/s)")
    
    # Motion parameters (%)
    lin: float = Field(description="Linearity (VSL/VCL) %")
    str: float = Field(description="Straightness (VSL/VAP) %")
    wob: float = Field(description="Wobble (VAP/VCL) %")
    
    # Path parameters
    alh: float = Field(description="Amplitude of lateral head displacement (μm)")
    bcf: float = Field(description="Beat cross frequency (Hz)")
    
    # Distance parameters (μm)
    total_distance: float = Field(description="Total distance traveled (μm)")
    net_distance: float = Field(description="Net displacement (μm)")
    
    # Optional trajectory data
    trajectory: Optional[List[List[float]]] = Field(None, description="Trajectory coordinates")

class PopulationStatistics(BaseModel):
    """Population-level statistics."""
    total_sperm_count: int = Field(description="Total number of sperm analyzed")
    
    # Counts by motility type
    motile_count: int = Field(description="Number of motile sperm")
    progressive_count: int = Field(description="Number of progressive sperm")
    slow_progressive_count: int = Field(description="Number of slow progressive sperm")
    non_progressive_count: int = Field(description="Number of non-progressive sperm")
    immotile_count: int = Field(description="Number of immotile sperm")
    
    # Percentages
    motility_percentage: float = Field(description="Overall motility percentage")
    progressive_percentage: float = Field(description="Progressive motility percentage")
    slow_progressive_percentage: float = Field(description="Slow progressive percentage")
    non_progressive_percentage: float = Field(description="Non-progressive percentage")
    immotile_percentage: float = Field(description="Immotile percentage")
    
    # Mean values for motile sperm
    mean_vcl: float = Field(description="Mean curvilinear velocity (μm/s)")
    mean_vsl: float = Field(description="Mean straight-line velocity (μm/s)")
    mean_vap: float = Field(description="Mean average path velocity (μm/s)")
    mean_lin: float = Field(description="Mean linearity (%)")
    mean_str: float = Field(description="Mean straightness (%)")
    mean_wob: float = Field(description="Mean wobble (%)")
    mean_alh: float = Field(description="Mean ALH (μm)")
    mean_bcf: float = Field(description="Mean BCF (Hz)")
    
    # Standard deviations
    std_vcl: float = Field(description="Standard deviation of VCL")
    std_vsl: float = Field(description="Standard deviation of VSL")
    std_vap: float = Field(description="Standard deviation of VAP")
    std_lin: float = Field(description="Standard deviation of LIN")

class AnalysisResults(BaseModel):
    """Complete analysis results."""
    analysis_id: str = Field(description="Unique analysis identifier")
    analysis_name: Optional[str] = Field(None, description="Analysis name")
    timestamp: datetime = Field(description="Analysis completion timestamp")
    
    # Video information
    video_filename: str = Field(description="Original video filename")
    video_duration: float = Field(description="Video duration in seconds")
    total_frames: int = Field(description="Total number of frames")
    fps: float = Field(description="Video frame rate")
    
    # Analysis configuration
    config: AnalysisConfig = Field(description="Analysis configuration used")
    
    # Results
    individual_sperm: List[SpermParameters] = Field(description="Individual sperm analysis")
    population_statistics: PopulationStatistics = Field(description="Population statistics")
    
    # Processing information
    processing_time: float = Field(description="Total processing time (seconds)")
    model_version: str = Field(description="AI model version used")
    
    # Files
    visualization_video: Optional[str] = Field(None, description="Path to visualization video")
    csv_export: Optional[str] = Field(None, description="Path to CSV export")
    json_export: Optional[str] = Field(None, description="Path to JSON export")

class AnalysisResponse(BaseModel):
    """Response for analysis submission."""
    analysis_id: str = Field(description="Unique analysis identifier")
    status: AnalysisStatus = Field(description="Current analysis status")
    message: str = Field(description="Status message")
    estimated_processing_time: Optional[float] = Field(None, description="Estimated processing time in seconds")
    created_at: datetime = Field(description="Analysis creation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "pending",
                "message": "Analysis queued for processing",
                "estimated_processing_time": 120.0,
                "created_at": "2024-01-01T12:00:00Z"
            }
        }

class StatusResponse(BaseModel):
    """Analysis status response."""
    analysis_id: str = Field(description="Analysis identifier")
    status: AnalysisStatus = Field(description="Current status")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    message: str = Field(description="Status message")
    created_at: datetime = Field(description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Progress details
    current_frame: Optional[int] = Field(None, description="Current frame being processed")
    total_frames: Optional[int] = Field(None, description="Total frames to process")
    processing_stage: Optional[str] = Field(None, description="Current processing stage")

class ResultsResponse(BaseModel):
    """Results retrieval response."""
    analysis_id: str = Field(description="Analysis identifier")
    status: AnalysisStatus = Field(description="Analysis status")
    results: Optional[AnalysisResults] = Field(None, description="Analysis results if completed")
    available_downloads: List[str] = Field(default_factory=list, description="Available download formats")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    timestamp: datetime = Field(description="Response timestamp")
    version: str = Field(description="API version")
    uptime: float = Field(description="Service uptime in seconds")
    
    # Service health
    database_connected: bool = Field(description="Database connection status")
    model_loaded: bool = Field(description="AI model loading status")
    gpu_available: bool = Field(description="GPU availability")
    
    # System information
    cpu_usage: float = Field(description="CPU usage percentage")
    memory_usage: float = Field(description="Memory usage percentage")
    disk_usage: float = Field(description="Disk usage percentage")
    
    # Analysis queue
    pending_analyses: int = Field(description="Number of pending analyses")
    processing_analyses: int = Field(description="Number of currently processing analyses")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

class FileUploadResponse(BaseModel):
    """File upload response."""
    filename: str = Field(description="Uploaded filename")
    size: int = Field(description="File size in bytes")
    format: VideoFormat = Field(description="Video format")
    duration: Optional[float] = Field(None, description="Video duration if available")
    upload_id: str = Field(description="Upload identifier")
    timestamp: datetime = Field(description="Upload timestamp")

class BulkAnalysisRequest(BaseModel):
    """Bulk analysis request for multiple videos."""
    analysis_name: str = Field(description="Batch analysis name")
    config: AnalysisConfig = Field(description="Analysis configuration")
    video_files: List[str] = Field(description="List of uploaded video file IDs")

class BulkAnalysisResponse(BaseModel):
    """Bulk analysis response."""
    batch_id: str = Field(description="Batch identifier")
    analysis_ids: List[str] = Field(description="Individual analysis IDs")
    total_videos: int = Field(description="Total number of videos")
    estimated_total_time: float = Field(description="Estimated total processing time")

# Validators
@validator('analysis_id', pre=True, always=True)
def validate_uuid(cls, v):
    """Validate UUID format."""
    if isinstance(v, str):
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('Invalid UUID format')
    return v