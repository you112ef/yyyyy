#!/usr/bin/env python3
"""
Analysis API Routes
Author: Youssef Shitiwi
Description: API endpoints for sperm analysis
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
import json
from loguru import logger

from backend.models.schemas import (
    AnalysisRequest, AnalysisResponse, AnalysisConfig, 
    ErrorResponse, FileUploadResponse, VideoFormat
)
from backend.services.analysis_service import AnalysisService

router = APIRouter()

# Dependency to get analysis service
async def get_analysis_service() -> AnalysisService:
    """Get analysis service instance."""
    # This would typically be injected via dependency injection
    # For now, we'll assume it's available globally
    from backend.main import analysis_service
    if analysis_service is None:
        raise HTTPException(status_code=503, detail="Analysis service not available")
    return analysis_service

@router.post("/upload", response_model=FileUploadResponse)
async def upload_video(
    file: UploadFile = File(..., description="Video file to upload"),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Upload a video file for analysis.
    
    - **file**: Video file (MP4, AVI, MOV supported)
    - Returns upload ID for use in analysis requests
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be a video file"
            )
        
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Upload file
        upload_id = await analysis_service.upload_video(content, file.filename)
        
        # Determine video format
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown'
        video_format = None
        if file_ext in ['mp4']:
            video_format = VideoFormat.MP4
        elif file_ext in ['avi']:
            video_format = VideoFormat.AVI
        elif file_ext in ['mov']:
            video_format = VideoFormat.MOV
        
        response = FileUploadResponse(
            filename=file.filename,
            size=len(content),
            format=video_format,
            upload_id=upload_id,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"File uploaded successfully: {file.filename} -> {upload_id}")
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@router.post("/analyze", response_model=AnalysisResponse)
async def start_analysis(
    upload_id: str = Form(..., description="Upload ID from previous upload"),
    analysis_name: Optional[str] = Form(None, description="Optional analysis name"),
    config: str = Form(..., description="Analysis configuration as JSON string"),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Start sperm analysis on uploaded video.
    
    - **upload_id**: ID from previous video upload
    - **analysis_name**: Optional name for the analysis
    - **config**: Analysis configuration in JSON format
    
    Returns analysis ID for tracking progress.
    
    ### Example Config:
    ```json
    {
        "fps": 30.0,
        "pixel_to_micron": 0.5,
        "confidence_threshold": 0.3,
        "iou_threshold": 0.5,
        "min_track_length": 10,
        "enable_visualization": true,
        "export_trajectories": true
    }
    ```
    """
    try:
        # Parse configuration
        try:
            config_data = json.loads(config)
            analysis_config = AnalysisConfig(**config_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in config")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
        
        # Start analysis
        response = await analysis_service.start_analysis(
            upload_id=upload_id,
            config=analysis_config,
            analysis_name=analysis_name
        )
        
        logger.info(f"Analysis started: {response.analysis_id}")
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis start failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start analysis")

@router.post("/analyze-direct", response_model=AnalysisResponse)
async def analyze_direct(
    file: UploadFile = File(..., description="Video file to analyze"),
    analysis_name: Optional[str] = Form(None, description="Optional analysis name"),
    fps: float = Form(30.0, description="Video frame rate"),
    pixel_to_micron: float = Form(1.0, description="Pixel to micron conversion factor"),
    confidence_threshold: float = Form(0.3, description="Detection confidence threshold"),
    iou_threshold: float = Form(0.5, description="IoU threshold for NMS"),
    min_track_length: int = Form(10, description="Minimum track length for analysis"),
    enable_visualization: bool = Form(True, description="Generate visualization video"),
    export_trajectories: bool = Form(True, description="Include trajectory data"),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Direct analysis endpoint - upload and analyze in one request.
    
    This endpoint combines upload and analysis into a single request.
    Useful for simple integrations or testing.
    """
    try:
        # Upload file
        content = await file.read()
        upload_id = await analysis_service.upload_video(content, file.filename)
        
        # Create configuration
        config = AnalysisConfig(
            fps=fps,
            pixel_to_micron=pixel_to_micron,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            min_track_length=min_track_length,
            enable_visualization=enable_visualization,
            export_trajectories=export_trajectories
        )
        
        # Start analysis
        response = await analysis_service.start_analysis(
            upload_id=upload_id,
            config=config,
            analysis_name=analysis_name
        )
        
        logger.info(f"Direct analysis started: {response.analysis_id}")
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Direct analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@router.delete("/analysis/{analysis_id}")
async def cancel_analysis(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Cancel a running analysis.
    
    - **analysis_id**: ID of the analysis to cancel
    """
    try:
        success = analysis_service.cancel_analysis(analysis_id)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail="Analysis not found or cannot be cancelled"
            )
        
        return {"message": "Analysis cancelled successfully", "analysis_id": analysis_id}
        
    except Exception as e:
        logger.error(f"Failed to cancel analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel analysis")

@router.delete("/upload/{upload_id}")
async def cleanup_upload(
    upload_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Clean up uploaded file.
    
    - **upload_id**: ID of the upload to clean up
    """
    try:
        analysis_service.cleanup_upload(upload_id)
        return {"message": "Upload cleaned up successfully", "upload_id": upload_id}
        
    except Exception as e:
        logger.error(f"Failed to cleanup upload {upload_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup upload")

# Add missing import
from datetime import datetime