#!/usr/bin/env python3
"""
Results API Routes
Author: Youssef Shitiwi
Description: API endpoints for retrieving analysis results
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from pathlib import Path
from loguru import logger

from backend.models.schemas import StatusResponse, ResultsResponse
from backend.services.analysis_service import AnalysisService

router = APIRouter()

# Dependency to get analysis service
async def get_analysis_service() -> AnalysisService:
    """Get analysis service instance."""
    from backend.main import analysis_service
    if analysis_service is None:
        raise HTTPException(status_code=503, detail="Analysis service not available")
    return analysis_service

@router.get("/status/{analysis_id}", response_model=StatusResponse)
async def get_analysis_status(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get analysis status and progress.
    
    - **analysis_id**: ID of the analysis to check
    
    Returns current status, progress percentage, and processing details.
    """
    try:
        status = analysis_service.get_analysis_status(analysis_id)
        
        if not status:
            raise HTTPException(
                status_code=404, 
                detail=f"Analysis not found: {analysis_id}"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analysis status")

@router.get("/results/{analysis_id}", response_model=ResultsResponse)
async def get_analysis_results(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get analysis results.
    
    - **analysis_id**: ID of the analysis
    
    Returns complete analysis results including CASA metrics and statistics.
    """
    try:
        results = analysis_service.get_analysis_results(analysis_id)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Results not found for analysis: {analysis_id}"
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get results for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analysis results")

@router.get("/download/{analysis_id}/{format_type}")
async def download_results(
    analysis_id: str,
    format_type: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Download analysis results in specified format.
    
    - **analysis_id**: ID of the analysis
    - **format_type**: Format to download (csv, json, statistics, trajectories, visualization)
    
    Available formats:
    - **csv**: Sperm analysis data in CSV format
    - **json**: Complete results in JSON format  
    - **statistics**: Population statistics in JSON format
    - **trajectories**: Trajectory data in JSON format
    - **visualization**: Annotated video with tracking visualization
    """
    try:
        # Validate format type
        valid_formats = ['csv', 'json', 'statistics', 'trajectories', 'visualization']
        if format_type not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format type. Must be one of: {', '.join(valid_formats)}"
            )
        
        # Get file path
        file_path = analysis_service.get_download_path(analysis_id, format_type)
        
        if not file_path or not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found for analysis {analysis_id} in format {format_type}"
            )
        
        # Determine media type and filename
        media_type_map = {
            'csv': 'text/csv',
            'json': 'application/json',
            'statistics': 'application/json',
            'trajectories': 'application/json',
            'visualization': 'video/mp4'
        }
        
        filename_map = {
            'csv': f'sperm_analysis_{analysis_id}.csv',
            'json': f'analysis_results_{analysis_id}.json',
            'statistics': f'population_stats_{analysis_id}.json',
            'trajectories': f'trajectories_{analysis_id}.json',
            'visualization': f'visualization_{analysis_id}.mp4'
        }
        
        media_type = media_type_map.get(format_type, 'application/octet-stream')
        filename = filename_map.get(format_type, f'{format_type}_{analysis_id}')
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed for {analysis_id}/{format_type}: {e}")
        raise HTTPException(status_code=500, detail="Download failed")

@router.get("/download/{analysis_id}")
async def list_available_downloads(
    analysis_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    List available download formats for an analysis.
    
    - **analysis_id**: ID of the analysis
    
    Returns list of available download formats and their descriptions.
    """
    try:
        results = analysis_service.get_analysis_results(analysis_id)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found: {analysis_id}"
            )
        
        # Get available downloads
        available_downloads = results.available_downloads
        
        # Add descriptions
        download_info = []
        for format_type in available_downloads:
            info = {
                "format": format_type,
                "url": f"/api/v1/download/{analysis_id}/{format_type}",
            }
            
            if format_type == "csv":
                info["description"] = "Sperm analysis data in CSV format"
                info["content_type"] = "text/csv"
            elif format_type == "json":
                info["description"] = "Complete analysis results in JSON format"
                info["content_type"] = "application/json"
            elif format_type == "statistics":
                info["description"] = "Population statistics in JSON format"
                info["content_type"] = "application/json"
            elif format_type == "trajectories":
                info["description"] = "Trajectory data in JSON format"
                info["content_type"] = "application/json"
            elif format_type == "visualization":
                info["description"] = "Annotated video with tracking visualization"
                info["content_type"] = "video/mp4"
            
            download_info.append(info)
        
        return {
            "analysis_id": analysis_id,
            "status": results.status,
            "available_downloads": download_info,
            "total_formats": len(download_info)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list downloads for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list downloads")

@router.delete("/results/{analysis_id}")
async def cleanup_analysis(
    analysis_id: str,
    keep_results: bool = True,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Clean up analysis data.
    
    - **analysis_id**: ID of the analysis to clean up
    - **keep_results**: Whether to keep result files (default: True)
    
    Removes analysis from processing queue and optionally deletes result files.
    """
    try:
        analysis_service.cleanup_analysis(analysis_id, keep_results)
        
        message = f"Analysis {analysis_id} cleaned up"
        if not keep_results:
            message += " (result files deleted)"
        
        return {
            "message": message,
            "analysis_id": analysis_id,
            "results_kept": keep_results
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup analysis")

@router.get("/queue")
async def get_queue_status(
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get processing queue status.
    
    Returns information about pending, processing, and completed analyses.
    """
    try:
        stats = analysis_service.get_service_stats()
        
        return {
            "queue_status": stats["processing_queue"],
            "total_uploads": stats["total_uploads"],
            "disk_usage": stats["disk_usage"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue status")

# Add missing import
from datetime import datetime