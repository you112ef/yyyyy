#!/usr/bin/env python3
"""
Sperm Analysis API - FastAPI Backend
Author: Youssef Shitiwi
Description: REST API for sperm analysis using AI models
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import asyncio
from datetime import datetime
import uuid

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.video_processor import VideoProcessor
from backend.services.analysis_service import AnalysisService
from backend.models.schemas import (
    AnalysisResponse, 
    AnalysisRequest, 
    HealthResponse,
    AnalysisStatus,
    ResultsResponse
)
from backend.utils.config import get_settings
from backend.utils.database import get_database
from backend.routes import analysis, results, health

# Global variables
video_processor: VideoProcessor = None
analysis_service: AnalysisService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global video_processor, analysis_service
    
    logger.info("Starting Sperm Analysis API...")
    
    # Initialize services
    settings = get_settings()
    video_processor = VideoProcessor(
        model_path=settings.model_path,
        output_dir=settings.output_dir
    )
    analysis_service = AnalysisService(video_processor)
    
    # Create necessary directories
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.results_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Services initialized successfully")
    
    yield
    
    logger.info("Shutting down Sperm Analysis API...")

# Create FastAPI app
app = FastAPI(
    title="Sperm Analysis API",
    description="""
    ## 🧬 AI-Powered Sperm Analysis System
    
    **Developer: Youssef Shitiwi (يوسف شتيوي)**
    
    This API provides comprehensive Computer-Assisted Sperm Analysis (CASA) using state-of-the-art AI models.
    
    ### Features:
    - **Real-time Detection**: YOLOv8-based sperm detection
    - **Multi-Object Tracking**: DeepSORT for trajectory analysis  
    - **CASA Metrics**: VCL, VSL, LIN, MOT%, and comprehensive motility analysis
    - **Video Processing**: Support for MP4 and AVI formats
    - **Export Options**: JSON and CSV result formats
    
    ### Technology Stack:
    - Deep Learning: PyTorch, YOLOv8 (Ultralytics)
    - Computer Vision: OpenCV, Albumentations
    - Tracking: DeepSORT
    - Backend: FastAPI, Uvicorn
    - Data Processing: NumPy, Pandas, SciPy
    
    ### Usage:
    1. Upload sperm video using `/analyze` endpoint
    2. Monitor analysis progress with `/status/{analysis_id}`
    3. Download results using `/results/{analysis_id}`
    
    For clinical and research applications in fertility analysis.
    """,
    version="1.0.0",
    contact={
        "name": "Youssef Shitiwi",
        "url": "https://github.com/youssef-shitiwi",
        "email": "youssef.shitiwi@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Include routers
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(results.router, prefix="/api/v1", tags=["Results"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sperm Analysis API",
        "version": "1.0.0",
        "developer": "Youssef Shitiwi (يوسف شتيوي)",
        "description": "AI-Powered Computer-Assisted Sperm Analysis",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "analyze": "/api/v1/analyze",
            "status": "/api/v1/status/{analysis_id}",
            "results": "/api/v1/results/{analysis_id}",
            "download": "/api/v1/download/{analysis_id}"
        }
    }

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    return FileResponse("backend/static/favicon.ico")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Dependency to get services
async def get_video_processor() -> VideoProcessor:
    """Get video processor service."""
    if video_processor is None:
        raise HTTPException(status_code=503, detail="Video processor not initialized")
    return video_processor

async def get_analysis_service() -> AnalysisService:
    """Get analysis service."""
    if analysis_service is None:
        raise HTTPException(status_code=503, detail="Analysis service not initialized")
    return analysis_service

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )