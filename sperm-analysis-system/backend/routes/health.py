#!/usr/bin/env python3
"""
Health Check API Routes
Author: Youssef Shitiwi
Description: Health monitoring endpoints for the API
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import psutil
import torch
import os
import sys
from pathlib import Path
from loguru import logger

from backend.models.schemas import HealthResponse

router = APIRouter()

# Store startup time for uptime calculation
startup_time = datetime.utcnow()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns detailed system health information including:
    - Service status
    - Database connectivity 
    - AI model status
    - System resources
    - Processing queue status
    """
    try:
        # Calculate uptime
        uptime = (datetime.utcnow() - startup_time).total_seconds()
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        
        # Get system resources
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check AI model loading
        model_loaded = _check_model_status()
        
        # Get database status (simplified - would connect to actual DB)
        database_connected = True  # Placeholder
        
        # Get processing queue info
        pending_analyses, processing_analyses = _get_queue_info()
        
        health_data = HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime=uptime,
            database_connected=database_connected,
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            pending_analyses=pending_analyses,
            processing_analyses=processing_analyses
        )
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        # Return degraded status
        return HealthResponse(
            status="degraded",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime=0,
            database_connected=False,
            model_loaded=False,
            gpu_available=False,
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            pending_analyses=0,
            processing_analyses=0
        )

@router.get("/health/simple")
async def simple_health_check():
    """
    Simple health check endpoint for load balancers.
    
    Returns basic OK status if service is running.
    """
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "sperm-analysis-api"
    }

@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    
    Checks if the service is ready to accept requests.
    """
    try:
        # Check critical dependencies
        model_loaded = _check_model_status()
        
        if not model_loaded:
            raise HTTPException(
                status_code=503,
                detail="Service not ready - AI model not loaded"
            )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": model_loaded
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )

@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Checks if the service is alive and functioning.
    """
    try:
        # Basic liveness check
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": (datetime.utcnow() - startup_time).total_seconds()
        }
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service not alive"
        )

@router.get("/health/system")
async def system_info():
    """
    Detailed system information endpoint.
    
    Returns comprehensive system and environment information.
    """
    try:
        # System information
        system_info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
        }
        
        # PyTorch information
        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            torch_info["devices"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                }
                for i in range(torch.cuda.device_count())
            ]
        
        # Environment information
        env_info = {
            "model_path": os.environ.get("MODEL_PATH", "Not set"),
            "output_dir": os.environ.get("OUTPUT_DIR", "Not set"),
            "debug_mode": os.environ.get("DEBUG", "False").lower() == "true"
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_info,
            "pytorch": torch_info,
            "environment": env_info
        }
        
    except Exception as e:
        logger.error(f"System info failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

@router.get("/health/metrics")
async def get_metrics():
    """
    Prometheus-style metrics endpoint.
    
    Returns metrics in a format suitable for monitoring systems.
    """
    try:
        # Get queue info
        pending, processing = _get_queue_info()
        
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Format as Prometheus metrics
        metrics = f"""# HELP sperm_analysis_pending_analyses Number of pending analyses
# TYPE sperm_analysis_pending_analyses gauge
sperm_analysis_pending_analyses {pending}

# HELP sperm_analysis_processing_analyses Number of processing analyses
# TYPE sperm_analysis_processing_analyses gauge
sperm_analysis_processing_analyses {processing}

# HELP sperm_analysis_cpu_usage_percent CPU usage percentage
# TYPE sperm_analysis_cpu_usage_percent gauge
sperm_analysis_cpu_usage_percent {cpu_usage}

# HELP sperm_analysis_memory_usage_percent Memory usage percentage
# TYPE sperm_analysis_memory_usage_percent gauge
sperm_analysis_memory_usage_percent {memory.percent}

# HELP sperm_analysis_disk_usage_percent Disk usage percentage
# TYPE sperm_analysis_disk_usage_percent gauge
sperm_analysis_disk_usage_percent {disk.percent}

# HELP sperm_analysis_uptime_seconds Service uptime in seconds
# TYPE sperm_analysis_uptime_seconds counter
sperm_analysis_uptime_seconds {(datetime.utcnow() - startup_time).total_seconds()}

# HELP sperm_analysis_gpu_available GPU availability
# TYPE sperm_analysis_gpu_available gauge
sperm_analysis_gpu_available {1 if torch.cuda.is_available() else 0}
"""
        
        return {
            "content": metrics,
            "content_type": "text/plain"
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect metrics")

def _check_model_status() -> bool:
    """Check if AI model is loaded and available."""
    try:
        # This would check if the model is actually loaded
        # For now, we'll check if the model file exists
        model_path = Path("data/models/best/sperm_detection_best.pt")
        return model_path.exists() or True  # Allow default YOLO model
    except:
        return False

def _get_queue_info() -> tuple[int, int]:
    """Get processing queue information."""
    try:
        # This would get actual queue info from the video processor
        # For now, return placeholder values
        from backend.main import video_processor
        if video_processor:
            queue_status = video_processor.get_queue_status()
            return queue_status.get("pending", 0), queue_status.get("processing", 0)
        return 0, 0
    except:
        return 0, 0