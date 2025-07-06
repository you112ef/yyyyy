#!/usr/bin/env python3
"""
Configuration Management
Author: Youssef Shitiwi
Description: Application configuration and settings
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = Field(default="Sperm Analysis API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # AI Model Configuration
    model_path: str = Field(
        default="data/models/best/sperm_detection_best.pt", 
        env="MODEL_PATH"
    )
    default_confidence_threshold: float = Field(default=0.3, env="DEFAULT_CONFIDENCE_THRESHOLD")
    default_iou_threshold: float = Field(default=0.5, env="DEFAULT_IOU_THRESHOLD")
    
    # Directory Configuration
    upload_dir: str = Field(default="data/uploads", env="UPLOAD_DIR")
    output_dir: str = Field(default="data/results", env="OUTPUT_DIR")
    results_dir: str = Field(default="data/results", env="RESULTS_DIR")
    temp_dir: str = Field(default="data/temp", env="TEMP_DIR")
    
    # Processing Configuration
    max_concurrent_analyses: int = Field(default=2, env="MAX_CONCURRENT_ANALYSES")
    max_upload_size_mb: int = Field(default=500, env="MAX_UPLOAD_SIZE_MB")
    cleanup_interval_hours: int = Field(default=24, env="CLEANUP_INTERVAL_HOURS")
    
    # Database Configuration (optional)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    use_database: bool = Field(default=False, env="USE_DATABASE")
    
    # Redis Configuration (optional, for caching)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    use_redis: bool = Field(default=False, env="USE_REDIS")
    
    # Security Configuration
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    enable_access_log: bool = Field(default=True, env="ENABLE_ACCESS_LOG")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Video Processing Configuration
    default_fps: float = Field(default=30.0, env="DEFAULT_FPS")
    default_pixel_to_micron: float = Field(default=1.0, env="DEFAULT_PIXEL_TO_MICRON")
    min_track_length: int = Field(default=10, env="MIN_TRACK_LENGTH")
    
    # Performance Configuration
    enable_gpu: bool = Field(default=True, env="ENABLE_GPU")
    batch_size: int = Field(default=16, env="BATCH_SIZE")
    num_workers: int = Field(default=4, env="NUM_WORKERS")
    
    # Feature Flags
    enable_visualization: bool = Field(default=True, env="ENABLE_VISUALIZATION")
    enable_trajectories: bool = Field(default=True, env="ENABLE_TRAJECTORIES")
    enable_bulk_analysis: bool = Field(default=True, env="ENABLE_BULK_ANALYSIS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.upload_dir,
            self.output_dir,
            self.results_dir,
            self.temp_dir,
            "data/models",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def cors_origins_list(self) -> list:
        """Get CORS origins as list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024
    
    def get_model_path(self) -> Path:
        """Get model path as Path object."""
        return Path(self.model_path)
    
    def is_model_available(self) -> bool:
        """Check if model file exists."""
        model_path = self.get_model_path()
        return model_path.exists() or self.model_path == "yolov8n.pt"  # Default model
    
    def get_database_config(self) -> dict:
        """Get database configuration."""
        if not self.use_database or not self.database_url:
            return {}
        
        return {
            "url": self.database_url,
            "echo": self.debug
        }
    
    def get_redis_config(self) -> dict:
        """Get Redis configuration."""
        if not self.use_redis or not self.redis_url:
            return {}
        
        return {
            "url": self.redis_url
        }

class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    enable_access_log: bool = True
    workers: int = 1

class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    log_level: str = "INFO"
    enable_access_log: bool = False
    workers: int = 4
    enable_auth: bool = True

class TestingSettings(Settings):
    """Testing environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    use_database: bool = False
    upload_dir: str = "test_data/uploads"
    output_dir: str = "test_data/results"
    max_concurrent_analyses: int = 1

@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    Returns appropriate settings based on environment.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

def get_env_info() -> dict:
    """Get environment information."""
    settings = get_settings()
    
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "debug": settings.debug,
        "model_available": settings.is_model_available(),
        "gpu_enabled": settings.enable_gpu,
        "database_enabled": settings.use_database,
        "redis_enabled": settings.use_redis,
        "auth_enabled": settings.enable_auth
    }