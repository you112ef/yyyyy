# 🔧 Advanced Configuration Guide

> **Developer:** Youssef Shitiwi  
> **System:** Sperm Analysis with AI-Powered CASA Metrics

This comprehensive guide covers all configuration options for customizing the sperm analysis system to meet specific research requirements and deployment environments.

## 📋 Table of Contents

1. [Environment Configuration](#environment-configuration)
2. [AI Model Parameters](#ai-model-parameters)
3. [CASA Metrics Customization](#casa-metrics-customization)
4. [Backend API Settings](#backend-api-settings)
5. [Database Configuration](#database-configuration)
6. [Video Processing Options](#video-processing-options)
7. [Android App Configuration](#android-app-configuration)
8. [Docker & Deployment](#docker--deployment)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

---

## 🌍 Environment Configuration

### Backend Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false
API_DEBUG=false

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/sperm_analysis
DATABASE_POOL_SIZE=20
DATABASE_ECHO=false

# Redis Configuration (for caching and queues)
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=10

# File Storage
UPLOAD_DIR=/app/data/uploads
RESULTS_DIR=/app/data/results
MODELS_DIR=/app/data/models
MAX_UPLOAD_SIZE=500MB

# AI Model Configuration
MODEL_PATH=/app/data/models/yolo_sperm_best.pt
MODEL_CONFIDENCE=0.5
MODEL_IOU_THRESHOLD=0.45
MODEL_DEVICE=auto  # auto, cpu, cuda, mps

# Video Processing
MAX_CONCURRENT_ANALYSES=3
FRAME_RATE_LIMIT=30
MAX_VIDEO_DURATION=300  # seconds
SUPPORTED_FORMATS=mp4,avi,mov,mkv

# Security
SECRET_KEY=your-super-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=detailed
LOG_FILE=/app/logs/api.log
LOG_ROTATION=1 week

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# External Services
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

### Training Environment Variables

Create a `.env.training` file:

```bash
# Training Configuration
DATASET_PATH=/app/data/training/sperm_dataset
TRAINING_OUTPUT=/app/data/models
EXPERIMENT_NAME=sperm_detection_v1

# Model Hyperparameters
EPOCHS=100
BATCH_SIZE=16
IMAGE_SIZE=640
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0005
MOMENTUM=0.937

# Data Augmentation
AUGMENT_HSV_H=0.015
AUGMENT_HSV_S=0.7
AUGMENT_HSV_V=0.4
AUGMENT_DEGREES=0.0
AUGMENT_TRANSLATE=0.1
AUGMENT_SCALE=0.5
AUGMENT_SHEAR=0.0
AUGMENT_PERSPECTIVE=0.0
AUGMENT_FLIPUD=0.0
AUGMENT_FLIPLR=0.5
AUGMENT_MIXUP=0.0

# Hardware Configuration
DEVICE=0  # GPU device or 'cpu'
WORKERS=8  # Number of dataloader workers
DDP=false  # Distributed training

# Logging & Monitoring
WANDB_PROJECT=sperm-analysis
WANDB_ENTITY=your-username
TENSORBOARD_DIR=/app/logs/tensorboard
SAVE_PERIOD=10  # Save checkpoint every N epochs
```

---

## 🤖 AI Model Parameters

### YOLOv8 Model Configuration

Edit `training/configs/yolo_config.yaml`:

```yaml
# Model Architecture
model:
  type: "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  nc: 1  # Number of classes (sperm)
  anchors: auto
  
# Dataset Configuration
data:
  train: /app/data/training/sperm_dataset/train
  val: /app/data/training/sperm_dataset/val
  test: /app/data/training/sperm_dataset/test
  names:
    0: sperm

# Training Hyperparameters
train:
  epochs: 100
  batch_size: 16
  imgsz: 640
  lr0: 0.01  # Initial learning rate
  lrf: 0.01  # Final learning rate
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
  
# Validation Settings
val:
  batch_size: 32
  imgsz: 640
  conf_thres: 0.001
  iou_thres: 0.6
  max_det: 300
  save_json: true
  save_hybrid: false
  
# Augmentation Settings
augment:
  hsv_h: 0.015  # Hue augmentation
  hsv_s: 0.7    # Saturation augmentation
  hsv_v: 0.4    # Value augmentation
  degrees: 0.0  # Rotation degrees
  translate: 0.1  # Translation
  scale: 0.5    # Scaling
  shear: 0.0    # Shearing
  perspective: 0.0  # Perspective
  flipud: 0.0   # Vertical flip probability
  fliplr: 0.5   # Horizontal flip probability
  mosaic: 1.0   # Mosaic augmentation probability
  mixup: 0.0    # Mixup augmentation probability
  copy_paste: 0.0  # Copy-paste augmentation probability

# Loss Configuration
loss:
  box: 7.5      # Box loss gain
  cls: 0.5      # Class loss gain
  dfl: 1.5      # DFL loss gain
  
# Post-processing
nms:
  conf_thres: 0.25
  iou_thres: 0.45
  max_det: 300
  agnostic: false
  multi_label: false
```

### DeepSORT Tracker Configuration

Create `backend/configs/tracker_config.yaml`:

```yaml
# DeepSORT Configuration
tracker:
  # Kalman Filter Parameters
  kalman:
    max_age: 70              # Max frames to keep lost tracks
    n_init: 3                # Frames to confirm track
    nn_budget: 100           # Max features per class
    
  # Distance Metrics
  distance:
    max_cosine_distance: 0.2  # Cosine distance threshold
    max_iou_distance: 0.7     # IoU distance threshold
    max_distance: 2.0         # Max Euclidean distance
    
  # Feature Extraction
  features:
    model_path: /app/data/models/deep_sort_reid.pb
    input_size: [64, 128]     # Width, Height
    feature_dim: 512          # Feature vector dimension
    
  # Motion Model
  motion:
    chi2_threshold: 9.4877    # Gating threshold
    std_weight_position: 1/20
    std_weight_velocity: 1/160
```

---

## 📊 CASA Metrics Customization

### Custom Metrics Configuration

Edit `backend/configs/casa_config.yaml`:

```yaml
# CASA Analysis Parameters
casa:
  # Frame Rate and Time Settings
  fps: 30                    # Frames per second
  time_window: 5.0           # Analysis window in seconds
  min_track_length: 10       # Minimum frames for valid track
  
  # Spatial Calibration
  calibration:
    pixels_per_micron: 2.5   # Spatial calibration factor
    field_of_view: [640, 480] # Width, Height in pixels
    real_dimensions: [256, 192] # Width, Height in microns
    
  # Velocity Calculations
  velocity:
    smoothing_window: 5      # Frames for velocity smoothing
    outlier_threshold: 3.0   # Standard deviations for outlier removal
    min_velocity: 5.0        # Minimum velocity (μm/s)
    
  # WHO Standards Compliance
  who_standards:
    # Velocity Thresholds (μm/s)
    rapid_velocity: 25.0
    medium_velocity: 5.0
    slow_velocity: 0.0
    
    # Linearity Thresholds (%)
    linear_threshold: 50.0
    non_linear_threshold: 80.0
    
    # Motility Classification
    progressive_lin_threshold: 45.0
    progressive_vap_threshold: 25.0
    
  # Advanced Parameters
  advanced:
    # Head Oscillation
    alh_calculation: true     # Calculate ALH
    alh_smoothing: 0.2       # ALH smoothing factor
    
    # Beat Frequency
    bcf_calculation: true     # Calculate BCF
    bcf_window: 1.0          # BCF time window (seconds)
    bcf_threshold: 0.5       # BCF detection threshold
    
    # Hyperactivation Detection
    hyperactivation:
      enabled: true
      vcl_threshold: 150.0   # VCL threshold (μm/s)
      alh_threshold: 7.0     # ALH threshold (μm)
      lin_threshold: 0.5     # LIN threshold (ratio)
      
  # Quality Control
  quality:
    min_sperm_count: 10      # Minimum sperm for analysis
    max_concentration: 1000  # Maximum sperm per field
    track_quality_threshold: 0.8  # Track quality score
    
  # Export Settings
  export:
    include_trajectories: true
    include_individual_metrics: true
    include_population_stats: true
    include_who_classification: true
    decimal_precision: 2
```

### Custom Metric Calculations

Add custom metrics in `training/models/custom_casa.py`:

```python
"""
Custom CASA Metrics Extension
Developer: Youssef Shitiwi
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy import signal

@dataclass
class CustomSpermMetrics:
    """Extended sperm analysis metrics"""
    
    # Standard CASA metrics
    vcl: float
    vsl: float
    vap: float
    lin: float
    str: float
    wob: float
    alh: float
    bcf: float
    
    # Custom research metrics
    path_efficiency: float        # Path efficiency index
    direction_change_rate: float  # Direction changes per second
    acceleration_variance: float  # Acceleration variability
    turning_angle_variance: float # Turning angle variability
    velocity_correlation: float   # Velocity autocorrelation
    fractal_dimension: float      # Path fractal dimension
    energy_index: float          # Swimming energy index
    rhythmicity_index: float     # Swimming rhythm consistency

class AdvancedCASAAnalyzer:
    """Advanced CASA metrics calculator"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def calculate_path_efficiency(self, trajectory: np.ndarray) -> float:
        """Calculate path efficiency (VSL/VCL ratio with time weighting)"""
        if len(trajectory) < 3:
            return 0.0
            
        # Calculate cumulative distances
        distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        
        if total_distance == 0:
            return 0.0
            
        # Straight line distance
        straight_distance = np.sqrt(np.sum((trajectory[-1] - trajectory[0])**2))
        
        return straight_distance / total_distance
    
    def calculate_direction_changes(self, trajectory: np.ndarray, 
                                  fps: float = 30.0) -> float:
        """Calculate direction change rate per second"""
        if len(trajectory) < 3:
            return 0.0
            
        # Calculate movement vectors
        vectors = np.diff(trajectory, axis=0)
        
        # Calculate angles between consecutive vectors
        angles = []
        for i in range(len(vectors) - 1):
            v1, v2 = vectors[i], vectors[i + 1]
            
            # Avoid division by zero
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue
                
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        if not angles:
            return 0.0
            
        # Count significant direction changes (> 30 degrees)
        threshold = np.pi / 6  # 30 degrees
        changes = np.sum(np.array(angles) > threshold)
        
        # Convert to changes per second
        time_duration = len(trajectory) / fps
        return changes / time_duration if time_duration > 0 else 0.0
    
    def calculate_acceleration_variance(self, trajectory: np.ndarray, 
                                      fps: float = 30.0) -> float:
        """Calculate acceleration variability"""
        if len(trajectory) < 3:
            return 0.0
            
        # Calculate velocity vectors
        dt = 1.0 / fps
        velocities = np.diff(trajectory, axis=0) / dt
        
        if len(velocities) < 2:
            return 0.0
            
        # Calculate acceleration vectors
        accelerations = np.diff(velocities, axis=0) / dt
        
        # Calculate acceleration magnitudes
        acc_magnitudes = np.sqrt(np.sum(accelerations**2, axis=1))
        
        return np.var(acc_magnitudes) if len(acc_magnitudes) > 0 else 0.0
    
    def calculate_fractal_dimension(self, trajectory: np.ndarray) -> float:
        """Calculate path fractal dimension using box counting"""
        if len(trajectory) < 4:
            return 1.0
            
        # Normalize trajectory to unit square
        traj = trajectory - np.min(trajectory, axis=0)
        max_range = np.max(np.max(trajectory, axis=0) - np.min(trajectory, axis=0))
        if max_range == 0:
            return 1.0
        traj = traj / max_range
        
        # Box counting algorithm
        scales = np.logspace(-2, 0, 20)  # From 0.01 to 1.0
        counts = []
        
        for scale in scales:
            # Create grid
            grid_size = int(1.0 / scale) + 1
            grid = np.zeros((grid_size, grid_size), dtype=bool)
            
            # Mark boxes containing trajectory points
            for point in traj:
                x_idx = min(int(point[0] / scale), grid_size - 1)
                y_idx = min(int(point[1] / scale), grid_size - 1)
                grid[x_idx, y_idx] = True
            
            counts.append(np.sum(grid))
        
        # Fit line to log-log plot
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        # Remove infinite values
        valid_idx = np.isfinite(log_scales) & np.isfinite(log_counts)
        if np.sum(valid_idx) < 2:
            return 1.0
            
        slope, _ = np.polyfit(log_scales[valid_idx], log_counts[valid_idx], 1)
        return -slope  # Fractal dimension
    
    def calculate_velocity_correlation(self, trajectory: np.ndarray,
                                     fps: float = 30.0) -> float:
        """Calculate velocity autocorrelation"""
        if len(trajectory) < 10:
            return 0.0
            
        # Calculate velocity magnitudes
        dt = 1.0 / fps
        velocities = np.diff(trajectory, axis=0) / dt
        vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        
        if len(vel_magnitudes) < 5:
            return 0.0
            
        # Calculate autocorrelation
        correlation = np.correlate(vel_magnitudes, vel_magnitudes, mode='full')
        correlation = correlation[correlation.size // 2:]
        
        # Normalize
        correlation = correlation / correlation[0] if correlation[0] != 0 else correlation
        
        # Return correlation at lag=1
        return correlation[1] if len(correlation) > 1 else 0.0
    
    def calculate_energy_index(self, trajectory: np.ndarray,
                              fps: float = 30.0) -> float:
        """Calculate swimming energy index based on acceleration work"""
        if len(trajectory) < 3:
            return 0.0
            
        dt = 1.0 / fps
        
        # Calculate velocities and accelerations
        velocities = np.diff(trajectory, axis=0) / dt
        accelerations = np.diff(velocities, axis=0) / dt
        
        if len(accelerations) == 0:
            return 0.0
            
        # Calculate work done (F·v approximation)
        # Assuming unit mass, F = ma
        work_elements = []
        for i in range(len(accelerations)):
            if i < len(velocities) - 1:
                # Power = F·v = ma·v
                power = np.dot(accelerations[i], velocities[i + 1])
                work_elements.append(abs(power))
        
        return np.mean(work_elements) if work_elements else 0.0
    
    def calculate_rhythmicity_index(self, trajectory: np.ndarray,
                                   fps: float = 30.0) -> float:
        """Calculate swimming rhythm consistency"""
        if len(trajectory) < 20:  # Need sufficient data for frequency analysis
            return 0.0
            
        # Calculate velocity magnitudes
        dt = 1.0 / fps
        velocities = np.diff(trajectory, axis=0) / dt
        vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        
        if len(vel_magnitudes) < 10:
            return 0.0
            
        # Perform FFT to find dominant frequencies
        fft = np.fft.fft(vel_magnitudes - np.mean(vel_magnitudes))
        power_spectrum = np.abs(fft[:len(fft)//2])**2
        
        if len(power_spectrum) == 0:
            return 0.0
            
        # Find peak frequency power
        peak_power = np.max(power_spectrum)
        total_power = np.sum(power_spectrum)
        
        # Rhythmicity index: ratio of peak power to total power
        return peak_power / total_power if total_power > 0 else 0.0

# Usage example configuration
CUSTOM_CASA_CONFIG = {
    'advanced_metrics': {
        'path_efficiency': True,
        'direction_change_rate': True,
        'acceleration_variance': True,
        'fractal_dimension': True,
        'velocity_correlation': True,
        'energy_index': True,
        'rhythmicity_index': True
    },
    'thresholds': {
        'min_path_efficiency': 0.1,
        'max_direction_changes': 10.0,
        'fractal_dimension_range': [1.0, 2.0]
    }
}
```

---

## 🌐 Backend API Settings

### FastAPI Configuration

Create `backend/configs/api_config.yaml`:

```yaml
# API Configuration
api:
  title: "Sperm Analysis API"
  description: "AI-Powered Computer-Assisted Sperm Analysis (CASA) System by Youssef Shitiwi"
  version: "1.0.0"
  docs_url: "/docs"
  redoc_url: "/redoc"
  openapi_url: "/openapi.json"
  
# CORS Settings
cors:
  allow_origins: ["*"]
  allow_credentials: true
  allow_methods: ["*"]
  allow_headers: ["*"]
  max_age: 600

# Rate Limiting
rate_limiting:
  enabled: true
  default_limit: "100/minute"
  upload_limit: "10/minute"
  analysis_limit: "5/minute"
  
# File Upload Settings
upload:
  max_file_size: 524288000  # 500MB
  allowed_extensions: [".mp4", ".avi", ".mov", ".mkv"]
  temp_dir: "/tmp/uploads"
  cleanup_interval: 3600  # 1 hour
  
# Analysis Queue Settings
queue:
  max_concurrent: 3
  timeout: 1800  # 30 minutes
  retry_attempts: 3
  priority_levels: ["low", "normal", "high"]
  
# Result Storage
storage:
  retention_days: 30
  compression: true
  backup_enabled: true
  cleanup_schedule: "0 2 * * *"  # Daily at 2 AM
  
# Monitoring
monitoring:
  metrics_enabled: true
  health_check_interval: 30
  performance_tracking: true
  error_reporting: true
  
# Security
security:
  api_key_required: false
  jwt_enabled: false
  https_only: false
  request_validation: true
```

---

## 🗄️ Database Configuration

### PostgreSQL Setup

Create `docker/database/postgresql.conf`:

```conf
# PostgreSQL Configuration for Sperm Analysis System
# Optimized for video analysis workloads

# Connection Settings
listen_addresses = '*'
port = 5432
max_connections = 100
superuser_reserved_connections = 3

# Memory Settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Write-Ahead Logging
wal_level = replica
max_wal_size = 1GB
min_wal_size = 80MB
checkpoint_timeout = 5min

# Query Tuning
work_mem = 4MB
max_worker_processes = 8
max_parallel_workers_per_gather = 2
max_parallel_workers = 8
max_parallel_maintenance_workers = 2

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_truncate_on_rotation = on
log_rotation_age = 1d
log_rotation_size = 10MB
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

# Autovacuum
autovacuum = on
log_autovacuum_min_duration = 0
autovacuum_max_workers = 3
autovacuum_naptime = 1min
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.2
autovacuum_analyze_scale_factor = 0.1
autovacuum_freeze_max_age = 200000000
autovacuum_multixact_freeze_max_age = 400000000
autovacuum_vacuum_cost_delay = 20ms
autovacuum_vacuum_cost_limit = -1
```

### Database Schema Customization

Create `backend/models/custom_schema.py`:

```python
"""
Custom Database Schema Extensions
Developer: Youssef Shitiwi
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid
from datetime import datetime

Base = declarative_base()

class CustomAnalysisRecord(Base):
    """Extended analysis record with custom fields"""
    __tablename__ = "custom_analysis_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True)
    session_id = Column(String, index=True)
    
    # File information
    original_filename = Column(String)
    file_hash = Column(String, unique=True, index=True)
    file_size = Column(Integer)
    video_duration = Column(Float)
    video_fps = Column(Float)
    video_resolution = Column(String)
    
    # Analysis configuration
    model_version = Column(String)
    analysis_config = Column(JSON)
    calibration_data = Column(JSON)
    
    # Processing information
    processing_start = Column(DateTime, default=datetime.utcnow)
    processing_end = Column(DateTime)
    processing_duration = Column(Float)
    frames_processed = Column(Integer)
    
    # Quality metrics
    video_quality_score = Column(Float)
    tracking_quality_score = Column(Float)
    confidence_score = Column(Float)
    
    # Results summary
    total_sperm_count = Column(Integer)
    motile_sperm_count = Column(Integer)
    progressive_sperm_count = Column(Integer)
    
    # WHO classification
    who_concentration = Column(Float)
    who_motility = Column(Float)
    who_progressive_motility = Column(Float)
    who_morphology = Column(Float)
    
    # Advanced metrics
    population_vcl_mean = Column(Float)
    population_vsl_mean = Column(Float)
    population_lin_mean = Column(Float)
    hyperactivated_percentage = Column(Float)
    
    # Custom research fields
    research_notes = Column(Text)
    experimental_conditions = Column(JSON)
    researcher_id = Column(String)
    study_group = Column(String)
    
    # Status and metadata
    status = Column(String, default="pending")
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SpermTrajectoryData(Base):
    """Detailed sperm trajectory storage"""
    __tablename__ = "sperm_trajectories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), index=True)
    track_id = Column(Integer)
    
    # Trajectory data
    x_coordinates = Column(ARRAY(Float))
    y_coordinates = Column(ARRAY(Float))
    timestamps = Column(ARRAY(Float))
    confidences = Column(ARRAY(Float))
    
    # Calculated metrics
    vcl = Column(Float)
    vsl = Column(Float)
    vap = Column(Float)
    lin = Column(Float)
    str = Column(Float)
    wob = Column(Float)
    alh = Column(Float)
    bcf = Column(Float)
    
    # WHO classification
    motility_class = Column(String)  # PR, NP, IM
    
    # Track quality
    track_length = Column(Integer)
    track_duration = Column(Float)
    quality_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class CalibrationSettings(Base):
    """Calibration settings for different equipment"""
    __tablename__ = "calibration_settings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True)
    description = Column(Text)
    
    # Spatial calibration
    pixels_per_micron = Column(Float)
    magnification = Column(Float)
    objective_lens = Column(String)
    
    # Temporal calibration
    frame_rate = Column(Float)
    exposure_time = Column(Float)
    
    # Equipment details
    microscope_model = Column(String)
    camera_model = Column(String)
    software_version = Column(String)
    
    # Validation data
    is_validated = Column(Boolean, default=False)
    validation_date = Column(DateTime)
    validation_accuracy = Column(Float)
    
    created_by = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

---

## 🎥 Video Processing Options

### OpenCV Configuration

Create `backend/configs/video_config.yaml`:

```yaml
# Video Processing Configuration
video_processing:
  # Input Settings
  input:
    supported_formats: [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
    max_file_size: 500  # MB
    max_duration: 300   # seconds
    min_duration: 5     # seconds
    
  # Frame Processing
  frames:
    target_fps: 30
    max_fps: 60
    min_fps: 10
    resize_method: "bilinear"  # nearest, bilinear, bicubic
    maintain_aspect_ratio: true
    
  # Quality Enhancement
  enhancement:
    enabled: true
    brightness_adjustment: true
    contrast_enhancement: true
    noise_reduction: true
    sharpening: false
    histogram_equalization: false
    
  # Region of Interest
  roi:
    auto_detection: true
    manual_override: true
    padding: 50  # pixels
    min_area: 10000  # pixels²
    
  # Output Settings
  output:
    visualization_video: true
    trajectory_overlay: true
    metrics_overlay: true
    compression_quality: 85
    output_fps: 30

# Advanced Processing Options
advanced:
  # Background Subtraction
  background_subtraction:
    enabled: true
    method: "MOG2"  # MOG2, KNN, GMM
    learning_rate: 0.01
    threshold: 16
    detect_shadows: true
    
  # Motion Detection
  motion_detection:
    enabled: true
    sensitivity: 0.1
    min_motion_area: 100
    blur_kernel_size: 21
    
  # Tracking Optimization
  tracking:
    multi_threading: true
    gpu_acceleration: true
    memory_optimization: true
    batch_processing: false
```

---

## 📱 Android App Configuration

### Build Configuration

Edit `android/app/build.gradle` for custom builds:

```gradle
android {
    compileSdk 34
    
    defaultConfig {
        applicationId "com.spermanalysis"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "1.0"
        
        // Custom build configurations
        buildConfigField "String", "API_BASE_URL", "\"http://10.0.2.2:8000\""
        buildConfigField "String", "DEVELOPER_NAME", "\"Youssef Shitiwi\""
        buildConfigField "boolean", "DEBUG_MODE", "true"
        buildConfigField "int", "MAX_VIDEO_SIZE_MB", "500"
        buildConfigField "int", "MAX_VIDEO_DURATION_SEC", "300"
    }
    
    buildTypes {
        debug {
            debuggable true
            minifyEnabled false
            buildConfigField "String", "API_BASE_URL", "\"http://10.0.2.2:8000\""
            buildConfigField "boolean", "DEBUG_MODE", "true"
            applicationIdSuffix ".debug"
            versionNameSuffix "-debug"
        }
        
        release {
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
            buildConfigField "String", "API_BASE_URL", "\"https://api.spermanalysis.com\""
            buildConfigField "boolean", "DEBUG_MODE", "false"
            
            // Signing configuration
            signingConfig signingConfigs.create("release") {
                storeFile file("../keystore/sperm-analysis.keystore")
                storePassword "spermanalysis123"
                keyAlias "sperm-analysis"
                keyPassword "spermanalysis123"
            }
        }
        
        staging {
            initWith debug
            buildConfigField "String", "API_BASE_URL", "\"https://staging-api.spermanalysis.com\""
            applicationIdSuffix ".staging"
            versionNameSuffix "-staging"
        }
    }
    
    // Product flavors for different deployments
    flavorDimensions "version"
    productFlavors {
        research {
            dimension "version"
            applicationIdSuffix ".research"
            versionNameSuffix "-research"
            buildConfigField "boolean", "RESEARCH_MODE", "true"
        }
        
        clinical {
            dimension "version"
            applicationIdSuffix ".clinical"
            versionNameSuffix "-clinical"
            buildConfigField "boolean", "RESEARCH_MODE", "false"
        }
        
        demo {
            dimension "version"
            applicationIdSuffix ".demo"
            versionNameSuffix "-demo"
            buildConfigField "boolean", "DEMO_MODE", "true"
        }
    }
}
```

### App Configuration

Create `android/app/src/main/assets/config.json`:

```json
{
  "app": {
    "name": "Sperm Analysis",
    "developer": "Youssef Shitiwi",
    "version": "1.0.0",
    "build_date": "2024-01-01"
  },
  "api": {
    "base_url": "http://10.0.2.2:8000",
    "timeout": 60000,
    "retry_attempts": 3,
    "endpoints": {
      "upload": "/api/v1/analysis/upload",
      "analyze": "/api/v1/analysis/analyze",
      "status": "/api/v1/results/status",
      "results": "/api/v1/results",
      "download": "/api/v1/results/download"
    }
  },
  "video": {
    "max_file_size_mb": 500,
    "max_duration_sec": 300,
    "min_duration_sec": 5,
    "supported_formats": ["mp4", "avi", "mov"],
    "quality_settings": {
      "resolution": "720p",
      "frame_rate": 30,
      "bitrate": 2000000
    }
  },
  "ui": {
    "theme": "material",
    "dark_mode": false,
    "language": "en",
    "animations_enabled": true,
    "haptic_feedback": true
  },
  "analysis": {
    "auto_start": false,
    "progress_updates": true,
    "notification_enabled": true,
    "background_processing": false
  },
  "storage": {
    "cache_size_mb": 100,
    "auto_cleanup": true,
    "cleanup_interval_days": 7,
    "export_formats": ["csv", "json", "pdf"]
  },
  "privacy": {
    "analytics_enabled": false,
    "crash_reporting": true,
    "data_encryption": true,
    "local_storage_only": true
  }
}
```

---

## 🐳 Docker & Deployment

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  # Main API Service
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: backend
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://sperm_user:${DB_PASSWORD}@postgres:5432/sperm_analysis
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - sperm-network
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  # Load Balancer
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./docker/nginx/ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - api
    networks:
      - sperm-network

  # Database
  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=sperm_analysis
      - POSTGRES_USER=sperm_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/database/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./backups:/backups
    networks:
      - sperm-network
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  # Cache & Queue
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - sperm-network

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    volumes:
      - ./docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - sperm-network

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/grafana/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - sperm-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  sperm-network:
    driver: bridge
```

---

## 🔍 Troubleshooting

### Common Configuration Issues

1. **Model Loading Errors**
   ```bash
   # Check model file permissions
   ls -la /app/data/models/
   chmod 644 /app/data/models/*.pt
   
   # Verify model compatibility
   python -c "import torch; print(torch.__version__)"
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connection
   psql -h localhost -U sperm_user -d sperm_analysis
   
   # Check connection pool
   docker logs sperm-analysis-postgres-1
   ```

3. **Android Build Issues**
   ```bash
   # Clean build
   cd android && ./gradlew clean
   
   # Check Java version
   java -version
   
   # Verify Android SDK
   echo $ANDROID_HOME
   ```

4. **Performance Tuning**
   ```yaml
   # backend/configs/performance.yaml
   performance:
     cpu_optimization: true
     gpu_acceleration: true
     memory_limit: "4GB"
     concurrent_analyses: 3
     batch_size: 16
     threading:
       max_workers: 8
       queue_size: 100
   ```

---

This comprehensive configuration guide enables full customization of the sperm analysis system according to specific research requirements and deployment environments. All configurations maintain the high-quality standards set by developer **Youssef Shitiwi**.

For additional support or advanced customization requirements, refer to the troubleshooting section or contact the development team.