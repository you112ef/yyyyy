# Sperm Analysis System - Production Dockerfile
# Author: Youssef Shitiwi (يوسف شتيوي)
# Description: Multi-stage Docker build for the complete sperm analysis system

# ==============================================================================
# Stage 1: Python Backend Build
# ==============================================================================
FROM python:3.11-slim as backend-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLOv8 model if custom model not available
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ==============================================================================
# Stage 2: Application Setup
# ==============================================================================
FROM backend-base as production

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    data/uploads \
    data/results \
    data/models \
    data/temp \
    logs \
    && chown -R appuser:appuser /app

# Copy startup script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown appuser:appuser /entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/simple || exit 1

# Start application
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Stage 3: Development Build
# ==============================================================================
FROM backend-base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio black flake8 mypy

# Copy application code
COPY . .

# Create directories
RUN mkdir -p \
    data/uploads \
    data/results \
    data/models \
    data/temp \
    logs

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true

# Expose port
EXPOSE 8000

# Start with hot reload
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ==============================================================================
# Stage 4: Training Environment
# ==============================================================================
FROM backend-base as training

# Install additional ML dependencies
RUN pip install jupyter notebook wandb tensorboard

# Install additional system packages for training
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Copy training code
COPY training/ /app/training/
COPY data/ /app/data/

# Create directories
RUN mkdir -p \
    data/datasets \
    data/models \
    training/logs \
    training/checkpoints

# Expose Jupyter port
EXPOSE 8888

# Default command for training
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]