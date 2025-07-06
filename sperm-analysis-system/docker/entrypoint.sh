#!/bin/bash
# Sperm Analysis System - Docker Entrypoint
# Author: Youssef Shitiwi (يوسف شتيوي)

set -e

echo "🧬 Starting Sperm Analysis System"
echo "Developer: Youssef Shitiwi (يوسف شتيوي)"
echo "======================================"

# Function to wait for service
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    
    echo "⏳ Waiting for $service to be ready..."
    while ! nc -z "$host" "$port"; do
        sleep 1
    done
    echo "✅ $service is ready!"
}

# Wait for database if enabled
if [ "$USE_DATABASE" = "true" ] && [ -n "$DATABASE_URL" ]; then
    # Extract host and port from DATABASE_URL
    DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
        wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL"
    fi
fi

# Wait for Redis if enabled
if [ "$USE_REDIS" = "true" ] && [ -n "$REDIS_URL" ]; then
    # Extract host and port from REDIS_URL
    REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
    REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    if [ -n "$REDIS_HOST" ] && [ -n "$REDIS_PORT" ]; then
        wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis"
    fi
fi

# Run database migrations if needed
if [ "$USE_DATABASE" = "true" ]; then
    echo "🔄 Running database setup..."
    python -c "
import asyncio
from backend.utils.database import init_database
asyncio.run(init_database())
print('✅ Database initialized!')
" || echo "⚠️  Database initialization failed (may already be initialized)"
fi

# Create necessary directories
echo "📁 Setting up directories..."
mkdir -p data/{uploads,results,models,temp} logs

# Download default model if custom model doesn't exist
MODEL_PATH="${MODEL_PATH:-data/models/best/sperm_detection_best.pt}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Downloading default YOLOv8 model..."
    python -c "
from ultralytics import YOLO
import os
os.makedirs(os.path.dirname('$MODEL_PATH'), exist_ok=True)
model = YOLO('yolov8n.pt')
print('✅ Default model ready!')
"
fi

# Set proper permissions
echo "🔒 Setting permissions..."
chmod -R 755 data/ logs/ 2>/dev/null || true

# Display configuration
echo "🔧 Configuration:"
echo "   Environment: ${ENVIRONMENT:-production}"
echo "   Debug: ${DEBUG:-false}"
echo "   Database: ${USE_DATABASE:-false}"
echo "   Redis: ${USE_REDIS:-false}"
echo "   Model: ${MODEL_PATH}"
echo "   Max Workers: ${MAX_CONCURRENT_ANALYSES:-2}"
echo "   Max Upload: ${MAX_UPLOAD_SIZE_MB:-500}MB"

echo ""
echo "🚀 Starting application..."
echo "   API will be available at: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/api/v1/health"

# Execute the main command
exec "$@"