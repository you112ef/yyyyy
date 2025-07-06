# 🧬 Sperm Analysis API Documentation

**Developer:** Youssef Shitiwi (يوسف شتيوي)  
**Version:** 1.0.0  
**Technology:** FastAPI + AI Models

## Overview

The Sperm Analysis API provides comprehensive Computer-Assisted Sperm Analysis (CASA) using state-of-the-art AI models. This RESTful API enables automated sperm detection, tracking, and motility analysis from video inputs.

## Base URL

- **Development:** `http://localhost:8000/api/v1`
- **Production:** `https://your-domain.com/api/v1`

## Authentication

Currently, the API operates without authentication for development. In production, implement API key authentication:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -X GET https://your-domain.com/api/v1/health
```

## API Endpoints

### 🏥 Health Check

#### GET `/health`
Get comprehensive system health information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600.5,
  "database_connected": true,
  "model_loaded": true,
  "gpu_available": true,
  "cpu_usage": 25.5,
  "memory_usage": 45.2,
  "disk_usage": 60.1,
  "pending_analyses": 2,
  "processing_analyses": 1
}
```

#### GET `/health/simple`
Simple health check for load balancers.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T12:00:00Z",
  "service": "sperm-analysis-api"
}
```

### 📤 File Upload

#### POST `/upload`
Upload a video file for analysis.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (file): Video file (MP4, AVI, MOV supported, max 500MB)

**Response:**
```json
{
  "filename": "sample_video.mp4",
  "size": 15728640,
  "format": "mp4",
  "duration": 30.5,
  "upload_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Example:**
```bash
curl -X POST \
  http://localhost:8000/api/v1/upload \
  -F "file=@video.mp4"
```

### 🔬 Analysis

#### POST `/analyze`
Start analysis on uploaded video.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `upload_id` (string): Upload ID from previous upload
- `analysis_name` (string, optional): Custom name for analysis
- `config` (string): Analysis configuration as JSON

**Config JSON Structure:**
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

**Response:**
```json
{
  "analysis_id": "456e7890-e89b-12d3-a456-426614174000",
  "status": "pending",
  "message": "Analysis queued for processing",
  "estimated_processing_time": 120.0,
  "created_at": "2024-01-01T12:00:00Z"
}
```

**Example:**
```bash
curl -X POST \
  http://localhost:8000/api/v1/analyze \
  -F "upload_id=123e4567-e89b-12d3-a456-426614174000" \
  -F "analysis_name=Sample Analysis" \
  -F 'config={"fps":30.0,"pixel_to_micron":0.5,"confidence_threshold":0.3}'
```

#### POST `/analyze-direct`
Upload and analyze in one request.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (file): Video file
- `analysis_name` (string, optional): Analysis name
- `fps` (float): Frame rate (default: 30.0)
- `pixel_to_micron` (float): Conversion factor (default: 1.0)
- `confidence_threshold` (float): Detection threshold (default: 0.3)
- `iou_threshold` (float): IoU threshold (default: 0.5)
- `min_track_length` (int): Minimum track length (default: 10)
- `enable_visualization` (boolean): Generate visualization (default: true)
- `export_trajectories` (boolean): Include trajectories (default: true)

**Example:**
```bash
curl -X POST \
  http://localhost:8000/api/v1/analyze-direct \
  -F "file=@video.mp4" \
  -F "analysis_name=Direct Analysis" \
  -F "fps=30.0" \
  -F "pixel_to_micron=0.5" \
  -F "confidence_threshold=0.3"
```

### 📊 Status & Results

#### GET `/status/{analysis_id}`
Get analysis status and progress.

**Response:**
```json
{
  "analysis_id": "456e7890-e89b-12d3-a456-426614174000",
  "status": "processing",
  "progress": 45.5,
  "message": "Processing frame 500/1000",
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:01:00Z",
  "current_frame": 500,
  "total_frames": 1000,
  "processing_stage": "frame_processing"
}
```

#### GET `/results/{analysis_id}`
Get complete analysis results.

**Response:**
```json
{
  "analysis_id": "456e7890-e89b-12d3-a456-426614174000",
  "status": "completed",
  "results": {
    "analysis_id": "456e7890-e89b-12d3-a456-426614174000",
    "timestamp": "2024-01-01T12:10:00Z",
    "video_filename": "sample_video.mp4",
    "video_duration": 30.5,
    "total_frames": 915,
    "fps": 30.0,
    "individual_sperm": [...],
    "population_statistics": {
      "total_sperm_count": 150,
      "motility_percentage": 65.5,
      "progressive_percentage": 45.2,
      "mean_vcl": 125.6,
      "mean_vsl": 89.3,
      "mean_lin": 71.2
    },
    "processing_time": 125.3,
    "model_version": "YOLOv8+DeepSORT v1.0"
  },
  "available_downloads": ["csv", "json", "visualization"]
}
```

### 📥 Downloads

#### GET `/download/{analysis_id}/{format_type}`
Download analysis results in specified format.

**Format Types:**
- `csv`: Sperm analysis data in CSV format
- `json`: Complete results in JSON format
- `statistics`: Population statistics in JSON
- `trajectories`: Trajectory data in JSON
- `visualization`: Annotated video with tracking

**Example:**
```bash
# Download CSV results
curl -X GET \
  http://localhost:8000/api/v1/download/456e7890-e89b-12d3-a456-426614174000/csv \
  -o analysis_results.csv

# Download visualization video
curl -X GET \
  http://localhost:8000/api/v1/download/456e7890-e89b-12d3-a456-426614174000/visualization \
  -o visualization.mp4
```

#### GET `/download/{analysis_id}`
List available download formats.

**Response:**
```json
{
  "analysis_id": "456e7890-e89b-12d3-a456-426614174000",
  "status": "completed",
  "available_downloads": [
    {
      "format": "csv",
      "url": "/api/v1/download/456e7890-e89b-12d3-a456-426614174000/csv",
      "description": "Sperm analysis data in CSV format",
      "content_type": "text/csv"
    },
    {
      "format": "visualization",
      "url": "/api/v1/download/456e7890-e89b-12d3-a456-426614174000/visualization",
      "description": "Annotated video with tracking visualization",
      "content_type": "video/mp4"
    }
  ],
  "total_formats": 2
}
```

### 🗑️ Cleanup

#### DELETE `/analysis/{analysis_id}`
Cancel a running analysis.

**Response:**
```json
{
  "message": "Analysis cancelled successfully",
  "analysis_id": "456e7890-e89b-12d3-a456-426614174000"
}
```

#### DELETE `/results/{analysis_id}?keep_results=true`
Clean up analysis data.

**Parameters:**
- `keep_results` (boolean): Whether to keep result files (default: true)

#### DELETE `/upload/{upload_id}`
Clean up uploaded file.

### 📋 Queue Management

#### GET `/queue`
Get processing queue status.

**Response:**
```json
{
  "queue_status": {
    "pending": 3,
    "processing": 1,
    "completed": 25,
    "failed": 2,
    "total": 31
  },
  "total_uploads": 50,
  "disk_usage": {
    "uploads_mb": 1250.5,
    "results_mb": 2380.7,
    "total_mb": 3631.2
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## CASA Metrics Explained

### Velocity Parameters
- **VCL (Curvilinear Velocity)**: Total path velocity (μm/s)
- **VSL (Straight-line Velocity)**: Net velocity from start to end (μm/s)  
- **VAP (Average Path Velocity)**: Velocity along smoothed path (μm/s)

### Motion Parameters
- **LIN (Linearity)**: VSL/VCL × 100 (%)
- **STR (Straightness)**: VSL/VAP × 100 (%)
- **WOB (Wobble)**: VAP/VCL × 100 (%)

### Path Parameters
- **ALH**: Amplitude of lateral head displacement (μm)
- **BCF**: Beat cross frequency (Hz)

### Motility Classification
- **Progressive**: VSL ≥ 25 μm/s and LIN ≥ 50%
- **Slow Progressive**: 5 ≤ VSL < 25 μm/s or LIN < 50%
- **Non-Progressive**: Motile but not progressive
- **Immotile**: VCL < 10 μm/s

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid input parameters
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: File too large
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

**Error Response Format:**
```json
{
  "error": "validation_error",
  "message": "Invalid video format",
  "details": {
    "field": "file",
    "value": "document.pdf"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Rate Limiting

- Upload: 10 requests per minute per IP
- Analysis: 5 concurrent analyses per user
- Downloads: 50 requests per minute per IP

## SDKs and Examples

### Python SDK Example
```python
import requests
import json

# Upload video
with open('video.mp4', 'rb') as f:
    upload_response = requests.post(
        'http://localhost:8000/api/v1/upload',
        files={'file': f}
    )
upload_data = upload_response.json()
upload_id = upload_data['upload_id']

# Start analysis
config = {
    "fps": 30.0,
    "pixel_to_micron": 0.5,
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5,
    "min_track_length": 10,
    "enable_visualization": True,
    "export_trajectories": True
}

analysis_response = requests.post(
    'http://localhost:8000/api/v1/analyze',
    data={
        'upload_id': upload_id,
        'analysis_name': 'Python SDK Test',
        'config': json.dumps(config)
    }
)
analysis_data = analysis_response.json()
analysis_id = analysis_data['analysis_id']

# Check status
status_response = requests.get(
    f'http://localhost:8000/api/v1/status/{analysis_id}'
)
print(status_response.json())

# Get results when completed
results_response = requests.get(
    f'http://localhost:8000/api/v1/results/{analysis_id}'
)
results = results_response.json()

# Download CSV
csv_response = requests.get(
    f'http://localhost:8000/api/v1/download/{analysis_id}/csv'
)
with open('results.csv', 'wb') as f:
    f.write(csv_response.content)
```

### JavaScript/Node.js Example
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const baseURL = 'http://localhost:8000/api/v1';

async function analyzeSpermVideo() {
    // Upload video
    const form = new FormData();
    form.append('file', fs.createReadStream('video.mp4'));
    
    const uploadResponse = await axios.post(`${baseURL}/upload`, form, {
        headers: form.getHeaders()
    });
    const uploadId = uploadResponse.data.upload_id;
    
    // Start analysis
    const analysisForm = new FormData();
    analysisForm.append('upload_id', uploadId);
    analysisForm.append('analysis_name', 'JS SDK Test');
    analysisForm.append('config', JSON.stringify({
        fps: 30.0,
        pixel_to_micron: 0.5,
        confidence_threshold: 0.3
    }));
    
    const analysisResponse = await axios.post(`${baseURL}/analyze`, analysisForm, {
        headers: analysisForm.getHeaders()
    });
    const analysisId = analysisResponse.data.analysis_id;
    
    // Poll for completion
    let status;
    do {
        await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
        const statusResponse = await axios.get(`${baseURL}/status/${analysisId}`);
        status = statusResponse.data.status;
        console.log(`Status: ${status} (${statusResponse.data.progress}%)`);
    } while (status === 'pending' || status === 'processing');
    
    if (status === 'completed') {
        // Get results
        const resultsResponse = await axios.get(`${baseURL}/results/${analysisId}`);
        console.log('Analysis completed:', resultsResponse.data.results.population_statistics);
        
        // Download CSV
        const csvResponse = await axios.get(`${baseURL}/download/${analysisId}/csv`, {
            responseType: 'arraybuffer'
        });
        fs.writeFileSync('results.csv', csvResponse.data);
    }
}

analyzeSpermVideo().catch(console.error);
```

## Support

For technical support or questions about the API:

- **Developer:** Youssef Shitiwi (يوسف شتيوي)
- **Documentation:** See `/docs` endpoint
- **Issues:** Create an issue in the project repository
- **Health Check:** Monitor `/api/v1/health` endpoint