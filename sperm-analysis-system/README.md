# 🧬 Sperm Analysis System - AI-Powered CASA Solution

## 👨‍💻 Developer
**Youssef Shitiwi (يوسف شتيوي)**
- Biomedical AI Engineer
- Specialized in Computer Vision and Medical Image Analysis
- Expert in Deep Learning for Healthcare Applications
- Contact: [LinkedIn](https://linkedin.com/in/youssef-shitiwi) | [GitHub](https://github.com/youssef-shitiwi)

## 📋 Project Overview

This project implements a complete Computer-Assisted Sperm Analysis (CASA) system using modern AI technologies. The system provides real-time sperm detection, tracking, and motility analysis through video processing.

### 🎯 Features
- **AI-Powered Detection**: YOLOv8-based sperm detection
- **Multi-Object Tracking**: DeepSORT for trajectory analysis
- **CASA Metrics**: VCL, VSL, LIN, MOT%, and sperm count
- **REST API**: FastAPI backend for video processing
- **Mobile Support**: Android application with APK
- **Docker Support**: Containerized deployment

### 🛠️ Technology Stack
- **Deep Learning**: PyTorch, YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV, Albumentations
- **Tracking**: DeepSORT
- **Backend**: FastAPI, Uvicorn
- **Data Processing**: NumPy, Pandas, SciPy
- **Mobile**: Android Native
- **Deployment**: Docker, Gunicorn

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Backend API
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Model Training
```bash
cd training
python scripts/train_model.py --config configs/yolo_config.yaml
```

### Android App
```bash
cd android
./gradlew assembleDebug
```

## 📊 API Endpoints

- `POST /analyze` - Upload and analyze sperm video
- `GET /status` - Health check
- `GET /results/{id}` - Get analysis results
- `GET /download/{id}` - Download CSV results

## 🔬 Supported Formats
- Video: MP4, AVI
- Output: JSON, CSV
- Models: PyTorch (.pt), ONNX

## 📱 Mobile Integration
The Android app provides:
- Video capture and upload
- Real-time analysis results
- Offline processing capability
- Report generation and sharing

## 🐳 Docker Deployment
```bash
docker build -t sperm-analysis .
docker run -p 8000:8000 sperm-analysis
```

## 📖 Documentation
- [API Documentation](docs/api.md)
- [Training Guide](docs/training.md)
- [Android Setup](docs/android.md)

## 🏥 Clinical Applications
This system is designed for:
- Fertility clinics
- Research laboratories
- Andrology centers
- Veterinary applications

## 📄 License
MIT License - See LICENSE file for details

## 🤝 Contributing
Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---
*Developed by Youssef Shitiwi - Advancing healthcare through AI innovation*