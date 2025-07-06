# 🧠 Custom Model Training - Complete Guide
# Developer: Youssef Shitiwi (يوسف شتيوي)

## 1. QUICK START - Train with Your Data

### Step 1: Prepare Your Dataset
```bash
# Create dataset structure
mkdir -p data/datasets/my_sperm_dataset/{images,labels}/{train,val,test}

# Your video files should be placed in:
mkdir -p data/videos/training_videos/
```

### Step 2: Convert Videos to Training Dataset
```python
# training/scripts/prepare_my_dataset.py
import cv2
import os
from pathlib import Path

def extract_frames_from_videos(video_dir, output_dir, interval=10):
    """Extract frames from your sperm videos for annotation."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in video_dir.glob("*.mp4"):
        print(f"Processing {video_path.name}...")
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save every nth frame for annotation
            if frame_count % interval == 0:
                frame_name = f"{video_path.stem}_frame_{saved_count:06d}.jpg"
                output_path = output_dir / frame_name
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
                
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames from {video_path.name}")

# Usage - Extract frames from your videos
extract_frames_from_videos(
    "data/videos/training_videos/",
    "data/datasets/my_sperm_dataset/images/temp_annotation/"
)
```

### Step 3: Annotate Your Data

#### Option A: Use LabelImg (Recommended)
```bash
# Install LabelImg
pip install labelImg

# Start annotation tool
labelImg data/datasets/my_sperm_dataset/images/temp_annotation/

# Instructions:
# 1. Draw boxes around each sperm
# 2. Label as "sperm" (class 0)
# 3. Save labels in YOLO format
# 4. Move annotated images to train/val folders (80/20 split)
```

#### Option B: Use CVAT (Web-based)
```bash
# Run CVAT with Docker
docker run -it --rm -p 8080:8080 cvat/server

# Access at http://localhost:8080
# Create project, upload images, annotate sperm
```

### Step 4: Organize Your Dataset
```bash
# training/scripts/organize_dataset.py
import os
import shutil
import random
from pathlib import Path

def organize_annotated_data(source_dir, dataset_dir, train_split=0.8):
    """Organize annotated data into train/val splits."""
    source_dir = Path(source_dir)
    dataset_dir = Path(dataset_dir)
    
    # Get all annotated images (those with corresponding .txt files)
    image_files = []
    for img_file in source_dir.glob("*.jpg"):
        label_file = img_file.with_suffix(".txt")
        if label_file.exists():
            image_files.append(img_file)
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Create directories
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Move files
    for files, split in [(train_files, 'train'), (val_files, 'val')]:
        for img_file in files:
            label_file = img_file.with_suffix(".txt")
            
            # Copy image and label
            shutil.copy2(img_file, dataset_dir / 'images' / split / img_file.name)
            shutil.copy2(label_file, dataset_dir / 'labels' / split / label_file.name)
    
    print(f"Dataset organized: {len(train_files)} train, {len(val_files)} val images")

# Usage
organize_annotated_data(
    "data/datasets/my_sperm_dataset/images/temp_annotation/",
    "data/datasets/my_sperm_dataset/"
)
```

### Step 5: Create Dataset Configuration
```yaml
# data/datasets/my_sperm_dataset/dataset.yaml
path: /app/data/datasets/my_sperm_dataset
train: images/train
val: images/val
test: images/test  # optional

# Classes
nc: 1
names: ['sperm']

# Dataset metadata
created_by: "Your Name"
description: "Custom sperm dataset for CASA analysis"
version: "1.0"
date_created: "2024-01-01"
```

## 2. ADVANCED TRAINING CONFIGURATIONS

### Custom Training Configuration
```yaml
# training/configs/my_custom_config.yaml
# Dataset
path: ../data/datasets/my_sperm_dataset

# Model selection based on your needs:
model: 'yolov8n.pt'  # Fast, smaller model (6MB)
# model: 'yolov8s.pt'  # Better accuracy (22MB)
# model: 'yolov8m.pt'  # High accuracy (50MB)
# model: 'yolov8l.pt'  # Best accuracy (87MB)

# Training parameters
epochs: 150  # Increase for better accuracy
batch: 16    # Adjust based on GPU memory
imgsz: 640   # Image size (increase for small objects)
lr0: 0.01    # Learning rate
weight_decay: 0.0005
momentum: 0.937

# Data augmentation for sperm (they can rotate freely)
degrees: 360     # Full rotation
flipud: 0.5      # Vertical flip
fliplr: 0.5      # Horizontal flip
translate: 0.1   # Translation
scale: 0.5       # Scale variation
hsv_h: 0.015     # Hue shift
hsv_s: 0.7       # Saturation
hsv_v: 0.4       # Value/brightness

# Advanced options for small objects (sperm)
mosaic: 0.8      # Mosaic augmentation
copy_paste: 0.3  # Copy-paste augmentation
perspective: 0.0 # Minimal perspective for microscope images

# Training behavior
patience: 30     # Early stopping patience
save_period: 10  # Save checkpoint every N epochs
amp: true        # Automatic Mixed Precision
```

### Specialized Configurations

#### For High-Resolution Microscope Videos
```yaml
# training/configs/high_res_config.yaml
model: 'yolov8s.pt'  # Better for high resolution
imgsz: 1280          # Higher resolution input
batch: 8             # Smaller batch for memory
lr0: 0.005           # Lower learning rate
```

#### For Low-Quality/Noisy Videos
```yaml
# training/configs/noisy_video_config.yaml
# Stronger augmentation to handle noise
hsv_h: 0.05
hsv_s: 0.9
hsv_v: 0.6
degrees: 360
blur: 0.01         # Add blur augmentation
noise: 0.02        # Add noise augmentation
```

## 3. TRAINING EXECUTION

### Method 1: Using Docker (Recommended)
```bash
# Start training environment
docker-compose --profile training up -d training

# Access Jupyter and run training
# Navigate to http://localhost:8888
# Run training notebook or use terminal
```

### Method 2: Direct Training Script
```bash
# Run training with your custom config
python training/scripts/train_model.py \
    --config training/configs/my_custom_config.yaml \
    --project my_sperm_model \
    --name experiment_v1 \
    --wandb

# Monitor training progress
tensorboard --logdir training/runs
```

### Method 3: Interactive Training Notebook
```python
# training/notebooks/custom_training.ipynb
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Train on your dataset
results = model.train(
    data='/app/data/datasets/my_sperm_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='my_sperm_model',
    project='experiments',
    save_period=10,
    plots=True
)

# Validate the model
validation_results = model.val()
print(f"mAP50: {validation_results.box.map50:.4f}")
print(f"mAP50-95: {validation_results.box.map:.4f}")

# Test on a sample image
test_results = model('data/test_images/sample_sperm.jpg')
test_results[0].show()  # Display results
```

## 4. TRAINING OPTIMIZATION

### Hyperparameter Optimization
```python
# training/scripts/optimize_hyperparameters.py
from ultralytics import YOLO
import optuna
import numpy as np

def objective(trial):
    # Suggest hyperparameters
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
    batch = trial.suggest_categorical('batch', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    degrees = trial.suggest_int('degrees', 0, 360)
    
    # Train model with suggested parameters
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='/app/data/datasets/my_sperm_dataset/dataset.yaml',
        epochs=50,  # Shorter for optimization
        lr0=lr0,
        batch=batch,
        weight_decay=weight_decay,
        degrees=degrees,
        save=False,
        plots=False,
        verbose=False
    )
    
    # Return mAP50 as optimization target
    return results.box.map50

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"Best mAP50: {study.best_value:.4f}")
```

### Progressive Training Strategy
```python
# training/scripts/progressive_training.py
from ultralytics import YOLO

# Stage 1: Train head only (frozen backbone)
print("Stage 1: Training head layers only...")
model = YOLO('yolov8n.pt')
model.train(
    data='/app/data/datasets/my_sperm_dataset/dataset.yaml',
    epochs=30,
    freeze=10,  # Freeze first 10 layers
    name='stage1_head_only'
)

# Stage 2: Fine-tune all layers with lower learning rate
print("Stage 2: Fine-tuning all layers...")
model.train(
    data='/app/data/datasets/my_sperm_dataset/dataset.yaml',
    epochs=70,
    freeze=0,   # Unfreeze all layers
    lr0=0.001,  # Lower learning rate
    name='stage2_full_finetune'
)

# Stage 3: Final fine-tuning with very low learning rate
print("Stage 3: Final refinement...")
model.train(
    data='/app/data/datasets/my_sperm_dataset/dataset.yaml',
    epochs=50,
    lr0=0.0001,  # Very low learning rate
    name='stage3_final'
)
```

## 5. MODEL VALIDATION & TESTING

### Comprehensive Validation
```python
# training/scripts/validate_my_model.py
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def validate_model_thoroughly(model_path, test_data_dir):
    """Comprehensive model validation."""
    model = YOLO(model_path)
    
    # 1. Standard validation metrics
    results = model.val(data='/app/data/datasets/my_sperm_dataset/dataset.yaml')
    
    print("=== Validation Metrics ===")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    # 2. Test on individual videos
    test_videos = Path(test_data_dir).glob("*.mp4")
    
    for video_path in test_videos:
        print(f"\n=== Testing on {video_path.name} ===")
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        detections_per_frame = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection
            detection_results = model(frame, verbose=False)
            num_detections = len(detection_results[0].boxes) if detection_results[0].boxes else 0
            detections_per_frame.append(num_detections)
            
            frame_count += 1
            
            # Process every 30th frame to save time
            if frame_count % 30 != 0:
                continue
        
        cap.release()
        
        # Statistics
        avg_detections = np.mean(detections_per_frame)
        max_detections = np.max(detections_per_frame)
        std_detections = np.std(detections_per_frame)
        
        print(f"  Average sperm per frame: {avg_detections:.2f}")
        print(f"  Maximum sperm detected: {max_detections}")
        print(f"  Detection stability (std): {std_detections:.2f}")

# Usage
validate_model_thoroughly(
    "training/runs/detect/my_sperm_model/weights/best.pt",
    "data/test_videos/"
)
```

### A/B Testing Different Models
```python
# training/scripts/model_comparison.py
from ultralytics import YOLO
import time

def compare_models(model_paths, test_video):
    """Compare different trained models."""
    results = {}
    
    for name, model_path in model_paths.items():
        print(f"\n=== Testing {name} ===")
        model = YOLO(model_path)
        
        # Speed test
        start_time = time.time()
        detection_results = model(test_video, save=False, verbose=False)
        inference_time = time.time() - start_time
        
        # Count detections
        total_detections = sum(
            len(result.boxes) if result.boxes else 0 
            for result in detection_results
        )
        
        results[name] = {
            'inference_time': inference_time,
            'total_detections': total_detections,
            'fps': len(detection_results) / inference_time
        }
        
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Total detections: {total_detections}")
        print(f"  Processing FPS: {results[name]['fps']:.2f}")
    
    return results

# Compare your models
model_comparison = compare_models({
    'YOLOv8n_custom': 'training/runs/detect/my_sperm_model_n/weights/best.pt',
    'YOLOv8s_custom': 'training/runs/detect/my_sperm_model_s/weights/best.pt',
    'Original_pretrained': 'yolov8n.pt'
}, 'data/test_videos/sample.mp4')
```

## 6. MODEL EXPORT & DEPLOYMENT

### Export Trained Model
```python
# training/scripts/export_my_model.py
from ultralytics import YOLO

def export_trained_model(model_path, export_formats=None):
    """Export your trained model to various formats."""
    if export_formats is None:
        export_formats = ['onnx', 'torchscript', 'tflite', 'engine']
    
    model = YOLO(model_path)
    
    exported_models = {}
    for format_type in export_formats:
        try:
            print(f"Exporting to {format_type}...")
            exported_path = model.export(format=format_type)
            exported_models[format_type] = exported_path
            print(f"✅ {format_type}: {exported_path}")
        except Exception as e:
            print(f"❌ Failed to export {format_type}: {e}")
    
    return exported_models

# Export your best model
exported = export_trained_model(
    'training/runs/detect/my_sperm_model/weights/best.pt',
    ['onnx', 'torchscript', 'tflite']
)

# Copy to deployment location
import shutil
shutil.copy2(
    'training/runs/detect/my_sperm_model/weights/best.pt',
    'data/models/best/sperm_detection_best.pt'
)
print("✅ Model deployed to production location")
```

## 7. CONTINUOUS TRAINING & IMPROVEMENT

### Active Learning Pipeline
```python
# training/scripts/active_learning.py
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def find_difficult_cases(model, video_dir, confidence_threshold=0.3):
    """Find frames where model struggles for additional annotation."""
    model = YOLO(model)
    difficult_frames = []
    
    for video_path in Path(video_dir).glob("*.mp4"):
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, verbose=False)
            
            # Check for difficult cases
            if results[0].boxes is not None:
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Frames with many low-confidence detections
                low_conf_count = np.sum(confidences < confidence_threshold)
                total_detections = len(confidences)
                
                if total_detections > 0 and (low_conf_count / total_detections) > 0.5:
                    difficult_frames.append({
                        'video': video_path.name,
                        'frame': frame_idx,
                        'avg_confidence': np.mean(confidences),
                        'low_conf_ratio': low_conf_count / total_detections
                    })
            
            frame_idx += 1
        
        cap.release()
    
    return difficult_frames

# Find difficult cases for additional annotation
difficult_cases = find_difficult_cases(
    'data/models/best/sperm_detection_best.pt',
    'data/new_videos/'
)

print(f"Found {len(difficult_cases)} difficult frames for annotation")
```

### Model Versioning & Experiment Tracking
```python
# training/scripts/experiment_tracking.py
import json
import datetime
from pathlib import Path

def track_experiment(experiment_name, config, results, model_path):
    """Track training experiments for comparison."""
    experiment_data = {
        'name': experiment_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config,
        'results': {
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr)
        },
        'model_path': str(model_path)
    }
    
    # Save experiment data
    experiments_file = Path('training/experiments.json')
    experiments = []
    
    if experiments_file.exists():
        with open(experiments_file, 'r') as f:
            experiments = json.load(f)
    
    experiments.append(experiment_data)
    
    with open(experiments_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"Experiment '{experiment_name}' tracked successfully")

# Usage after training
# track_experiment('my_sperm_model_v1', config_dict, results, model_path)
```

## 8. QUICK COMMANDS REFERENCE

```bash
# Quick training commands for common scenarios:

# Basic training with your dataset
python training/scripts/train_model.py --config training/configs/my_custom_config.yaml

# High-quality training (longer, better accuracy)
python training/scripts/train_model.py --config training/configs/my_custom_config.yaml --epochs 200

# Fast training for testing
python training/scripts/train_model.py --config training/configs/my_custom_config.yaml --epochs 30

# Resume interrupted training
python training/scripts/train_model.py --resume training/runs/detect/my_sperm_model/weights/last.pt

# Validate existing model
python -c "from ultralytics import YOLO; YOLO('path/to/model.pt').val()"

# Export model
python -c "from ultralytics import YOLO; YOLO('path/to/model.pt').export(format='onnx')"
```

## 🎯 Success Metrics

**Target Performance Goals:**
- mAP50 > 0.90 (90% accuracy at 50% IoU)
- Processing speed > 15 FPS on GPU
- Model size < 50MB for deployment
- False positive rate < 5%

**Training Completion Checklist:**
- [ ] Dataset properly annotated and organized
- [ ] Training converged without overfitting
- [ ] Validation metrics meet target goals
- [ ] Model tested on unseen videos
- [ ] Model exported to deployment formats
- [ ] Performance benchmarked
- [ ] Model integrated into backend API

**Developer: Youssef Shitiwi (يوسف شتيوي)**