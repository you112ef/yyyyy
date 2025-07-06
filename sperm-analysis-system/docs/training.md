# 🧠 AI Model Training Guide - Sperm Analysis System

**Developer:** Youssef Shitiwi (يوسف شتيوي)  
**Models:** YOLOv8 + DeepSORT for Sperm Detection & Tracking

## Overview

This guide covers the complete process of training custom AI models for sperm detection and tracking. The system uses YOLOv8 for object detection and DeepSORT for multi-object tracking to provide accurate Computer-Assisted Sperm Analysis (CASA).

## 🗂️ Dataset Preparation

### Dataset Structure
```
data/datasets/sperm_dataset/
├── images/
│   ├── train/
│   │   ├── video1_frame_001.jpg
│   │   ├── video1_frame_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── video2_frame_001.jpg
│   │   └── ...
│   └── test/ (optional)
├── labels/
│   ├── train/
│   │   ├── video1_frame_001.txt
│   │   ├── video1_frame_002.txt
│   │   └── ...
│   ├── val/
│   │   ├── video2_frame_001.txt
│   │   └── ...
│   └── test/ (optional)
└── dataset.yaml
```

### Annotation Format (YOLO)
Each label file should contain one line per sperm with format:
```
class_id center_x center_y width height
```

Example `video1_frame_001.txt`:
```
0 0.5123 0.3456 0.0234 0.0456
0 0.7891 0.6543 0.0187 0.0398
0 0.2345 0.8901 0.0245 0.0421
```

Where:
- `class_id`: 0 (sperm class)
- `center_x, center_y`: Normalized center coordinates (0-1)
- `width, height`: Normalized box dimensions (0-1)

### Dataset Configuration
Create `dataset.yaml`:
```yaml
# Sperm Dataset Configuration
path: /app/data/datasets/sperm_dataset
train: images/train
val: images/val
test: images/test  # optional

# Classes
nc: 1  # number of classes
names: ['sperm']  # class names

# Dataset info
created_by: "Youssef Shitiwi"
description: "Sperm detection dataset for CASA analysis"
version: "1.0"
```

## 🎥 Video to Dataset Conversion

### Automatic Frame Extraction
```python
# training/scripts/video_to_dataset.py
import cv2
import os
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, interval: int = 5):
    """Extract frames from video for annotation."""
    cap = cv2.VideoCapture(video_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save every nth frame
        if frame_count % interval == 0:
            frame_name = f"{Path(video_path).stem}_frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(output_path / frame_name), frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

# Usage
extract_frames("data/videos/sample1.mp4", "data/datasets/temp_frames", interval=10)
```

### Annotation Tools
Recommended annotation tools:
1. **LabelImg** - Free, open-source
2. **CVAT** - Web-based annotation
3. **Roboflow** - Cloud-based with automation
4. **Label Studio** - Versatile annotation platform

#### Using LabelImg
```bash
pip install labelImg
labelImg data/datasets/temp_frames
```

## 🚀 Model Training

### Environment Setup
```bash
# Using Docker (recommended)
docker-compose --profile training up -d training

# Access Jupyter at http://localhost:8888
# Or run training script directly:
docker-compose exec training python training/scripts/train_model.py
```

### Training Script Usage
```bash
# Basic training
python training/scripts/train_model.py --config training/configs/yolo_config.yaml

# With Weights & Biases logging
python training/scripts/train_model.py --config training/configs/yolo_config.yaml --wandb

# Export models after training
python training/scripts/train_model.py --config training/configs/yolo_config.yaml --export onnx torchscript
```

### Training Configuration

#### Basic Configuration (`training/configs/yolo_config.yaml`)
```yaml
# Model Configuration
model: 'yolov8n.pt'  # Base model (nano/small/medium/large/extra-large)
epochs: 100
batch: 16
imgsz: 640
lr0: 0.01
weight_decay: 0.0005
momentum: 0.937

# Data Augmentation
hsv_h: 0.015  # Hue augmentation
hsv_s: 0.7    # Saturation augmentation  
hsv_v: 0.4    # Value augmentation
degrees: 0.0  # Rotation (degrees)
translate: 0.1 # Translation (fraction)
scale: 0.5    # Scale (gain)
shear: 0.0    # Shear (degrees)
perspective: 0.0 # Perspective
flipud: 0.0   # Flip up-down
fliplr: 0.5   # Flip left-right
mosaic: 1.0   # Mosaic augmentation
mixup: 0.0    # MixUp augmentation

# Training Parameters
patience: 50  # Early stopping patience
optimizer: 'SGD'
cos_lr: false
warmup_epochs: 3
amp: true     # Automatic Mixed Precision
```

#### Advanced Configuration
```yaml
# For better accuracy on small objects (sperm)
model: 'yolov8s.pt'  # Use small model for better accuracy
imgsz: 1024          # Higher resolution
batch: 8             # Smaller batch due to higher resolution

# Specialized augmentation for sperm
degrees: 360         # Full rotation (sperm can be in any orientation)
translate: 0.2       # More translation variety
scale: 0.9           # Scale variation for different zoom levels
mosaic: 0.8          # Reduce mosaic to preserve small objects
copy_paste: 0.3      # Copy-paste augmentation for more variety

# Hyperparameter optimization
box: 7.5             # Box loss gain
cls: 0.5             # Classification loss gain
dfl: 1.5             # Distribution focal loss gain
```

### Transfer Learning Strategy

#### 1. Start with Pre-trained COCO Model
```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # COCO pre-trained

# Fine-tune on sperm dataset
results = model.train(
    data='data/datasets/sperm_dataset/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)
```

#### 2. Progressive Training
```python
# Stage 1: Train head only (frozen backbone)
model = YOLO('yolov8n.pt')
model.train(
    data='dataset.yaml',
    epochs=20,
    freeze=10  # Freeze first 10 layers
)

# Stage 2: Fine-tune all layers
model.train(
    data='dataset.yaml',
    epochs=50,
    freeze=0   # Unfreeze all layers
)
```

### Multi-Scale Training
```yaml
# For handling various video resolutions
multiscale: true
scale_range: [0.5, 1.5]  # Scale range for multiscale training
```

## 📊 Training Monitoring

### Weights & Biases Integration
```python
import wandb

# Initialize wandb
wandb.init(
    project="sperm-analysis",
    name="yolov8_experiment_1",
    config={
        "model": "yolov8n",
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.01
    }
)

# Training automatically logs to wandb
model.train(data='dataset.yaml', epochs=100, project='sperm-analysis')
```

### TensorBoard Logging
```bash
# Start TensorBoard
tensorboard --logdir training/runs

# View at http://localhost:6006
```

### Key Metrics to Monitor
- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Box Loss**: Bounding box regression loss
- **Class Loss**: Classification loss
- **DFL Loss**: Distribution focal loss

## 🔧 Hyperparameter Optimization

### Automated Hyperparameter Tuning
```python
# training/scripts/hyperparameter_tuning.py
from ultralytics import YOLO
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
    batch = trial.suggest_categorical('batch', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Train model
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=50,
        lr0=lr0,
        batch=batch,
        weight_decay=weight_decay,
        save=False,
        plots=False
    )
    
    # Return validation mAP50
    return results.metrics['mAP50(B)']

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best parameters:", study.best_params)
```

### Grid Search
```python
# training/scripts/grid_search.py
import itertools
from ultralytics import YOLO

# Define parameter grid
param_grid = {
    'lr0': [0.001, 0.01, 0.1],
    'batch': [8, 16, 32],
    'weight_decay': [0.0001, 0.0005, 0.001]
}

best_map = 0
best_params = {}

for params in itertools.product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=30,
        **param_dict,
        save=False
    )
    
    current_map = results.metrics['mAP50(B)']
    if current_map > best_map:
        best_map = current_map
        best_params = param_dict
        
print(f"Best mAP50: {best_map}")
print(f"Best parameters: {best_params}")
```

## 🎯 Advanced Training Techniques

### Knowledge Distillation
```python
# training/scripts/knowledge_distillation.py
from ultralytics import YOLO

# Teacher model (larger, more accurate)
teacher = YOLO('yolov8l.pt')
teacher.train(data='dataset.yaml', epochs=100, name='teacher')

# Student model (smaller, faster)
student = YOLO('yolov8n.pt')

# Custom training loop with distillation
# Implementation would involve custom loss function
# combining detection loss and distillation loss
```

### Data Augmentation Strategies

#### Online Augmentation Pipeline
```python
# training/augmentation/custom_augments.py
import albumentations as A
import cv2

def get_sperm_augmentation():
    """Custom augmentation pipeline for sperm detection."""
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=360, p=0.8),  # Sperm can be in any orientation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=0,
            p=0.8
        ),
        
        # Optical distortions (simulate microscope effects)
        A.OpticalDistortion(
            distort_limit=0.1,
            shift_limit=0.1,
            p=0.3
        ),
        A.GridDistortion(p=0.3),
        A.ElasticTransform(p=0.3),
        
        # Color and contrast (simulate different lighting conditions)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.CLAHE(clip_limit=2.0, p=0.3),
        
        # Noise (simulate camera noise)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ISONoise(p=0.3),
        
        # Blur (simulate motion or focus issues)
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=0.2),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))
```

#### Synthetic Data Generation
```python
# training/data_generation/synthetic_sperm.py
import numpy as np
import cv2
from typing import List, Tuple

class SyntheticSpermGenerator:
    """Generate synthetic sperm images for data augmentation."""
    
    def __init__(self, image_size: Tuple[int, int] = (640, 640)):
        self.image_size = image_size
    
    def generate_sperm_shape(self) -> np.ndarray:
        """Generate realistic sperm shape."""
        # Head (ellipse)
        head_width = np.random.randint(8, 15)
        head_height = np.random.randint(12, 20)
        
        # Tail (bezier curve)
        tail_length = np.random.randint(80, 120)
        tail_width = np.random.randint(2, 4)
        
        # Create sperm silhouette
        sperm = np.zeros((150, 150), dtype=np.uint8)
        
        # Draw head
        cv2.ellipse(sperm, (75, 30), (head_width//2, head_height//2), 0, 0, 360, 255, -1)
        
        # Draw tail with curvature
        points = []
        for i in range(tail_length):
            x = 75 + int(5 * np.sin(i * 0.1))  # Add curvature
            y = 50 + i
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(sperm, [points], False, 255, tail_width)
        
        return sperm
    
    def generate_dataset(self, num_images: int, sperm_per_image: Tuple[int, int] = (5, 25)):
        """Generate synthetic dataset."""
        images = []
        labels = []
        
        for i in range(num_images):
            # Create background
            img = np.random.randint(0, 50, (*self.image_size, 3), dtype=np.uint8)
            
            # Add noise
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            # Generate sperm count for this image
            num_sperm = np.random.randint(*sperm_per_image)
            image_labels = []
            
            for _ in range(num_sperm):
                # Generate sperm
                sperm = self.generate_sperm_shape()
                
                # Random position
                x = np.random.randint(0, self.image_size[1] - sperm.shape[1])
                y = np.random.randint(0, self.image_size[0] - sperm.shape[0])
                
                # Random rotation
                angle = np.random.randint(0, 360)
                M = cv2.getRotationMatrix2D((sperm.shape[1]//2, sperm.shape[0]//2), angle, 1)
                sperm = cv2.warpAffine(sperm, M, (sperm.shape[1], sperm.shape[0]))
                
                # Random brightness
                brightness = np.random.randint(100, 255)
                sperm = (sperm > 0).astype(np.uint8) * brightness
                
                # Place on image
                roi = img[y:y+sperm.shape[0], x:x+sperm.shape[1]]
                mask = sperm > 0
                roi[mask] = sperm[mask]
                
                # Create label (YOLO format)
                center_x = (x + sperm.shape[1]//2) / self.image_size[1]
                center_y = (y + sperm.shape[0]//2) / self.image_size[0]
                width = sperm.shape[1] / self.image_size[1]
                height = sperm.shape[0] / self.image_size[0]
                
                image_labels.append([0, center_x, center_y, width, height])
            
            images.append(img)
            labels.append(image_labels)
        
        return images, labels
```

## 🧪 Model Validation & Testing

### Validation Metrics
```python
# training/scripts/validate_model.py
from ultralytics import YOLO
import numpy as np

def validate_model(model_path: str, test_data: str):
    """Comprehensive model validation."""
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=test_data)
    
    # Print metrics
    print(f"mAP50: {results.metrics['mAP50(B)']:.4f}")
    print(f"mAP50-95: {results.metrics['mAP50-95(B)']:.4f}")
    print(f"Precision: {results.metrics['precision(B)']:.4f}")
    print(f"Recall: {results.metrics['recall(B)']:.4f}")
    
    # Per-class metrics
    if hasattr(results, 'ap_class_index'):
        for i, class_idx in enumerate(results.ap_class_index):
            print(f"Class {class_idx} AP50: {results.ap50[i]:.4f}")
    
    return results

# Size-based analysis
def analyze_by_size(results, size_ranges=[(0, 32), (32, 96), (96, float('inf'))]):
    """Analyze performance by object size."""
    for i, (min_size, max_size) in enumerate(size_ranges):
        size_name = f"{min_size}-{max_size if max_size != float('inf') else '∞'}"
        # Implementation depends on YOLOv8 results structure
        print(f"Size range {size_name}: AP50 = {results.ap50[i]:.4f}")
```

### Cross-Validation
```python
# training/scripts/cross_validation.py
from sklearn.model_selection import KFold
import os
import shutil

def k_fold_validation(dataset_path: str, k: int = 5):
    """Perform k-fold cross-validation."""
    results = []
    
    # Split dataset
    images_dir = os.path.join(dataset_path, 'images/train')
    labels_dir = os.path.join(dataset_path, 'labels/train')
    
    image_files = os.listdir(images_dir)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        print(f"Training fold {fold + 1}/{k}")
        
        # Create fold directories
        fold_dir = f"fold_{fold}"
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data
        train_files = [image_files[i] for i in train_idx]
        val_files = [image_files[i] for i in val_idx]
        
        # Create fold dataset
        create_fold_dataset(fold_dir, train_files, val_files, images_dir, labels_dir)
        
        # Train model
        model = YOLO('yolov8n.pt')
        fold_results = model.train(
            data=f'{fold_dir}/dataset.yaml',
            epochs=50,
            name=f'fold_{fold}'
        )
        
        results.append(fold_results.metrics['mAP50(B)'])
        
        # Cleanup
        shutil.rmtree(fold_dir)
    
    print(f"CV Results: {np.mean(results):.4f} ± {np.std(results):.4f}")
    return results
```

## 📈 Model Optimization

### Model Pruning
```python
# training/optimization/pruning.py
import torch
import torch.nn.utils.prune as prune

def prune_model(model_path: str, pruning_ratio: float = 0.3):
    """Apply magnitude-based pruning to reduce model size."""
    model = torch.load(model_path)
    
    # Apply pruning to convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    
    # Save pruned model
    pruned_path = model_path.replace('.pt', '_pruned.pt')
    torch.save(model, pruned_path)
    
    return pruned_path
```

### Quantization
```python
# training/optimization/quantization.py
from ultralytics import YOLO

def quantize_model(model_path: str):
    """Convert model to quantized version for faster inference."""
    model = YOLO(model_path)
    
    # Export to different quantized formats
    model.export(format='onnx', int8=True)  # INT8 quantization
    model.export(format='tflite', int8=True)  # TensorFlow Lite INT8
    model.export(format='engine', half=True)  # TensorRT FP16
    
    print("Quantized models exported")
```

## 🚀 Model Deployment

### Export Formats
```python
# training/scripts/export_model.py
from ultralytics import YOLO

def export_trained_model(model_path: str):
    """Export trained model to various formats."""
    model = YOLO(model_path)
    
    # Export options
    formats = {
        'onnx': {'dynamic': True, 'simplify': True},
        'torchscript': {},
        'tflite': {'int8': True},
        'engine': {'half': True, 'workspace': 4},  # TensorRT
        'coreml': {},  # Core ML for iOS
        'paddle': {},  # PaddlePaddle
    }
    
    for format_name, kwargs in formats.items():
        try:
            exported_path = model.export(format=format_name, **kwargs)
            print(f"Exported {format_name}: {exported_path}")
        except Exception as e:
            print(f"Failed to export {format_name}: {e}")

# Usage
export_trained_model('training/runs/detect/train/weights/best.pt')
```

### Model Benchmarking
```python
# training/scripts/benchmark.py
import time
import numpy as np
from ultralytics import YOLO

def benchmark_model(model_path: str, image_size: int = 640, num_runs: int = 100):
    """Benchmark model inference speed."""
    model = YOLO(model_path)
    
    # Warm up
    dummy_input = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    for _ in range(10):
        model(dummy_input, verbose=False)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        results = model(dummy_input, verbose=False)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
    print(f"Std deviation: {np.std(times)*1000:.2f} ms")
    
    return avg_time, fps
```

## 📊 Training Best Practices

### Data Quality Guidelines
1. **Consistent Annotation**: Use the same annotation style across all images
2. **Balanced Dataset**: Ensure good representation of different scenarios
3. **Quality Control**: Review annotations for accuracy
4. **Diversity**: Include various lighting conditions, magnifications, and sperm concentrations

### Training Tips
1. **Start Small**: Begin with a subset of data to validate the pipeline
2. **Monitor Overfitting**: Watch validation metrics closely
3. **Learning Rate**: Use learning rate schedulers for better convergence
4. **Early Stopping**: Implement patience to avoid overfitting
5. **Ensemble Methods**: Combine multiple models for better accuracy

### Hardware Recommendations
- **GPU**: RTX 3090/4090 or Tesla V100/A100 for training
- **RAM**: 32GB+ system RAM
- **Storage**: SSD for fast data loading
- **VRAM**: 8GB+ GPU memory for batch training

## 🔍 Troubleshooting Training Issues

### Common Problems

#### Low mAP Scores
- Increase training epochs
- Improve data quality and annotation accuracy
- Try different model sizes (yolov8s, yolov8m)
- Adjust confidence thresholds
- Add more diverse training data

#### Overfitting
- Reduce model complexity
- Increase data augmentation
- Add regularization (weight decay)
- Use early stopping
- Increase dataset size

#### Training Instability
- Reduce learning rate
- Use gradient clipping
- Check for corrupted images/labels
- Ensure proper data loading

#### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training (AMP)
- Use smaller image sizes during training

## 📋 Training Checklist

Before starting training:
- [ ] Dataset is properly structured
- [ ] Annotations are validated
- [ ] Configuration file is set up
- [ ] Training environment is ready
- [ ] Monitoring tools are configured
- [ ] Backup strategy is in place

During training:
- [ ] Monitor training metrics
- [ ] Check for overfitting
- [ ] Validate on held-out data
- [ ] Save checkpoints regularly
- [ ] Log experiments properly

After training:
- [ ] Validate final model
- [ ] Export to deployment formats
- [ ] Benchmark inference speed
- [ ] Document model performance
- [ ] Archive training artifacts

---

**Next Steps:** After training, proceed to [API Documentation](api.md) for integrating your trained model into the analysis pipeline.

**Support:** For training assistance, contact Youssef Shitiwi (يوسف شتيوي)  
**Resources:** Additional training resources and pre-trained models available in the project repository.