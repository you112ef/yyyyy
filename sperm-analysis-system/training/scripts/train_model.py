#!/usr/bin/env python3
"""
Sperm Detection Model Training Script
Author: Youssef Shitiwi
Description: Train YOLOv8 model for sperm detection using transfer learning
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class SpermModelTrainer:
    def __init__(self, config_path: str):
        """Initialize the trainer with configuration."""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_logging()
        self.setup_directories()
        
    def load_config(self) -> dict:
        """Load training configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("../logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger.add(
            log_file,
            rotation="100 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        logger.info("Training session started")
        logger.info(f"Configuration loaded from: {self.config_path}")
    
    def setup_directories(self):
        """Create necessary directories for training."""
        directories = [
            "../models/checkpoints",
            "../models/best",
            "../logs",
            "../../data/datasets/sperm_dataset/images/train",
            "../../data/datasets/sperm_dataset/images/val",
            "../../data/datasets/sperm_dataset/labels/train", 
            "../../data/datasets/sperm_dataset/labels/val"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Training directories created")
    
    def setup_wandb(self, project_name: str = "sperm-analysis"):
        """Initialize Weights & Biases for experiment tracking."""
        try:
            wandb.init(
                project=project_name,
                name=f"yolov8_sperm_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config
            )
            logger.info("W&B initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
            return False
    
    def prepare_dataset(self):
        """Prepare and validate dataset structure."""
        dataset_path = Path(self.config['path'])
        
        # Check if dataset exists
        if not dataset_path.exists():
            logger.warning(f"Dataset path {dataset_path} does not exist")
            logger.info("Creating sample dataset structure...")
            self.create_sample_dataset()
        
        # Validate dataset structure
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for req_dir in required_dirs:
            dir_path = dataset_path / req_dir
            if not dir_path.exists():
                logger.error(f"Required directory missing: {dir_path}")
                raise FileNotFoundError(f"Dataset directory not found: {dir_path}")
        
        logger.info("Dataset validation completed")
    
    def create_sample_dataset(self):
        """Create sample dataset structure for demonstration."""
        dataset_path = Path("../../data/datasets/sperm_dataset")
        
        # Create directories
        for split in ['train', 'val']:
            (dataset_path / f'images/{split}').mkdir(parents=True, exist_ok=True)
            (dataset_path / f'labels/{split}').mkdir(parents=True, exist_ok=True)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['sperm']
        }
        
        with open(dataset_path / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)
        
        logger.info(f"Sample dataset structure created at: {dataset_path}")
        logger.warning("Please add your training images and labels before training")
    
    def train_model(self):
        """Train the YOLOv8 model."""
        logger.info("Starting model training...")
        
        # Initialize model
        model = YOLO(self.config.get('model', 'yolov8n.pt'))
        
        # Training parameters
        train_params = {
            'data': Path(self.config['path']) / 'dataset.yaml',
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch', 16),
            'imgsz': self.config.get('imgsz', 640),
            'lr0': self.config.get('lr0', 0.01),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'momentum': self.config.get('momentum', 0.937),
            'patience': self.config.get('patience', 50),
            'save_period': self.config.get('save_period', -1),
            'project': '../models',
            'name': f'sperm_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': self.config.get('exist_ok', False),
            'pretrained': self.config.get('pretrained', True),
            'optimizer': self.config.get('optimizer', 'SGD'),
            'verbose': self.config.get('verbose', True),
            'seed': self.config.get('seed', 0),
            'deterministic': self.config.get('deterministic', True),
            'single_cls': self.config.get('single_cls', True),
            'amp': self.config.get('amp', True),
            'val': self.config.get('val', True),
            'plots': self.config.get('plots', True)
        }
        
        logger.info(f"Training parameters: {train_params}")
        
        try:
            # Train the model
            results = model.train(**train_params)
            
            # Save best model
            best_model_path = "../models/best/sperm_detection_best.pt"
            model.save(best_model_path)
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Best model saved to: {best_model_path}")
            logger.info(f"Training results: {results}")
            
            return results, best_model_path
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, model_path: str):
        """Validate the trained model."""
        logger.info("Validating trained model...")
        
        model = YOLO(model_path)
        
        val_results = model.val(
            data=Path(self.config['path']) / 'dataset.yaml',
            imgsz=self.config.get('imgsz', 640),
            batch=self.config.get('batch', 16),
            conf=self.config.get('conf', 0.001),
            iou=self.config.get('iou', 0.6),
            max_det=self.config.get('max_det', 300),
            half=self.config.get('half', False),
            save_json=True,
            plots=True
        )
        
        logger.info(f"Validation results: {val_results}")
        return val_results
    
    def export_model(self, model_path: str, formats: list = ['onnx', 'torchscript']):
        """Export model to different formats."""
        logger.info(f"Exporting model to formats: {formats}")
        
        model = YOLO(model_path)
        exported_models = {}
        
        for format_type in formats:
            try:
                exported_path = model.export(format=format_type)
                exported_models[format_type] = exported_path
                logger.info(f"Model exported to {format_type}: {exported_path}")
            except Exception as e:
                logger.error(f"Failed to export to {format_type}: {e}")
        
        return exported_models

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for sperm detection')
    parser.add_argument('--config', type=str, default='../configs/yolo_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--export', nargs='+', default=['onnx'],
                       help='Export formats after training')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize trainer
    trainer = SpermModelTrainer(args.config)
    
    # Setup W&B if requested
    if args.wandb:
        trainer.setup_wandb()
    
    try:
        # Prepare dataset
        trainer.prepare_dataset()
        
        # Train model
        results, model_path = trainer.train_model()
        
        # Validate model
        val_results = trainer.validate_model(model_path)
        
        # Export model
        if args.export:
            exported_models = trainer.export_model(model_path, args.export)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)
    
    finally:
        if args.wandb:
            wandb.finish()

if __name__ == "__main__":
    main()