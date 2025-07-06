#!/usr/bin/env python3
"""
DeepSORT Tracker for Sperm Tracking
Author: Youssef Shitiwi
Description: Multi-object tracking implementation for sperm analysis
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from deep_sort_realtime import DeepSort
from ultralytics import YOLO
import torch
from pathlib import Path
import json

@dataclass
class Detection:
    """Single detection data structure."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    frame_id: int

@dataclass
class Track:
    """Single track data structure."""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    frame_id: int
    centroid: Tuple[float, float]

class SpermTracker:
    """
    Sperm tracking system using YOLOv8 for detection and DeepSORT for tracking.
    """
    
    def __init__(self, 
                 model_path: str,
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.5,
                 max_age: int = 30,
                 n_init: int = 3):
        """
        Initialize the sperm tracker.
        
        Args:
            model_path: Path to trained YOLOv8 model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_age: Maximum age of tracks before deletion
            n_init: Number of consecutive detections before track confirmation
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize YOLO model
        self.yolo_model = YOLO(model_path)
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=torch.cuda.is_available(),
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Storage for tracks and trajectories
        self.tracks_history: Dict[int, List[Track]] = {}
        self.active_tracks: Dict[int, Track] = {}
        self.frame_count = 0
        
    def detect_sperm(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect sperm in a single frame using YOLOv8.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of Detection objects
        """
        results = self.yolo_model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detection = Detection(
                        bbox=tuple(box.astype(float)),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        frame_id=self.frame_count
                    )
                    detections.append(detection)
        
        return detections
    
    def update_tracks(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update tracks using DeepSORT algorithm.
        
        Args:
            detections: List of current frame detections
            frame: Current frame for feature extraction
            
        Returns:
            List of updated Track objects
        """
        # Convert detections to format expected by DeepSORT
        detection_list = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            detection_list.append([x1, y1, x2 - x1, y2 - y1, det.confidence])
        
        # Update tracker
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Convert tracks to our Track format
        current_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            x1, y1, x2, y2 = track.to_ltrb()
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            track_obj = Track(
                track_id=track.track_id,
                bbox=(x1, y1, x2, y2),
                confidence=track.get_det_conf() if hasattr(track, 'get_det_conf') else 1.0,
                class_id=0,  # Assuming single class (sperm)
                frame_id=self.frame_count,
                centroid=centroid
            )
            
            current_tracks.append(track_obj)
            
            # Update tracks history
            if track.track_id not in self.tracks_history:
                self.tracks_history[track.track_id] = []
            self.tracks_history[track.track_id].append(track_obj)
            
            # Update active tracks
            self.active_tracks[track.track_id] = track_obj
        
        # Remove inactive tracks
        active_track_ids = {track.track_id for track in current_tracks}
        self.active_tracks = {
            tid: track for tid, track in self.active_tracks.items() 
            if tid in active_track_ids
        }
        
        return current_tracks
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], List[Track]]:
        """
        Process a single frame: detect and track sperm.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (detections, tracks)
        """
        # Detect sperm in current frame
        detections = self.detect_sperm(frame)
        
        # Update tracks
        tracks = self.update_tracks(detections, frame)
        
        # Increment frame counter
        self.frame_count += 1
        
        return detections, tracks
    
    def get_track_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Get the trajectory (centroids) of a specific track.
        
        Args:
            track_id: ID of the track
            
        Returns:
            List of (x, y) centroid coordinates
        """
        if track_id not in self.tracks_history:
            return []
        
        trajectory = [track.centroid for track in self.tracks_history[track_id]]
        return trajectory
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Get trajectories for all tracks.
        
        Returns:
            Dictionary mapping track_id to list of centroid coordinates
        """
        trajectories = {}
        for track_id in self.tracks_history:
            trajectories[track_id] = self.get_track_trajectory(track_id)
        return trajectories
    
    def visualize_tracking(self, frame: np.ndarray, 
                          detections: List[Detection], 
                          tracks: List[Track],
                          show_trajectories: bool = True) -> np.ndarray:
        """
        Visualize detections and tracks on frame.
        
        Args:
            frame: Input frame
            detections: Current detections
            tracks: Current tracks
            show_trajectories: Whether to show track trajectories
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det.bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f'Conf: {det.confidence:.2f}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw tracks
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                 (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0)]
        
        for track in tracks:
            color = colors[track.track_id % len(colors)]
            x1, y1, x2, y2 = [int(coord) for coord in track.bbox]
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            cv2.putText(vis_frame, f'ID: {track.track_id}', 
                       (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw centroid
            cx, cy = [int(coord) for coord in track.centroid]
            cv2.circle(vis_frame, (cx, cy), 3, color, -1)
        
        # Draw trajectories
        if show_trajectories:
            for track_id, trajectory in self.get_all_trajectories().items():
                if len(trajectory) < 2:
                    continue
                    
                color = colors[track_id % len(colors)]
                points = [(int(x), int(y)) for x, y in trajectory[-20:]]  # Last 20 points
                
                for i in range(1, len(points)):
                    cv2.line(vis_frame, points[i-1], points[i], color, 2)
        
        # Add frame info
        cv2.putText(vis_frame, f'Frame: {self.frame_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Tracks: {len(tracks)}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis_frame
    
    def reset(self):
        """Reset tracker state."""
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
            nn_budget=None
        )
        self.tracks_history.clear()
        self.active_tracks.clear()
        self.frame_count = 0
    
    def save_tracking_results(self, output_path: str):
        """
        Save tracking results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        results = {
            'frame_count': self.frame_count,
            'total_tracks': len(self.tracks_history),
            'tracks': {}
        }
        
        for track_id, track_history in self.tracks_history.items():
            track_data = {
                'track_id': track_id,
                'duration': len(track_history),
                'trajectory': [
                    {
                        'frame_id': track.frame_id,
                        'bbox': track.bbox,
                        'centroid': track.centroid,
                        'confidence': track.confidence
                    }
                    for track in track_history
                ]
            }
            results['tracks'][track_id] = track_data
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_tracking_results(self, input_path: str):
        """
        Load tracking results from JSON file.
        
        Args:
            input_path: Path to load results from
        """
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        self.frame_count = results['frame_count']
        self.tracks_history.clear()
        
        for track_id_str, track_data in results['tracks'].items():
            track_id = int(track_id_str)
            track_history = []
            
            for point in track_data['trajectory']:
                track = Track(
                    track_id=track_id,
                    bbox=tuple(point['bbox']),
                    confidence=point['confidence'],
                    class_id=0,
                    frame_id=point['frame_id'],
                    centroid=tuple(point['centroid'])
                )
                track_history.append(track)
            
            self.tracks_history[track_id] = track_history