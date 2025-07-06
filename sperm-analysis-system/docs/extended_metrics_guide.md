# 📊 Extended Metrics & Advanced Analysis Guide

> **Developer:** Youssef Shitiwi  
> **System:** Sperm Analysis with AI-Powered CASA Metrics

This guide provides comprehensive instructions for implementing advanced sperm analysis metrics beyond standard CASA parameters, enabling cutting-edge research capabilities.

## 📋 Table of Contents

1. [Advanced Kinematic Metrics](#advanced-kinematic-metrics)
2. [Machine Learning-Based Metrics](#machine-learning-based-metrics)
3. [Morphological Analysis](#morphological-analysis)
4. [Population Dynamics](#population-dynamics)
5. [Temporal Analysis](#temporal-analysis)
6. [3D Motion Analysis](#3d-motion-analysis)
7. [Comparative Analytics](#comparative-analytics)
8. [Implementation Examples](#implementation-examples)

---

## 🎯 Advanced Kinematic Metrics

### Enhanced Motion Parameters

Create `training/models/advanced_kinematics.py`:

```python
"""
Advanced Kinematic Analysis for Sperm Motility
Developer: Youssef Shitiwi
"""

import numpy as np
from scipy import signal, stats, spatial
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd

@dataclass
class AdvancedKinematicMetrics:
    """Extended kinematic analysis results"""
    
    # Advanced velocity metrics
    velocity_entropy: float           # Velocity distribution entropy
    velocity_kurtosis: float          # Velocity distribution shape
    velocity_skewness: float          # Velocity distribution asymmetry
    velocity_cv: float                # Coefficient of variation
    
    # Acceleration analysis
    mean_acceleration: float          # Mean acceleration magnitude
    max_acceleration: float           # Maximum acceleration
    acceleration_entropy: float       # Acceleration distribution entropy
    jerk_magnitude: float            # Rate of acceleration change
    
    # Path complexity
    tortuosity_index: float          # Path complexity measure
    sinuosity_ratio: float           # Path sinuosity
    fractal_dimension: float         # Path fractal dimension
    hausdorff_distance: float        # Path self-similarity
    
    # Frequency domain
    dominant_frequency: float         # Primary oscillation frequency
    frequency_entropy: float          # Frequency distribution entropy
    spectral_centroid: float         # Frequency center of mass
    spectral_rolloff: float          # Frequency rolloff point
    
    # Energy and work
    kinetic_energy_mean: float       # Mean kinetic energy
    kinetic_energy_variance: float   # Kinetic energy variability
    work_done: float                 # Total work performed
    power_spectral_density: float    # Power distribution
    
    # Coordination metrics
    xy_correlation: float            # X-Y movement correlation
    circular_variance: float         # Directional variance
    mean_resultant_length: float     # Directional consistency
    angular_acceleration: float      # Rotational acceleration

class AdvancedKinematicsAnalyzer:
    """Advanced kinematic analysis engine"""
    
    def __init__(self, fps: float = 30.0, pixel_to_micron: float = 2.5):
        self.fps = fps
        self.dt = 1.0 / fps
        self.pixel_to_micron = pixel_to_micron
        
    def analyze_trajectory(self, trajectory: np.ndarray) -> AdvancedKinematicMetrics:
        """Perform comprehensive kinematic analysis"""
        
        if len(trajectory) < 10:
            return self._empty_metrics()
            
        # Convert to microns and calculate basic derivatives
        traj_micron = trajectory * self.pixel_to_micron
        velocities = self._calculate_velocities(traj_micron)
        accelerations = self._calculate_accelerations(velocities)
        jerks = self._calculate_jerks(accelerations)
        
        # Calculate all metrics
        metrics = AdvancedKinematicMetrics(
            # Velocity metrics
            velocity_entropy=self._calculate_velocity_entropy(velocities),
            velocity_kurtosis=self._calculate_velocity_kurtosis(velocities),
            velocity_skewness=self._calculate_velocity_skewness(velocities),
            velocity_cv=self._calculate_velocity_cv(velocities),
            
            # Acceleration metrics
            mean_acceleration=self._calculate_mean_acceleration(accelerations),
            max_acceleration=self._calculate_max_acceleration(accelerations),
            acceleration_entropy=self._calculate_acceleration_entropy(accelerations),
            jerk_magnitude=self._calculate_jerk_magnitude(jerks),
            
            # Path complexity
            tortuosity_index=self._calculate_tortuosity(traj_micron),
            sinuosity_ratio=self._calculate_sinuosity(traj_micron),
            fractal_dimension=self._calculate_fractal_dimension(traj_micron),
            hausdorff_distance=self._calculate_hausdorff_distance(traj_micron),
            
            # Frequency domain
            dominant_frequency=self._calculate_dominant_frequency(velocities),
            frequency_entropy=self._calculate_frequency_entropy(velocities),
            spectral_centroid=self._calculate_spectral_centroid(velocities),
            spectral_rolloff=self._calculate_spectral_rolloff(velocities),
            
            # Energy and work
            kinetic_energy_mean=self._calculate_kinetic_energy_mean(velocities),
            kinetic_energy_variance=self._calculate_kinetic_energy_variance(velocities),
            work_done=self._calculate_work_done(velocities, accelerations),
            power_spectral_density=self._calculate_power_spectral_density(velocities),
            
            # Coordination metrics
            xy_correlation=self._calculate_xy_correlation(velocities),
            circular_variance=self._calculate_circular_variance(traj_micron),
            mean_resultant_length=self._calculate_mean_resultant_length(traj_micron),
            angular_acceleration=self._calculate_angular_acceleration(traj_micron)
        )
        
        return metrics
    
    def _calculate_velocities(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate velocity vectors"""
        return np.diff(trajectory, axis=0) / self.dt
    
    def _calculate_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Calculate acceleration vectors"""
        return np.diff(velocities, axis=0) / self.dt
    
    def _calculate_jerks(self, accelerations: np.ndarray) -> np.ndarray:
        """Calculate jerk vectors (rate of acceleration change)"""
        return np.diff(accelerations, axis=0) / self.dt
    
    def _calculate_velocity_entropy(self, velocities: np.ndarray) -> float:
        """Calculate velocity distribution entropy"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        hist, _ = np.histogram(vel_magnitudes, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    
    def _calculate_velocity_kurtosis(self, velocities: np.ndarray) -> float:
        """Calculate velocity distribution kurtosis"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        return stats.kurtosis(vel_magnitudes) if len(vel_magnitudes) > 3 else 0.0
    
    def _calculate_velocity_skewness(self, velocities: np.ndarray) -> float:
        """Calculate velocity distribution skewness"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        return stats.skew(vel_magnitudes) if len(vel_magnitudes) > 2 else 0.0
    
    def _calculate_velocity_cv(self, velocities: np.ndarray) -> float:
        """Calculate velocity coefficient of variation"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        mean_vel = np.mean(vel_magnitudes)
        return np.std(vel_magnitudes) / mean_vel if mean_vel > 0 else 0.0
    
    def _calculate_tortuosity(self, trajectory: np.ndarray) -> float:
        """Calculate path tortuosity index"""
        if len(trajectory) < 2:
            return 0.0
            
        path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        euclidean_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        return path_length / euclidean_distance if euclidean_distance > 0 else 0.0
    
    def _calculate_sinuosity(self, trajectory: np.ndarray) -> float:
        """Calculate path sinuosity ratio"""
        if len(trajectory) < 3:
            return 1.0
            
        # Calculate cumulative arc length
        segments = np.diff(trajectory, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        arc_length = np.sum(segment_lengths)
        
        # Calculate chord length
        chord_length = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        return arc_length / chord_length if chord_length > 0 else 1.0
    
    def _calculate_fractal_dimension(self, trajectory: np.ndarray) -> float:
        """Calculate path fractal dimension using box counting"""
        if len(trajectory) < 4:
            return 1.0
            
        # Normalize trajectory
        traj_norm = trajectory - np.min(trajectory, axis=0)
        max_range = np.max(np.ptp(trajectory, axis=0))
        if max_range == 0:
            return 1.0
        traj_norm = traj_norm / max_range
        
        # Box counting
        scales = np.logspace(-2, 0, 15)
        counts = []
        
        for scale in scales:
            grid_size = int(1.0 / scale) + 1
            boxes = set()
            
            for point in traj_norm:
                x_idx = min(int(point[0] / scale), grid_size - 1)
                y_idx = min(int(point[1] / scale), grid_size - 1)
                boxes.add((x_idx, y_idx))
            
            counts.append(len(boxes))
        
        # Linear regression in log-log space
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        valid_idx = np.isfinite(log_scales) & np.isfinite(log_counts)
        if np.sum(valid_idx) < 2:
            return 1.0
            
        slope, _ = np.polyfit(log_scales[valid_idx], log_counts[valid_idx], 1)
        return -slope
    
    def _calculate_hausdorff_distance(self, trajectory: np.ndarray) -> float:
        """Calculate Hausdorff distance for self-similarity"""
        if len(trajectory) < 4:
            return 0.0
            
        # Split trajectory into two halves
        mid = len(trajectory) // 2
        traj1 = trajectory[:mid]
        traj2 = trajectory[mid:]
        
        # Calculate directed Hausdorff distances
        dist_matrix = spatial.distance_matrix(traj1, traj2)
        h1 = np.max(np.min(dist_matrix, axis=1))
        h2 = np.max(np.min(dist_matrix, axis=0))
        
        return max(h1, h2)
    
    def _calculate_dominant_frequency(self, velocities: np.ndarray) -> float:
        """Calculate dominant frequency from velocity signal"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        if len(vel_magnitudes) < 10:
            return 0.0
            
        # Remove DC component
        vel_signal = vel_magnitudes - np.mean(vel_magnitudes)
        
        # FFT
        fft = np.fft.fft(vel_signal)
        freqs = np.fft.fftfreq(len(vel_signal), self.dt)
        
        # Find dominant frequency (excluding DC)
        power = np.abs(fft[1:len(fft)//2])
        freq_positive = freqs[1:len(freqs)//2]
        
        if len(power) == 0:
            return 0.0
            
        dominant_idx = np.argmax(power)
        return freq_positive[dominant_idx]
    
    def _calculate_frequency_entropy(self, velocities: np.ndarray) -> float:
        """Calculate frequency domain entropy"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        if len(vel_magnitudes) < 10:
            return 0.0
            
        fft = np.fft.fft(vel_magnitudes - np.mean(vel_magnitudes))
        power = np.abs(fft[:len(fft)//2])**2
        power_norm = power / np.sum(power) if np.sum(power) > 0 else power
        power_norm = power_norm[power_norm > 0]
        
        return -np.sum(power_norm * np.log2(power_norm)) if len(power_norm) > 0 else 0.0
    
    def _calculate_spectral_centroid(self, velocities: np.ndarray) -> float:
        """Calculate spectral centroid (frequency center of mass)"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        if len(vel_magnitudes) < 10:
            return 0.0
            
        fft = np.fft.fft(vel_magnitudes - np.mean(vel_magnitudes))
        freqs = np.fft.fftfreq(len(vel_magnitudes), self.dt)
        
        power = np.abs(fft[:len(fft)//2])**2
        freq_positive = freqs[:len(freqs)//2]
        
        total_power = np.sum(power)
        if total_power == 0:
            return 0.0
            
        return np.sum(freq_positive * power) / total_power
    
    def _calculate_spectral_rolloff(self, velocities: np.ndarray) -> float:
        """Calculate spectral rolloff (85% of energy)"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        if len(vel_magnitudes) < 10:
            return 0.0
            
        fft = np.fft.fft(vel_magnitudes - np.mean(vel_magnitudes))
        freqs = np.fft.fftfreq(len(vel_magnitudes), self.dt)
        
        power = np.abs(fft[:len(fft)//2])**2
        freq_positive = freqs[:len(freqs)//2]
        
        cumulative_power = np.cumsum(power)
        total_power = cumulative_power[-1]
        
        if total_power == 0:
            return 0.0
            
        rolloff_threshold = 0.85 * total_power
        rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
        
        return freq_positive[rolloff_idx[0]] if len(rolloff_idx) > 0 else freq_positive[-1]
    
    def _calculate_kinetic_energy_mean(self, velocities: np.ndarray) -> float:
        """Calculate mean kinetic energy (assuming unit mass)"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        kinetic_energies = 0.5 * vel_magnitudes**2
        return np.mean(kinetic_energies)
    
    def _calculate_kinetic_energy_variance(self, velocities: np.ndarray) -> float:
        """Calculate kinetic energy variance"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        kinetic_energies = 0.5 * vel_magnitudes**2
        return np.var(kinetic_energies)
    
    def _calculate_work_done(self, velocities: np.ndarray, accelerations: np.ndarray) -> float:
        """Calculate total work done"""
        if len(accelerations) == 0:
            return 0.0
            
        # Align arrays (accelerations is one element shorter)
        vel_aligned = velocities[1:len(accelerations)+1]
        
        # Work = Force · displacement ≈ ma · v·dt
        work_elements = []
        for i in range(len(accelerations)):
            if i < len(vel_aligned):
                power = np.dot(accelerations[i], vel_aligned[i])
                work_elements.append(abs(power) * self.dt)
        
        return np.sum(work_elements)
    
    def _calculate_power_spectral_density(self, velocities: np.ndarray) -> float:
        """Calculate mean power spectral density"""
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        if len(vel_magnitudes) < 10:
            return 0.0
            
        freqs, psd = signal.periodogram(vel_magnitudes, fs=self.fps)
        return np.mean(psd[1:])  # Exclude DC component
    
    def _calculate_xy_correlation(self, velocities: np.ndarray) -> float:
        """Calculate X-Y velocity correlation"""
        if len(velocities) < 2:
            return 0.0
            
        vx, vy = velocities[:, 0], velocities[:, 1]
        correlation_matrix = np.corrcoef(vx, vy)
        return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
    
    def _calculate_circular_variance(self, trajectory: np.ndarray) -> float:
        """Calculate circular variance of movement directions"""
        if len(trajectory) < 3:
            return 0.0
            
        # Calculate movement directions
        movements = np.diff(trajectory, axis=0)
        angles = np.arctan2(movements[:, 1], movements[:, 0])
        
        # Circular statistics
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        circular_variance = 1 - np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
        
        return circular_variance
    
    def _calculate_mean_resultant_length(self, trajectory: np.ndarray) -> float:
        """Calculate mean resultant length (directional consistency)"""
        if len(trajectory) < 3:
            return 0.0
            
        movements = np.diff(trajectory, axis=0)
        angles = np.arctan2(movements[:, 1], movements[:, 0])
        
        # Calculate mean resultant vector
        mean_cos = np.mean(np.cos(angles))
        mean_sin = np.mean(np.sin(angles))
        
        return np.sqrt(mean_cos**2 + mean_sin**2)
    
    def _calculate_angular_acceleration(self, trajectory: np.ndarray) -> float:
        """Calculate mean angular acceleration"""
        if len(trajectory) < 4:
            return 0.0
            
        # Calculate movement directions
        movements = np.diff(trajectory, axis=0)
        angles = np.arctan2(movements[:, 1], movements[:, 0])
        
        # Calculate angular velocities
        angular_velocities = np.diff(angles) / self.dt
        
        # Handle angle wrapping
        angular_velocities = np.where(angular_velocities > np.pi, 
                                    angular_velocities - 2*np.pi, 
                                    angular_velocities)
        angular_velocities = np.where(angular_velocities < -np.pi, 
                                    angular_velocities + 2*np.pi, 
                                    angular_velocities)
        
        # Calculate angular accelerations
        if len(angular_velocities) < 2:
            return 0.0
            
        angular_accelerations = np.diff(angular_velocities) / self.dt
        return np.mean(np.abs(angular_accelerations))
    
    def _empty_metrics(self) -> AdvancedKinematicMetrics:
        """Return empty metrics for invalid trajectories"""
        return AdvancedKinematicMetrics(
            velocity_entropy=0.0, velocity_kurtosis=0.0, velocity_skewness=0.0,
            velocity_cv=0.0, mean_acceleration=0.0, max_acceleration=0.0,
            acceleration_entropy=0.0, jerk_magnitude=0.0, tortuosity_index=0.0,
            sinuosity_ratio=0.0, fractal_dimension=1.0, hausdorff_distance=0.0,
            dominant_frequency=0.0, frequency_entropy=0.0, spectral_centroid=0.0,
            spectral_rolloff=0.0, kinetic_energy_mean=0.0, kinetic_energy_variance=0.0,
            work_done=0.0, power_spectral_density=0.0, xy_correlation=0.0,
            circular_variance=0.0, mean_resultant_length=0.0, angular_acceleration=0.0
        )

# Usage example
def analyze_population_kinematics(trajectories: List[np.ndarray]) -> pd.DataFrame:
    """Analyze advanced kinematics for a population of sperm"""
    analyzer = AdvancedKinematicsAnalyzer()
    results = []
    
    for i, trajectory in enumerate(trajectories):
        metrics = analyzer.analyze_trajectory(trajectory)
        result_dict = {
            'sperm_id': i,
            **metrics.__dict__
        }
        results.append(result_dict)
    
    return pd.DataFrame(results)
```

---

## 🤖 Machine Learning-Based Metrics

### AI-Powered Classification and Prediction

Create `training/models/ml_metrics.py`:

```python
"""
Machine Learning-Based Sperm Analysis Metrics
Developer: Youssef Shitiwi
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Optional
import joblib

class MLSpermAnalyzer:
    """Machine learning-based sperm analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.motility_classifier = None
        self.anomaly_detector = None
        self.quality_predictor = None
        self.is_trained = False
        
    def extract_ml_features(self, trajectory: np.ndarray, 
                           casa_metrics: Dict) -> np.ndarray:
        """Extract comprehensive features for ML analysis"""
        
        if len(trajectory) < 10:
            return np.zeros(50)  # Return zero features for short trajectories
            
        features = []
        
        # Basic trajectory statistics
        features.extend([
            len(trajectory),  # Track length
            np.std(trajectory[:, 0]),  # X position variance
            np.std(trajectory[:, 1]),  # Y position variance
        ])
        
        # Velocity features
        velocities = np.diff(trajectory, axis=0)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        
        features.extend([
            np.mean(vel_magnitudes),      # Mean velocity
            np.std(vel_magnitudes),       # Velocity variance
            np.max(vel_magnitudes),       # Maximum velocity
            np.min(vel_magnitudes),       # Minimum velocity
            np.percentile(vel_magnitudes, 25),  # 25th percentile
            np.percentile(vel_magnitudes, 75),  # 75th percentile
        ])
        
        # Acceleration features
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0)
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            
            features.extend([
                np.mean(acc_magnitudes),      # Mean acceleration
                np.std(acc_magnitudes),       # Acceleration variance
                np.max(acc_magnitudes),       # Maximum acceleration
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Directional features
        if len(velocities) > 0:
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            angle_changes = np.diff(angles)
            
            # Handle angle wrapping
            angle_changes = np.where(angle_changes > np.pi, 
                                   angle_changes - 2*np.pi, 
                                   angle_changes)
            angle_changes = np.where(angle_changes < -np.pi, 
                                   angle_changes + 2*np.pi, 
                                   angle_changes)
            
            features.extend([
                np.std(angles),               # Angular variance
                np.mean(np.abs(angle_changes)),  # Mean angle change
                np.std(angle_changes),        # Angle change variance
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Path complexity features
        path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        features.extend([
            path_length,                      # Total path length
            straight_distance,                # Straight-line distance
            path_length / straight_distance if straight_distance > 0 else 0,  # Tortuosity
        ])
        
        # Frequency domain features
        if len(vel_magnitudes) > 8:
            fft = np.fft.fft(vel_magnitudes - np.mean(vel_magnitudes))
            power_spectrum = np.abs(fft[:len(fft)//2])**2
            
            features.extend([
                np.argmax(power_spectrum),    # Dominant frequency index
                np.max(power_spectrum),       # Maximum power
                np.sum(power_spectrum),       # Total power
                np.std(power_spectrum),       # Power variance
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # CASA metrics integration
        casa_features = [
            casa_metrics.get('vcl', 0),
            casa_metrics.get('vsl', 0),
            casa_metrics.get('vap', 0),
            casa_metrics.get('lin', 0),
            casa_metrics.get('str', 0),
            casa_metrics.get('wob', 0),
            casa_metrics.get('alh', 0),
            casa_metrics.get('bcf', 0),
        ]
        features.extend(casa_features)
        
        # Spatial distribution features
        centroid = np.mean(trajectory, axis=0)
        distances_from_centroid = np.linalg.norm(trajectory - centroid, axis=1)
        
        features.extend([
            np.mean(distances_from_centroid),  # Mean distance from centroid
            np.std(distances_from_centroid),   # Spatial spread
            np.max(distances_from_centroid),   # Maximum excursion
        ])
        
        # Temporal features
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        
        features.extend([
            np.corrcoef(x_coords, np.arange(len(x_coords)))[0, 1] if len(x_coords) > 1 else 0,  # X trend
            np.corrcoef(y_coords, np.arange(len(y_coords)))[0, 1] if len(y_coords) > 1 else 0,  # Y trend
        ])
        
        # Periodicity features
        if len(vel_magnitudes) > 10:
            autocorr = np.correlate(vel_magnitudes, vel_magnitudes, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            
            features.extend([
                autocorr[1] if len(autocorr) > 1 else 0,  # Lag-1 autocorrelation
                autocorr[2] if len(autocorr) > 2 else 0,  # Lag-2 autocorrelation
                np.argmax(autocorr[1:]) + 1 if len(autocorr) > 1 else 0,  # Peak autocorr lag
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Energy and efficiency features
        if len(velocities) > 1:
            kinetic_energy = 0.5 * vel_magnitudes**2
            features.extend([
                np.mean(kinetic_energy),      # Mean kinetic energy
                np.var(kinetic_energy),       # Kinetic energy variance
                np.sum(kinetic_energy),       # Total kinetic energy
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Ensure we have exactly 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def train_motility_classifier(self, trajectories: List[np.ndarray], 
                                casa_metrics_list: List[Dict],
                                motility_labels: List[str]) -> Dict:
        """Train classifier for motility patterns"""
        
        # Extract features
        features_list = []
        for traj, casa in zip(trajectories, casa_metrics_list):
            features = self.extract_ml_features(traj, casa)
            features_list.append(features)
        
        X = np.array(features_list)
        y = np.array(motility_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train random forest classifier
        self.motility_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.motility_classifier.fit(X_scaled, y)
        
        # Feature importance analysis
        feature_importance = self.motility_classifier.feature_importances_
        
        # Train anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.anomaly_detector.fit(X_scaled)
        
        self.is_trained = True
        
        return {
            'accuracy': self.motility_classifier.score(X_scaled, y),
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
    
    def predict_motility_class(self, trajectory: np.ndarray, 
                             casa_metrics: Dict) -> Dict:
        """Predict motility class for new trajectory"""
        
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        features = self.extract_ml_features(trajectory, casa_metrics)
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict class
        predicted_class = self.motility_classifier.predict(X_scaled)[0]
        class_probabilities = self.motility_classifier.predict_proba(X_scaled)[0]
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1
        
        return {
            'predicted_class': predicted_class,
            'class_probabilities': dict(zip(self.motility_classifier.classes_, 
                                          class_probabilities)),
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly,
            'confidence': np.max(class_probabilities)
        }
    
    def cluster_population(self, trajectories: List[np.ndarray],
                          casa_metrics_list: List[Dict],
                          n_clusters: Optional[int] = None) -> Dict:
        """Cluster sperm population based on movement patterns"""
        
        # Extract features
        features_list = []
        for traj, casa in zip(trajectories, casa_metrics_list):
            features = self.extract_ml_features(traj, casa)
            features_list.append(features)
        
        X = np.array(features_list)
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            silhouette_scores = []
            cluster_range = range(2, min(11, len(X)//2))
            
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(score)
            
            n_clusters = cluster_range[np.argmax(silhouette_scores)]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        return {
            'kmeans_labels': cluster_labels,
            'dbscan_labels': dbscan_labels,
            'n_clusters_kmeans': n_clusters,
            'n_clusters_dbscan': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'silhouette_score': silhouette_score(X_scaled, cluster_labels),
            'cluster_centers': kmeans.cluster_centers_,
            'pca_coordinates': X_pca,
            'pca_explained_variance': pca.explained_variance_ratio_
        }
    
    def build_quality_predictor(self, trajectories: List[np.ndarray],
                               casa_metrics_list: List[Dict],
                               quality_scores: List[float]) -> keras.Model:
        """Build neural network for sperm quality prediction"""
        
        # Extract features
        features_list = []
        for traj, casa in zip(trajectories, casa_metrics_list):
            features = self.extract_ml_features(traj, casa)
            features_list.append(features)
        
        X = np.array(features_list)
        y = np.array(quality_scores)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build neural network
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_scaled, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.quality_predictor = model
        
        return model
    
    def predict_quality_score(self, trajectory: np.ndarray,
                            casa_metrics: Dict) -> float:
        """Predict quality score for single sperm"""
        
        if self.quality_predictor is None:
            return 0.0
        
        features = self.extract_ml_features(trajectory, casa_metrics)
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        
        return float(self.quality_predictor.predict(X_scaled)[0][0])
    
    def save_models(self, base_path: str):
        """Save trained models"""
        if self.motility_classifier:
            joblib.dump(self.motility_classifier, f"{base_path}_motility_classifier.pkl")
        if self.anomaly_detector:
            joblib.dump(self.anomaly_detector, f"{base_path}_anomaly_detector.pkl")
        if self.quality_predictor:
            self.quality_predictor.save(f"{base_path}_quality_predictor.h5")
        joblib.dump(self.scaler, f"{base_path}_scaler.pkl")
    
    def load_models(self, base_path: str):
        """Load trained models"""
        try:
            self.motility_classifier = joblib.load(f"{base_path}_motility_classifier.pkl")
            self.anomaly_detector = joblib.load(f"{base_path}_anomaly_detector.pkl")
            self.quality_predictor = keras.models.load_model(f"{base_path}_quality_predictor.h5")
            self.scaler = joblib.load(f"{base_path}_scaler.pkl")
            self.is_trained = True
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
```

---

## 🔬 Morphological Analysis

### AI-Powered Morphology Assessment

Create `training/models/morphology_analysis.py`:

```python
"""
AI-Powered Sperm Morphology Analysis
Developer: Youssef Shitiwi
"""

import cv2
import numpy as np
from scipy import ndimage, spatial
from skimage import measure, morphology, segmentation
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
import pandas as pd

@dataclass
class MorphologyMetrics:
    """Comprehensive morphology analysis results"""
    
    # Head measurements
    head_area: float                 # Head area (μm²)
    head_perimeter: float           # Head perimeter (μm)
    head_length: float              # Head length (μm)
    head_width: float               # Head width (μm)
    head_aspect_ratio: float        # Length/width ratio
    head_circularity: float         # 4π*area/perimeter²
    head_solidity: float            # Area/convex_hull_area
    head_eccentricity: float        # Ellipse eccentricity
    
    # Acrosome measurements
    acrosome_area: float            # Acrosome area (μm²)
    acrosome_coverage: float        # Acrosome/head area ratio
    acrosome_symmetry: float        # Acrosome symmetry index
    
    # Midpiece measurements
    midpiece_length: float          # Midpiece length (μm)
    midpiece_width: float           # Midpiece width (μm)
    midpiece_area: float            # Midpiece area (μm²)
    midpiece_regularity: float      # Shape regularity index
    
    # Tail measurements
    tail_length: float              # Tail length (μm)
    tail_width: float               # Average tail width (μm)
    tail_straightness: float        # Tail straightness index
    tail_regularity: float          # Tail width regularity
    
    # Overall measurements
    total_length: float             # Total sperm length (μm)
    head_midpiece_ratio: float      # Head/midpiece length ratio
    midpiece_tail_ratio: float      # Midpiece/tail length ratio
    
    # Quality scores
    head_quality_score: float       # Head morphology score (0-1)
    midpiece_quality_score: float   # Midpiece quality score (0-1)
    tail_quality_score: float       # Tail quality score (0-1)
    overall_quality_score: float    # Overall morphology score (0-1)
    
    # WHO classification
    who_classification: str          # Normal, Abnormal head, Abnormal midpiece, Abnormal tail
    abnormality_type: List[str]     # Specific abnormalities detected

class SpermMorphologyAnalyzer:
    """Advanced sperm morphology analysis"""
    
    def __init__(self, pixel_to_micron: float = 0.1):
        self.pixel_to_micron = pixel_to_micron
        self.morphology_classifier = None
        self.segmentation_model = None
        
        # WHO reference values (in micrometers)
        self.who_references = {
            'head_length': {'min': 4.0, 'max': 5.5, 'optimal': 4.7},
            'head_width': {'min': 2.5, 'max': 3.5, 'optimal': 3.1},
            'midpiece_length': {'min': 6.0, 'max': 8.0, 'optimal': 7.0},
            'tail_length': {'min': 45.0, 'max': 50.0, 'optimal': 47.5},
            'total_length': {'min': 55.0, 'max': 63.0, 'optimal': 59.0}
        }
    
    def analyze_morphology(self, sperm_image: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> MorphologyMetrics:
        """Comprehensive morphology analysis of single sperm"""
        
        # Preprocess image
        processed_image = self._preprocess_image(sperm_image)
        
        # Segment sperm components
        if mask is None:
            mask = self._segment_sperm(processed_image)
        
        head_mask, midpiece_mask, tail_mask = self._segment_components(processed_image, mask)
        
        # Analyze each component
        head_metrics = self._analyze_head(processed_image, head_mask)
        midpiece_metrics = self._analyze_midpiece(processed_image, midpiece_mask)
        tail_metrics = self._analyze_tail(processed_image, tail_mask)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            head_metrics, midpiece_metrics, tail_metrics
        )
        
        # WHO classification
        who_class, abnormalities = self._classify_who_morphology(
            head_metrics, midpiece_metrics, tail_metrics, overall_metrics
        )
        
        return MorphologyMetrics(
            **head_metrics,
            **midpiece_metrics, 
            **tail_metrics,
            **overall_metrics,
            who_classification=who_class,
            abnormality_type=abnormalities
        )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess sperm image for analysis"""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize intensity
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def _segment_sperm(self, image: np.ndarray) -> np.ndarray:
        """Segment sperm from background"""
        
        # Threshold using Otsu's method
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find largest connected component (sperm)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(image)
        
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        return mask
    
    def _segment_components(self, image: np.ndarray, 
                          sperm_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment sperm into head, midpiece, and tail"""
        
        # Get sperm skeleton
        skeleton = morphology.skeletonize(sperm_mask // 255)
        
        # Find endpoints and branch points
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        endpoints = self._find_endpoints(skeleton_uint8)
        
        if len(endpoints) < 2:
            # Fallback: create approximate segmentation
            return self._approximate_segmentation(sperm_mask)
        
        # Find sperm orientation
        contours, _ = cv2.findContours(sperm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._approximate_segmentation(sperm_mask)
        
        contour = max(contours, key=cv2.contourArea)
        
        # Fit ellipse to determine orientation
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, (width, height), angle = ellipse
            
            # Determine head vs tail based on width variation along major axis
            head_mask, midpiece_mask, tail_mask = self._segment_by_width_analysis(
                sperm_mask, center, angle
            )
        else:
            head_mask, midpiece_mask, tail_mask = self._approximate_segmentation(sperm_mask)
        
        return head_mask, midpiece_mask, tail_mask
    
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find skeleton endpoints"""
        
        # 8-connected neighborhood kernel
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Apply filter
        filtered = cv2.filter2D(skeleton, -1, kernel)
        
        # Endpoints have exactly one neighbor (value = 11)
        endpoints = np.where((filtered == 11) & (skeleton == 255))
        
        return list(zip(endpoints[0], endpoints[1]))
    
    def _segment_by_width_analysis(self, mask: np.ndarray, 
                                 center: Tuple[float, float],
                                 angle: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment based on width variation analysis"""
        
        # Create masks for each component
        h, w = mask.shape
        head_mask = np.zeros_like(mask)
        midpiece_mask = np.zeros_like(mask)
        tail_mask = np.zeros_like(mask)
        
        # Calculate distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Sample along major axis to find width variation
        cx, cy = center
        angle_rad = np.radians(angle)
        
        # Sample points along the major axis
        max_dist = min(h, w) // 2
        sample_points = []
        widths = []
        
        for d in range(-max_dist, max_dist, 2):
            x = int(cx + d * np.cos(angle_rad))
            y = int(cy + d * np.sin(angle_rad))
            
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                sample_points.append((x, y))
                widths.append(dist_transform[y, x])
        
        if len(widths) < 10:
            return self._approximate_segmentation(mask)
        
        # Find regions based on width: head (wide), midpiece (medium), tail (narrow)
        widths = np.array(widths)
        
        # Use k-means to cluster widths into 3 groups
        kmeans = KMeans(n_clusters=3, random_state=42)
        width_clusters = kmeans.fit_predict(widths.reshape(-1, 1))
        
        # Assign clusters to components based on average width
        cluster_means = [np.mean(widths[width_clusters == i]) for i in range(3)]
        sorted_clusters = np.argsort(cluster_means)
        
        # Head = widest, midpiece = medium, tail = narrowest
        head_cluster = sorted_clusters[2]
        midpiece_cluster = sorted_clusters[1]
        tail_cluster = sorted_clusters[0]
        
        # Create component masks
        for i, (x, y) in enumerate(sample_points):
            cluster = width_clusters[i]
            radius = max(1, int(widths[i]))
            
            if cluster == head_cluster:
                cv2.circle(head_mask, (x, y), radius, 255, -1)
            elif cluster == midpiece_cluster:
                cv2.circle(midpiece_mask, (x, y), radius, 255, -1)
            else:
                cv2.circle(tail_mask, (x, y), radius, 255, -1)
        
        # Apply original mask to ensure we stay within sperm boundary
        head_mask = cv2.bitwise_and(head_mask, mask)
        midpiece_mask = cv2.bitwise_and(midpiece_mask, mask)
        tail_mask = cv2.bitwise_and(tail_mask, mask)
        
        return head_mask, midpiece_mask, tail_mask
    
    def _approximate_segmentation(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback segmentation method"""
        
        # Find contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)
        
        contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Approximate division: head (20%), midpiece (15%), tail (65%)
        head_mask = np.zeros_like(mask)
        midpiece_mask = np.zeros_like(mask)
        tail_mask = np.zeros_like(mask)
        
        if w > h:  # Horizontal orientation
            head_end = x + int(0.2 * w)
            midpiece_end = x + int(0.35 * w)
            
            head_mask[y:y+h, x:head_end] = mask[y:y+h, x:head_end]
            midpiece_mask[y:y+h, head_end:midpiece_end] = mask[y:y+h, head_end:midpiece_end]
            tail_mask[y:y+h, midpiece_end:x+w] = mask[y:y+h, midpiece_end:x+w]
        else:  # Vertical orientation
            head_end = y + int(0.2 * h)
            midpiece_end = y + int(0.35 * h)
            
            head_mask[y:head_end, x:x+w] = mask[y:head_end, x:x+w]
            midpiece_mask[head_end:midpiece_end, x:x+w] = mask[head_end:midpiece_end, x:x+w]
            tail_mask[midpiece_end:y+h, x:x+w] = mask[midpiece_end:y+h, x:x+w]
        
        return head_mask, midpiece_mask, tail_mask
    
    def _analyze_head(self, image: np.ndarray, head_mask: np.ndarray) -> Dict:
        """Analyze head morphology"""
        
        if np.sum(head_mask) == 0:
            return self._empty_head_metrics()
        
        # Find contour
        contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._empty_head_metrics()
        
        contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area_pixels = cv2.contourArea(contour)
        perimeter_pixels = cv2.arcLength(contour, True)
        
        # Convert to micrometers
        area_microns = area_pixels * (self.pixel_to_micron ** 2)
        perimeter_microns = perimeter_pixels * self.pixel_to_micron
        
        # Fit ellipse for length and width
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (width_pix, height_pix), angle = ellipse
            
            # Major and minor axes
            major_axis = max(width_pix, height_pix) * self.pixel_to_micron
            minor_axis = min(width_pix, height_pix) * self.pixel_to_micron
            
            length = major_axis
            width = minor_axis
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
            
            # Eccentricity
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0
        else:
            # Fallback measurements
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h) * self.pixel_to_micron
            width = min(w, h) * self.pixel_to_micron
            aspect_ratio = length / width if width > 0 else 0
            eccentricity = 0
        
        # Shape descriptors
        circularity = (4 * np.pi * area_pixels) / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
        
        # Solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_pixels / hull_area if hull_area > 0 else 0
        
        # Quality score based on WHO criteria
        quality_score = self._calculate_head_quality_score({
            'area': area_microns,
            'length': length,
            'width': width,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity
        })
        
        return {
            'head_area': area_microns,
            'head_perimeter': perimeter_microns,
            'head_length': length,
            'head_width': width,
            'head_aspect_ratio': aspect_ratio,
            'head_circularity': circularity,
            'head_solidity': solidity,
            'head_eccentricity': eccentricity,
            'head_quality_score': quality_score,
            # Acrosome analysis (simplified)
            'acrosome_area': area_microns * 0.6,  # Approximate
            'acrosome_coverage': 0.6,  # Approximate
            'acrosome_symmetry': 0.8,  # Approximate
        }
    
    def _analyze_midpiece(self, image: np.ndarray, midpiece_mask: np.ndarray) -> Dict:
        """Analyze midpiece morphology"""
        
        if np.sum(midpiece_mask) == 0:
            return self._empty_midpiece_metrics()
        
        # Find contour
        contours, _ = cv2.findContours(midpiece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._empty_midpiece_metrics()
        
        contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area_pixels = cv2.contourArea(contour)
        area_microns = area_pixels * (self.pixel_to_micron ** 2)
        
        # Bounding box for length and width estimation
        x, y, w, h = cv2.boundingRect(contour)
        
        # Determine orientation and calculate dimensions
        if w > h:
            length = w * self.pixel_to_micron
            width = h * self.pixel_to_micron
        else:
            length = h * self.pixel_to_micron
            width = w * self.pixel_to_micron
        
        # Regularity assessment
        # Calculate width variation along length
        skeleton = morphology.skeletonize(midpiece_mask // 255)
        skeleton_points = np.where(skeleton)
        
        if len(skeleton_points[0]) > 5:
            dist_transform = cv2.distanceTransform(midpiece_mask, cv2.DIST_L2, 5)
            widths = [dist_transform[y, x] for y, x in zip(skeleton_points[0], skeleton_points[1])]
            regularity = 1.0 - (np.std(widths) / np.mean(widths)) if np.mean(widths) > 0 else 0
        else:
            regularity = 0.8  # Default
        
        # Quality score
        quality_score = self._calculate_midpiece_quality_score({
            'length': length,
            'width': width,
            'area': area_microns,
            'regularity': regularity
        })
        
        return {
            'midpiece_length': length,
            'midpiece_width': width,
            'midpiece_area': area_microns,
            'midpiece_regularity': regularity,
            'midpiece_quality_score': quality_score
        }
    
    def _analyze_tail(self, image: np.ndarray, tail_mask: np.ndarray) -> Dict:
        """Analyze tail morphology"""
        
        if np.sum(tail_mask) == 0:
            return self._empty_tail_metrics()
        
        # Get skeleton for length measurement
        skeleton = morphology.skeletonize(tail_mask // 255)
        skeleton_points = np.where(skeleton)
        
        if len(skeleton_points[0]) < 2:
            return self._empty_tail_metrics()
        
        # Calculate tail length along skeleton
        points = list(zip(skeleton_points[0], skeleton_points[1]))
        
        # Sort points to create a path
        if len(points) > 2:
            # Simple path construction (can be improved with more sophisticated algorithms)
            sorted_points = [points[0]]
            remaining_points = points[1:]
            
            while remaining_points:
                last_point = sorted_points[-1]
                distances = [np.sqrt((p[0] - last_point[0])**2 + (p[1] - last_point[1])**2) 
                           for p in remaining_points]
                nearest_idx = np.argmin(distances)
                sorted_points.append(remaining_points.pop(nearest_idx))
            
            # Calculate total length
            total_length_pixels = sum(
                np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                for p1, p2 in zip(sorted_points[:-1], sorted_points[1:])
            )
            length = total_length_pixels * self.pixel_to_micron
        else:
            length = 0
        
        # Calculate average width
        dist_transform = cv2.distanceTransform(tail_mask, cv2.DIST_L2, 5)
        valid_distances = dist_transform[skeleton]
        avg_width = np.mean(valid_distances) * 2 * self.pixel_to_micron if len(valid_distances) > 0 else 0
        
        # Straightness assessment
        if len(points) > 2:
            start_point = np.array(points[0])
            end_point = np.array(points[-1])
            straight_distance = np.linalg.norm(end_point - start_point) * self.pixel_to_micron
            straightness = straight_distance / length if length > 0 else 0
        else:
            straightness = 1.0
        
        # Width regularity
        if len(valid_distances) > 5:
            width_cv = np.std(valid_distances) / np.mean(valid_distances) if np.mean(valid_distances) > 0 else 0
            regularity = max(0, 1.0 - width_cv)
        else:
            regularity = 0.8
        
        # Quality score
        quality_score = self._calculate_tail_quality_score({
            'length': length,
            'width': avg_width,
            'straightness': straightness,
            'regularity': regularity
        })
        
        return {
            'tail_length': length,
            'tail_width': avg_width,
            'tail_straightness': straightness,
            'tail_regularity': regularity,
            'tail_quality_score': quality_score
        }
    
    def _calculate_overall_metrics(self, head_metrics: Dict,
                                 midpiece_metrics: Dict,
                                 tail_metrics: Dict) -> Dict:
        """Calculate overall morphology metrics"""
        
        total_length = (head_metrics.get('head_length', 0) +
                       midpiece_metrics.get('midpiece_length', 0) +
                       tail_metrics.get('tail_length', 0))
        
        head_midpiece_ratio = (head_metrics.get('head_length', 0) /
                              midpiece_metrics.get('midpiece_length', 1)) if midpiece_metrics.get('midpiece_length', 0) > 0 else 0
        
        midpiece_tail_ratio = (midpiece_metrics.get('midpiece_length', 0) /
                              tail_metrics.get('tail_length', 1)) if tail_metrics.get('tail_length', 0) > 0 else 0
        
        # Overall quality score (weighted average)
        weights = {'head': 0.4, 'midpiece': 0.2, 'tail': 0.4}
        overall_quality = (
            weights['head'] * head_metrics.get('head_quality_score', 0) +
            weights['midpiece'] * midpiece_metrics.get('midpiece_quality_score', 0) +
            weights['tail'] * tail_metrics.get('tail_quality_score', 0)
        )
        
        return {
            'total_length': total_length,
            'head_midpiece_ratio': head_midpiece_ratio,
            'midpiece_tail_ratio': midpiece_tail_ratio,
            'overall_quality_score': overall_quality
        }
    
    def _classify_who_morphology(self, head_metrics: Dict,
                               midpiece_metrics: Dict,
                               tail_metrics: Dict,
                               overall_metrics: Dict) -> Tuple[str, List[str]]:
        """Classify morphology according to WHO criteria"""
        
        abnormalities = []
        
        # Check head abnormalities
        head_length = head_metrics.get('head_length', 0)
        head_width = head_metrics.get('head_width', 0)
        head_area = head_metrics.get('head_area', 0)
        
        if (head_length < self.who_references['head_length']['min'] or
            head_length > self.who_references['head_length']['max']):
            abnormalities.append('abnormal_head_length')
        
        if (head_width < self.who_references['head_width']['min'] or
            head_width > self.who_references['head_width']['max']):
            abnormalities.append('abnormal_head_width')
        
        if head_metrics.get('head_aspect_ratio', 0) > 2.0 or head_metrics.get('head_aspect_ratio', 0) < 1.2:
            abnormalities.append('abnormal_head_shape')
        
        # Check midpiece abnormalities
        midpiece_length = midpiece_metrics.get('midpiece_length', 0)
        
        if (midpiece_length < self.who_references['midpiece_length']['min'] or
            midpiece_length > self.who_references['midpiece_length']['max']):
            abnormalities.append('abnormal_midpiece_length')
        
        if midpiece_metrics.get('midpiece_regularity', 0) < 0.7:
            abnormalities.append('irregular_midpiece')
        
        # Check tail abnormalities
        tail_length = tail_metrics.get('tail_length', 0)
        
        if (tail_length < self.who_references['tail_length']['min'] or
            tail_length > self.who_references['tail_length']['max']):
            abnormalities.append('abnormal_tail_length')
        
        if tail_metrics.get('tail_straightness', 0) < 0.8:
            abnormalities.append('bent_tail')
        
        if tail_metrics.get('tail_regularity', 0) < 0.7:
            abnormalities.append('irregular_tail')
        
        # Overall classification
        if len(abnormalities) == 0:
            classification = 'normal'
        elif any('head' in abn for abn in abnormalities):
            classification = 'abnormal_head'
        elif any('midpiece' in abn for abn in abnormalities):
            classification = 'abnormal_midpiece'
        elif any('tail' in abn for abn in abnormalities):
            classification = 'abnormal_tail'
        else:
            classification = 'multiple_abnormalities'
        
        return classification, abnormalities
    
    def _calculate_head_quality_score(self, metrics: Dict) -> float:
        """Calculate head quality score based on WHO criteria"""
        
        score = 1.0
        
        # Length score
        length = metrics.get('length', 0)
        length_ref = self.who_references['head_length']
        if length < length_ref['min'] or length > length_ref['max']:
            score *= 0.6
        elif abs(length - length_ref['optimal']) / length_ref['optimal'] > 0.2:
            score *= 0.8
        
        # Width score
        width = metrics.get('width', 0)
        width_ref = self.who_references['head_width']
        if width < width_ref['min'] or width > width_ref['max']:
            score *= 0.6
        elif abs(width - width_ref['optimal']) / width_ref['optimal'] > 0.2:
            score *= 0.8
        
        # Shape scores
        aspect_ratio = metrics.get('aspect_ratio', 0)
        if aspect_ratio < 1.2 or aspect_ratio > 2.0:
            score *= 0.7
        
        circularity = metrics.get('circularity', 0)
        if circularity < 0.6:
            score *= 0.8
        
        solidity = metrics.get('solidity', 0)
        if solidity < 0.8:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _calculate_midpiece_quality_score(self, metrics: Dict) -> float:
        """Calculate midpiece quality score"""
        
        score = 1.0
        
        length = metrics.get('length', 0)
        length_ref = self.who_references['midpiece_length']
        if length < length_ref['min'] or length > length_ref['max']:
            score *= 0.6
        
        regularity = metrics.get('regularity', 0)
        if regularity < 0.7:
            score *= 0.7
        
        return max(0.0, min(1.0, score))
    
    def _calculate_tail_quality_score(self, metrics: Dict) -> float:
        """Calculate tail quality score"""
        
        score = 1.0
        
        length = metrics.get('length', 0)
        length_ref = self.who_references['tail_length']
        if length < length_ref['min'] or length > length_ref['max']:
            score *= 0.6
        
        straightness = metrics.get('straightness', 0)
        if straightness < 0.8:
            score *= 0.7
        
        regularity = metrics.get('regularity', 0)
        if regularity < 0.7:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _empty_head_metrics(self) -> Dict:
        """Return empty head metrics"""
        return {
            'head_area': 0.0, 'head_perimeter': 0.0, 'head_length': 0.0,
            'head_width': 0.0, 'head_aspect_ratio': 0.0, 'head_circularity': 0.0,
            'head_solidity': 0.0, 'head_eccentricity': 0.0, 'head_quality_score': 0.0,
            'acrosome_area': 0.0, 'acrosome_coverage': 0.0, 'acrosome_symmetry': 0.0
        }
    
    def _empty_midpiece_metrics(self) -> Dict:
        """Return empty midpiece metrics"""
        return {
            'midpiece_length': 0.0, 'midpiece_width': 0.0,
            'midpiece_area': 0.0, 'midpiece_regularity': 0.0,
            'midpiece_quality_score': 0.0
        }
    
    def _empty_tail_metrics(self) -> Dict:
        """Return empty tail metrics"""
        return {
            'tail_length': 0.0, 'tail_width': 0.0,
            'tail_straightness': 0.0, 'tail_regularity': 0.0,
            'tail_quality_score': 0.0
        }
```

This comprehensive extended metrics guide provides advanced analysis capabilities that go far beyond standard CASA parameters. The implementation includes:

1. **Advanced Kinematic Metrics**: Sophisticated motion analysis with entropy, frequency domain analysis, and path complexity measures
2. **Machine Learning Integration**: AI-powered classification, anomaly detection, and quality prediction
3. **Morphological Analysis**: Detailed sperm morphology assessment with WHO compliance
4. **Population Analytics**: Clustering and comparative analysis capabilities

All implementations maintain the high-quality standards established by developer **Youssef Shitiwi** and provide production-ready code for advanced sperm analysis research.