#!/usr/bin/env python3
"""
CASA (Computer-Assisted Sperm Analysis) Metrics Calculator
Author: Youssef Shitiwi
Description: Calculate sperm motility and kinematic parameters
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math
import json
from pathlib import Path

@dataclass
class SpermParameters:
    """Sperm kinematic parameters following WHO guidelines."""
    # Basic tracking info
    track_id: int
    duration_frames: int
    duration_seconds: float
    
    # Motion classification
    is_motile: bool
    is_progressive: bool
    is_slow_progressive: bool
    is_non_progressive: bool
    is_immotile: bool
    
    # Velocity parameters (μm/s)
    vcl: float  # Curvilinear velocity
    vsl: float  # Straight-line velocity
    vap: float  # Average path velocity
    
    # Motion parameters
    lin: float  # Linearity (VSL/VCL)
    str: float  # Straightness (VSL/VAP)
    wob: float  # Wobble (VAP/VCL)
    
    # Path parameters
    alh: float  # Amplitude of lateral head displacement (μm)
    bcf: float  # Beat cross frequency (Hz)
    
    # Distance parameters (μm)
    total_distance: float
    net_distance: float
    
    # Coordinates and trajectory
    trajectory_x: List[float]
    trajectory_y: List[float]
    smoothed_trajectory_x: List[float]
    smoothed_trajectory_y: List[float]

class CASAMetrics:
    """
    CASA metrics calculator for sperm analysis.
    
    Calculates WHO standard parameters for sperm motility analysis.
    """
    
    def __init__(self, 
                 fps: float = 30.0,
                 pixel_to_micron: float = 1.0,
                 min_track_length: int = 10,
                 vsl_threshold: float = 25.0,
                 vcl_threshold: float = 10.0):
        """
        Initialize CASA metrics calculator.
        
        Args:
            fps: Video frame rate
            pixel_to_micron: Conversion factor from pixels to micrometers
            min_track_length: Minimum track length for analysis
            vsl_threshold: VSL threshold for progressive motility (μm/s)
            vcl_threshold: VCL threshold for motility classification (μm/s)
        """
        self.fps = fps
        self.pixel_to_micron = pixel_to_micron
        self.min_track_length = min_track_length
        self.vsl_threshold = vsl_threshold
        self.vcl_threshold = vcl_threshold
        
        # WHO 2010 guidelines thresholds
        self.progressive_threshold = 25.0  # μm/s
        self.slow_progressive_threshold = 5.0  # μm/s
        
    def calculate_velocity(self, trajectory: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        """
        Calculate VCL, VSL, and VAP from trajectory.
        
        Args:
            trajectory: List of (x, y) coordinates in pixels
            
        Returns:
            Tuple of (VCL, VSL, VAP) in μm/s
        """
        if len(trajectory) < 2:
            return 0.0, 0.0, 0.0
        
        # Convert to micrometers
        trajectory_um = [(x * self.pixel_to_micron, y * self.pixel_to_micron) 
                        for x, y in trajectory]
        
        # Calculate VCL (Curvilinear Velocity)
        total_distance = 0.0
        for i in range(1, len(trajectory_um)):
            dist = euclidean(trajectory_um[i-1], trajectory_um[i])
            total_distance += dist
        
        duration = (len(trajectory) - 1) / self.fps
        vcl = total_distance / duration if duration > 0 else 0.0
        
        # Calculate VSL (Straight-line Velocity)
        if len(trajectory_um) >= 2:
            net_distance = euclidean(trajectory_um[0], trajectory_um[-1])
            vsl = net_distance / duration if duration > 0 else 0.0
        else:
            vsl = 0.0
        
        # Calculate VAP (Average Path Velocity)
        if len(trajectory_um) >= 4:
            # Smooth trajectory for VAP calculation
            smoothed_trajectory = self.smooth_trajectory(trajectory_um)
            
            # Calculate distance along smoothed path
            smoothed_distance = 0.0
            for i in range(1, len(smoothed_trajectory)):
                dist = euclidean(smoothed_trajectory[i-1], smoothed_trajectory[i])
                smoothed_distance += dist
            
            vap = smoothed_distance / duration if duration > 0 else 0.0
        else:
            vap = vcl  # Fallback to VCL if trajectory too short
        
        return vcl, vsl, vap
    
    def smooth_trajectory(self, trajectory: List[Tuple[float, float]], 
                         window_length: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Smooth trajectory using Savitzky-Golay filter.
        
        Args:
            trajectory: Original trajectory points
            window_length: Window length for smoothing
            
        Returns:
            Smoothed trajectory
        """
        if len(trajectory) < 4:
            return trajectory
        
        if window_length is None:
            window_length = min(5, len(trajectory) // 2 * 2 + 1)  # Ensure odd number
        
        window_length = max(3, min(window_length, len(trajectory)))
        if window_length % 2 == 0:
            window_length -= 1  # Ensure odd
        
        x_coords = [point[0] for point in trajectory]
        y_coords = [point[1] for point in trajectory]
        
        try:
            if len(x_coords) >= window_length:
                x_smooth = savgol_filter(x_coords, window_length, 2)
                y_smooth = savgol_filter(y_coords, window_length, 2)
            else:
                x_smooth = x_coords
                y_smooth = y_coords
        except:
            x_smooth = x_coords
            y_smooth = y_coords
        
        return list(zip(x_smooth, y_smooth))
    
    def calculate_linearity_parameters(self, vcl: float, vsl: float, vap: float) -> Tuple[float, float, float]:
        """
        Calculate LIN, STR, and WOB parameters.
        
        Args:
            vcl: Curvilinear velocity
            vsl: Straight-line velocity
            vap: Average path velocity
            
        Returns:
            Tuple of (LIN, STR, WOB)
        """
        # Linearity (LIN) = VSL/VCL
        lin = (vsl / vcl) * 100 if vcl > 0 else 0.0
        
        # Straightness (STR) = VSL/VAP
        str_val = (vsl / vap) * 100 if vap > 0 else 0.0
        
        # Wobble (WOB) = VAP/VCL
        wob = (vap / vcl) * 100 if vcl > 0 else 0.0
        
        # Ensure values are between 0 and 100
        lin = max(0, min(100, lin))
        str_val = max(0, min(100, str_val))
        wob = max(0, min(100, wob))
        
        return lin, str_val, wob
    
    def calculate_alh_bcf(self, trajectory: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate ALH (Amplitude of Lateral Head displacement) and BCF (Beat Cross Frequency).
        
        Args:
            trajectory: List of (x, y) coordinates
            
        Returns:
            Tuple of (ALH, BCF)
        """
        if len(trajectory) < 4:
            return 0.0, 0.0
        
        # Convert to micrometers
        trajectory_um = [(x * self.pixel_to_micron, y * self.pixel_to_micron) 
                        for x, y in trajectory]
        
        # Calculate smooth average path
        smoothed_trajectory = self.smooth_trajectory(trajectory_um)
        
        # Calculate lateral displacements
        lateral_displacements = []
        
        for i, point in enumerate(trajectory_um):
            if i < len(smoothed_trajectory):
                # Find closest point on smoothed path
                smooth_point = smoothed_trajectory[i]
                displacement = euclidean(point, smooth_point)
                lateral_displacements.append(displacement)
        
        # ALH is the mean of lateral displacements
        alh = np.mean(lateral_displacements) if lateral_displacements else 0.0
        
        # BCF calculation (simplified approach)
        # Count sign changes in lateral displacement derivative
        if len(lateral_displacements) > 2:
            displacement_diff = np.diff(lateral_displacements)
            sign_changes = 0
            for i in range(1, len(displacement_diff)):
                if displacement_diff[i-1] * displacement_diff[i] < 0:
                    sign_changes += 1
            
            duration = (len(trajectory) - 1) / self.fps
            bcf = (sign_changes / 2) / duration if duration > 0 else 0.0
        else:
            bcf = 0.0
        
        return alh, bcf
    
    def classify_motility(self, vcl: float, vsl: float, lin: float) -> Tuple[bool, bool, bool, bool, bool]:
        """
        Classify sperm motility according to WHO guidelines.
        
        Args:
            vcl: Curvilinear velocity
            vsl: Straight-line velocity
            lin: Linearity
            
        Returns:
            Tuple of (motile, progressive, slow_progressive, non_progressive, immotile)
        """
        # Basic motility threshold
        is_motile = vcl >= self.vcl_threshold
        
        if not is_motile:
            return False, False, False, False, True
        
        # Progressive motility (WHO: VSL ≥ 25 μm/s and LIN ≥ 50%)
        is_progressive = (vsl >= self.progressive_threshold and lin >= 50)
        
        # Slow progressive (WHO: VSL ≥ 5 μm/s but < 25 μm/s, or LIN < 50%)
        is_slow_progressive = (
            (self.slow_progressive_threshold <= vsl < self.progressive_threshold) or
            (vsl >= self.progressive_threshold and lin < 50)
        )
        
        # Non-progressive (motile but not progressive or slow progressive)
        is_non_progressive = is_motile and not is_progressive and not is_slow_progressive
        
        is_immotile = not is_motile
        
        return is_motile, is_progressive, is_slow_progressive, is_non_progressive, is_immotile
    
    def analyze_sperm_track(self, track_id: int, 
                           trajectory: List[Tuple[float, float]]) -> Optional[SpermParameters]:
        """
        Analyze a single sperm track and calculate all CASA parameters.
        
        Args:
            track_id: Unique track identifier
            trajectory: List of (x, y) coordinates
            
        Returns:
            SpermParameters object or None if track too short
        """
        if len(trajectory) < self.min_track_length:
            return None
        
        # Basic parameters
        duration_frames = len(trajectory)
        duration_seconds = duration_frames / self.fps
        
        # Calculate velocities
        vcl, vsl, vap = self.calculate_velocity(trajectory)
        
        # Calculate linearity parameters
        lin, str_val, wob = self.calculate_linearity_parameters(vcl, vsl, vap)
        
        # Calculate ALH and BCF
        alh, bcf = self.calculate_alh_bcf(trajectory)
        
        # Calculate distances
        trajectory_um = [(x * self.pixel_to_micron, y * self.pixel_to_micron) 
                        for x, y in trajectory]
        
        total_distance = 0.0
        for i in range(1, len(trajectory_um)):
            total_distance += euclidean(trajectory_um[i-1], trajectory_um[i])
        
        net_distance = euclidean(trajectory_um[0], trajectory_um[-1]) if len(trajectory_um) >= 2 else 0.0
        
        # Classify motility
        is_motile, is_progressive, is_slow_progressive, is_non_progressive, is_immotile = \
            self.classify_motility(vcl, vsl, lin)
        
        # Smooth trajectory for output
        smoothed_trajectory = self.smooth_trajectory(trajectory_um)
        
        # Create parameters object
        parameters = SpermParameters(
            track_id=track_id,
            duration_frames=duration_frames,
            duration_seconds=duration_seconds,
            is_motile=is_motile,
            is_progressive=is_progressive,
            is_slow_progressive=is_slow_progressive,
            is_non_progressive=is_non_progressive,
            is_immotile=is_immotile,
            vcl=vcl,
            vsl=vsl,
            vap=vap,
            lin=lin,
            str=str_val,
            wob=wob,
            alh=alh,
            bcf=bcf,
            total_distance=total_distance,
            net_distance=net_distance,
            trajectory_x=[x for x, y in trajectory],
            trajectory_y=[y for x, y in trajectory],
            smoothed_trajectory_x=[x for x, y in smoothed_trajectory],
            smoothed_trajectory_y=[y for x, y in smoothed_trajectory]
        )
        
        return parameters
    
    def analyze_multiple_tracks(self, tracks_data: Dict[int, List[Tuple[float, float]]]) -> List[SpermParameters]:
        """
        Analyze multiple sperm tracks.
        
        Args:
            tracks_data: Dictionary mapping track_id to trajectory
            
        Returns:
            List of SpermParameters objects
        """
        results = []
        
        for track_id, trajectory in tracks_data.items():
            parameters = self.analyze_sperm_track(track_id, trajectory)
            if parameters is not None:
                results.append(parameters)
        
        return results
    
    def calculate_population_statistics(self, sperm_list: List[SpermParameters]) -> Dict:
        """
        Calculate population-level statistics.
        
        Args:
            sperm_list: List of SpermParameters
            
        Returns:
            Dictionary with population statistics
        """
        if not sperm_list:
            return {}
        
        total_count = len(sperm_list)
        
        # Count motility categories
        motile_count = sum(1 for s in sperm_list if s.is_motile)
        progressive_count = sum(1 for s in sperm_list if s.is_progressive)
        slow_progressive_count = sum(1 for s in sperm_list if s.is_slow_progressive)
        non_progressive_count = sum(1 for s in sperm_list if s.is_non_progressive)
        immotile_count = sum(1 for s in sperm_list if s.is_immotile)
        
        # Calculate percentages
        motility_percentage = (motile_count / total_count) * 100
        progressive_percentage = (progressive_count / total_count) * 100
        slow_progressive_percentage = (slow_progressive_count / total_count) * 100
        non_progressive_percentage = (non_progressive_count / total_count) * 100
        immotile_percentage = (immotile_count / total_count) * 100
        
        # Calculate mean values for motile sperm
        motile_sperm = [s for s in sperm_list if s.is_motile]
        
        if motile_sperm:
            mean_vcl = np.mean([s.vcl for s in motile_sperm])
            mean_vsl = np.mean([s.vsl for s in motile_sperm])
            mean_vap = np.mean([s.vap for s in motile_sperm])
            mean_lin = np.mean([s.lin for s in motile_sperm])
            mean_str = np.mean([s.str for s in motile_sperm])
            mean_wob = np.mean([s.wob for s in motile_sperm])
            mean_alh = np.mean([s.alh for s in motile_sperm])
            mean_bcf = np.mean([s.bcf for s in motile_sperm])
            
            std_vcl = np.std([s.vcl for s in motile_sperm])
            std_vsl = np.std([s.vsl for s in motile_sperm])
            std_vap = np.std([s.vap for s in motile_sperm])
            std_lin = np.std([s.lin for s in motile_sperm])
        else:
            mean_vcl = mean_vsl = mean_vap = mean_lin = mean_str = mean_wob = mean_alh = mean_bcf = 0.0
            std_vcl = std_vsl = std_vap = std_lin = 0.0
        
        statistics = {
            'total_sperm_count': total_count,
            'motile_count': motile_count,
            'progressive_count': progressive_count,
            'slow_progressive_count': slow_progressive_count,
            'non_progressive_count': non_progressive_count,
            'immotile_count': immotile_count,
            'motility_percentage': motility_percentage,
            'progressive_percentage': progressive_percentage,
            'slow_progressive_percentage': slow_progressive_percentage,
            'non_progressive_percentage': non_progressive_percentage,
            'immotile_percentage': immotile_percentage,
            'mean_vcl': mean_vcl,
            'mean_vsl': mean_vsl,
            'mean_vap': mean_vap,
            'mean_lin': mean_lin,
            'mean_str': mean_str,
            'mean_wob': mean_wob,
            'mean_alh': mean_alh,
            'mean_bcf': mean_bcf,
            'std_vcl': std_vcl,
            'std_vsl': std_vsl,
            'std_vap': std_vap,
            'std_lin': std_lin
        }
        
        return statistics
    
    def export_results_to_csv(self, sperm_list: List[SpermParameters], 
                             output_path: str, include_trajectories: bool = False):
        """
        Export results to CSV file.
        
        Args:
            sperm_list: List of SpermParameters
            output_path: Output CSV file path
            include_trajectories: Whether to include trajectory data
        """
        if not sperm_list:
            return
        
        # Convert to DataFrame
        data = []
        for sperm in sperm_list:
            row = {
                'track_id': sperm.track_id,
                'duration_frames': sperm.duration_frames,
                'duration_seconds': sperm.duration_seconds,
                'motile': sperm.is_motile,
                'progressive': sperm.is_progressive,
                'slow_progressive': sperm.is_slow_progressive,
                'non_progressive': sperm.is_non_progressive,
                'immotile': sperm.is_immotile,
                'vcl_um_s': sperm.vcl,
                'vsl_um_s': sperm.vsl,
                'vap_um_s': sperm.vap,
                'lin_percent': sperm.lin,
                'str_percent': sperm.str,
                'wob_percent': sperm.wob,
                'alh_um': sperm.alh,
                'bcf_hz': sperm.bcf,
                'total_distance_um': sperm.total_distance,
                'net_distance_um': sperm.net_distance
            }
            
            if include_trajectories:
                row['trajectory_x'] = json.dumps(sperm.trajectory_x)
                row['trajectory_y'] = json.dumps(sperm.trajectory_y)
                row['smoothed_trajectory_x'] = json.dumps(sperm.smoothed_trajectory_x)
                row['smoothed_trajectory_y'] = json.dumps(sperm.smoothed_trajectory_y)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def export_statistics_to_json(self, statistics: Dict, output_path: str):
        """
        Export population statistics to JSON file.
        
        Args:
            statistics: Statistics dictionary
            output_path: Output JSON file path
        """
        with open(output_path, 'w') as f:
            json.dump(statistics, f, indent=2)