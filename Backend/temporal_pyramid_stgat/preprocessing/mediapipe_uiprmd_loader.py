"""
UI-PRMD Loader with MediaPipe 33-joint representation.

Loads Vicon data, converts to MediaPipe 33-landmark format,
and extracts angles for training.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np

from temporal_pyramid_stgat.preprocessing.uiprmd_loader import UIPRMDLoader
from temporal_pyramid_stgat.preprocessing.mediapipe_mapper import MediaPipeMapper
from temporal_pyramid_stgat.preprocessing.mediapipe_angle_calculator import MediaPipeAngleCalculator


class MediaPipeUIsprmdLoader:
    """
    Load UI-PRMD data in MediaPipe 33-joint representation.
    
    Pipeline:
    1. Load Vicon 39-joint sequences from UI-PRMD
    2. Map to MediaPipe 33-landmark format
    3. Extract angular features using MediaPipe triplets
    4. Return normalized sequences for training
    """

    def __init__(self, data_root: str):
        """
        Args:
            data_root: Path to UI-PRMD root directory
        """
        self.data_root = Path(data_root)
        self.base_loader = UIPRMDLoader(str(data_root))
        self.num_joints = 33
        self.num_angles = MediaPipeAngleCalculator.NUM_ANGLES

    def load_all(self, exercise_id: int = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load all sequences in MediaPipe 33-joint format.
        
        Args:
            exercise_id: Filter by exercise ID (None = all exercises)
            
        Returns:
            coords: (N, T, 33, 3) - MediaPipe landmark sequences
            labels: (N,) - 0=correct, 1=incorrect
            metadata: List of metadata dicts
        """
        # Load original Vicon sequences
        vicon_coords, labels, metadata = self.base_loader.load_all(exercise_id)
        
        # Convert each to MediaPipe format
        mediapipe_coords = np.zeros(
            (vicon_coords.shape[0], vicon_coords.shape[1], 33, 3),
            dtype=np.float32
        )
        
        for idx in range(vicon_coords.shape[0]):
            mediapipe_coords[idx] = MediaPipeMapper.vicon_to_mediapipe(vicon_coords[idx])
        
        return mediapipe_coords, labels, metadata

    def load_with_angles(self, exercise_id: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Load sequences with both coordinates and angular features.
        
        Returns:
            coords: (N, T, 33, 3) - MediaPipe landmarks
            angles: (N, T, NUM_ANGLES) - Angular features (standardized across dataset)
            labels: (N,) - Binary labels
            metadata: List of metadata dicts
        """
        coords, labels, metadata = self.load_all(exercise_id)
        
        # Extract angles for each sequence (raw, not yet standardized)
        angles = np.zeros(
            (coords.shape[0], coords.shape[1], self.num_angles),
            dtype=np.float32
        )
        
        for idx in range(coords.shape[0]):
            angles[idx] = MediaPipeAngleCalculator.extract_angles(coords[idx])
        
        # Compute statistics across entire dataset (all sequences and frames)
        # Reshape to (N*T, NUM_ANGLES) for global statistics
        angles_flat = angles.reshape(-1, self.num_angles)
        mean = np.mean(angles_flat, axis=0, keepdims=True)
        std = np.std(angles_flat, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        
        # Standardize all angles using global statistics
        angles_standardized = (angles - mean) / std
        
        return coords, angles_standardized.astype(np.float32), labels, metadata

    def get_joint_names(self) -> List[str]:
        """Return names of the 33 MediaPipe landmarks."""
        return MediaPipeMapper.mediapipe_33_joint_names()
