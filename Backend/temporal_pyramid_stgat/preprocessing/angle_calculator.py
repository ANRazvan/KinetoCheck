"""
2D/3D Joint Angle Calculation

Computes flexion angles for skeleton joints for the two-stream approach.
Uses law of cosines to calculate angles between adjacent joints.
"""

import numpy as np
from typing import Tuple, List, Dict


class AngleCalculator:
    """
    Calculate joint angles from 3D skeleton positions.
    
    Used for the second stream in two-stream STGAT model.
    """
    
    # UI-PRMD 17-joint skeleton - anatomically connected joint triplets
    # Each triplet (parent, joint, child) defines an angle
    ANGLE_TRIPLETS_17 = [
        # Spine angles
        (0, 1, 2),    # Pelvis -> L5 -> L3
        (1, 2, 3),    # L5 -> L3 -> T12
        (2, 3, 4),    # L3 -> T12 -> T8
        (3, 4, 5),    # T12 -> T8 -> Neck
        (4, 5, 6),    # T8 -> Neck -> Head
        
        # Right arm
        (4, 7, 9),    # T8 -> R_Shoulder -> R_Arm
        (7, 9, 10),   # R_Shoulder -> R_Arm -> R_Forearm
        (9, 10, 11),  # R_Arm -> R_Forearm -> R_Hand
        
        # Left arm
        (4, 8, 12),   # T8 -> L_Shoulder -> L_Arm
        (8, 12, 13),  # L_Shoulder -> L_Arm -> L_Forearm
        (12, 13, 14), # L_Arm -> L_Forearm -> L_Hand
        
        # Right leg
        (0, 15, 15),  # Treated as special: use velocity instead
        
        # Left leg
        (0, 16, 16),  # Treated as special: use velocity instead
    ]
    
    NUM_ANGLES = len(ANGLE_TRIPLETS_17)
    
    @staticmethod
    def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute angle (in degrees) formed by three points.
        Using law of cosines: cos(theta) = (a^2 + b^2 - c^2) / (2*a*b)
        
        Args:
            p1, p2, p3: (3,) position vectors
            
        Returns:
            Angle in degrees [0, 180]
        """
        v1 = p1 - p2  # Vector from p2 to p1
        v2 = p3 - p2  # Vector from p2 to p3
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    @staticmethod
    def extract_angles(skeleton_seq: np.ndarray) -> np.ndarray:
        """
        Extract angles from skeleton sequence.
        
        Args:
            skeleton_seq: (T, 17, 3) array of joint positions
            
        Returns:
            angles: (T, num_angles) array
        """
        T = skeleton_seq.shape[0]
        num_angles = AngleCalculator.NUM_ANGLES
        angles = np.zeros((T, num_angles), dtype=np.float32)
        
        for t in range(T):
            for angle_idx, (p1_idx, p2_idx, p3_idx) in enumerate(AngleCalculator.ANGLE_TRIPLETS_17):
                p1 = skeleton_seq[t, p1_idx]
                p2 = skeleton_seq[t, p2_idx]
                p3 = skeleton_seq[t, p3_idx]
                
                angle = AngleCalculator.compute_angle(p1, p2, p3)
                angles[t, angle_idx] = angle
        
        return angles
    
    @staticmethod
    def extract_angle_velocities(angles: np.ndarray) -> np.ndarray:
        """
        Compute temporal derivatives of angles (angular velocities).
        
        Args:
            angles: (T, num_angles) array
            
        Returns:
            velocities: (T-1, num_angles) array
        """
        velocities = np.diff(angles, axis=0)
        return velocities
    
    @staticmethod
    def compute_angle_statistics(angles: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute angle statistics.
        
        Args:
            angles: (T, num_angles) array
            
        Returns:
            Dict with mean, std, min, max angles
        """
        return {
            "mean": np.mean(angles, axis=0),
            "std": np.std(angles, axis=0),
            "min": np.min(angles, axis=0),
            "max": np.max(angles, axis=0),
        }


class TwoStreamFeatures:
    """
    Combines coordinate-based and angle-based features for two-stream model.
    """
    
    @staticmethod
    def extract_both_streams(skeleton_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both coordinate stream and angle stream.
        
        Args:
            skeleton_seq: (T, 17, 3) coordinates
            
        Returns:
            coord_stream: (T, 17, 3) normalized coordinates
            angle_stream: (T, num_angles) angles
        """
        # Stream 1: Normalized coordinates
        coord_stream = AngleCalculator._normalize_coordinates(skeleton_seq)
        
        # Stream 2: Joint angles
        angle_stream = AngleCalculator.extract_angles(skeleton_seq)
        
        return coord_stream, angle_stream
    
    @staticmethod
    def _normalize_coordinates(skeleton_seq: np.ndarray, 
                               center_idx: int = 0) -> np.ndarray:
        """
        Normalize skeleton coordinates by centering on pelvis.
        
        Args:
            skeleton_seq: (T, 17, 3)
            center_idx: Joint index to use as origin (0=Pelvis)
            
        Returns:
            Centered coordinates
        """
        center = skeleton_seq[:, center_idx:center_idx+1, :]  # (T, 1, 3)
        normalized = skeleton_seq - center
        return normalized.astype(np.float32)
    
    @staticmethod
    def standardize_angles(angles: np.ndarray, 
                          mean: np.ndarray = None, 
                          std: np.ndarray = None) -> np.ndarray:
        """
        Standardize angles (z-score normalization).
        
        Args:
            angles: (T, num_angles)
            mean: Pre-computed mean or None
            std: Pre-computed std or None
            
        Returns:
            Standardized angles
        """
        if mean is None or std is None:
            mean = np.mean(angles, axis=0)
            std = np.std(angles, axis=0) + 1e-6
        
        return ((angles - mean) / std).astype(np.float32)
    
    @staticmethod
    def fuse_streams(coord_features: np.ndarray, 
                     angle_features: np.ndarray) -> np.ndarray:
        """
        Fuse two streams (concatenation).
        
        Args:
            coord_features: (T, 17, 3) or (T, 17*3)
            angle_features: (T, num_angles)
            
        Returns:
            Fused features
        """
        # Flatten coordinates if needed
        if coord_features.ndim == 3:
            coord_flat = coord_features.reshape(coord_features.shape[0], -1)
        else:
            coord_flat = coord_features
        
        fused = np.concatenate([coord_flat, angle_features], axis=1)
        return fused.astype(np.float32)
