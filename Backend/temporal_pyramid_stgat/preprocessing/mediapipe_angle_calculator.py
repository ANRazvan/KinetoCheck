"""
Angle calculator for MediaPipe 33-landmark format.

Extracts angular features from joint sequences to capture movement patterns
independent of scale and translation.
"""

import numpy as np
from typing import List, Tuple


class MediaPipeAngleCalculator:
    """
    Calculate joint angles from MediaPipe 33-landmark sequences.
    
    Selects anatomically meaningful joint triplets (parent-joint-child)
    and computes the angle at the middle joint.
    """

    # Key anatomical angle triplets in MediaPipe format
    # (parent_idx, joint_idx, child_idx)
    # These represent major body segments and their movements
    ANGLE_TRIPLETS_33 = [
        # Right arm
        (12, 13, 15),   # Right shoulder -> elbow -> wrist
        (13, 15, 17),   # Right elbow -> wrist -> palm
        # Left arm
        (11, 14, 16),   # Left shoulder -> elbow -> wrist
        (14, 16, 18),   # Left elbow -> wrist -> palm
        # Right leg
        (25, 27, 29),   # Right hip -> knee -> ankle
        (27, 29, 31),   # Right knee -> ankle -> heel
        # Left leg
        (26, 28, 30),   # Left hip -> knee -> ankle
        (28, 30, 32),   # Left knee -> ankle -> heel
        # Torso/core
        (25, 26, 12),   # Right hip -> left hip -> right shoulder (pelvis-shoulder)
        (26, 25, 11),   # Left hip -> right hip -> left shoulder (pelvis-shoulder)
        (11, 12, 25),   # Left shoulder -> right shoulder -> right hip
        (12, 11, 26),   # Right shoulder -> left shoulder -> left hip
    ]

    ANGLE_NAMES = [
        "Right elbow flexion",
        "Right wrist bend",
        "Left elbow flexion",
        "Left wrist bend",
        "Right knee flexion",
        "Right ankle bend",
        "Left knee flexion",
        "Left ankle bend",
        "Pelvis-to-right-shoulder tilt",
        "Pelvis-to-left-shoulder tilt",
        "Shoulder-hip right chain",
        "Shoulder-hip left chain",
    ]

    NUM_ANGLES = len(ANGLE_TRIPLETS_33)

    @staticmethod
    def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute angle at p2 formed by vectors (p1->p2) and (p2->p3).
        
        Args:
            p1, p2, p3: 3D points (x, y, z)
            
        Returns:
            Angle in radians [0, π]
        """
        v1 = p1 - p2
        v2 = p3 - p2

        # Avoid division by zero
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        if len_v1 < 1e-6 or len_v2 < 1e-6:
            return 0.0

        cos_angle = np.clip(
            np.dot(v1, v2) / (len_v1 * len_v2),
            -1.0, 1.0
        )
        return float(np.arccos(cos_angle))

    @classmethod
    def extract_angles(cls, sequence: np.ndarray) -> np.ndarray:
        """
        Extract angular features from a MediaPipe 33-landmark sequence.
        
        Args:
            sequence: (T, 33, 3) array of MediaPipe landmarks
            
        Returns:
            (T, NUM_ANGLES) array of angle values in radians
        """
        T = sequence.shape[0]
        angles = np.zeros((T, cls.NUM_ANGLES), dtype=np.float32)

        for frame_idx in range(T):
            for angle_idx, (p1_idx, p2_idx, p3_idx) in enumerate(cls.ANGLE_TRIPLETS_33):
                if (p1_idx < 33 and p2_idx < 33 and p3_idx < 33):
                    p1 = sequence[frame_idx, p1_idx, :]
                    p2 = sequence[frame_idx, p2_idx, :]
                    p3 = sequence[frame_idx, p3_idx, :]
                    angles[frame_idx, angle_idx] = cls.compute_angle(p1, p2, p3)

        return angles

    @staticmethod
    def standardize_angles(angle_array: np.ndarray) -> np.ndarray:
        """
        Standardize angle features (zero mean, unit std).
        
        Args:
            angle_array: (T, NUM_ANGLES) array
            
        Returns:
            (T, NUM_ANGLES) standardized array
        """
        mean = np.mean(angle_array, axis=0, keepdims=True)
        std = np.std(angle_array, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (angle_array - mean) / std
