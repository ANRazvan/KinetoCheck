import numpy as np
from config import settings


class SkeletonPreprocessor:
    """
    Normalize and prepare skeleton sequences for model input.
    Handles IntelliRehab (25 Kinect joints × 3D) data format.
    Based on the reference Preprocessor that works with IntelliRehab data.
    """

    def __init__(self, seq_length: int = None):
        self.seq_length = seq_length or settings.SEQUENCE_LENGTH

    def normalize(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Z-score normalization across the entire sequence.
        Centers each feature to mean=0, std=1.
        
        Args:
            keypoints: (num_frames, num_joints, 3) or (num_frames, num_joints, 2)
        Returns:
            Normalized keypoints of same shape.
        """
        mean = np.mean(keypoints)
        std = np.std(keypoints)
        if std > 0:
            keypoints = (keypoints - mean) / std
        return keypoints

    def pad_or_truncate(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Ensure sequence has exactly self.seq_length frames.
        Uses linear interpolation for resampling (like make_equal_length in reference code).
        """
        num_frames = keypoints.shape[0]

        if num_frames == self.seq_length:
            return keypoints

        if num_frames >= self.seq_length:
            # Uniformly sample seq_length frames
            indices = np.linspace(0, num_frames - 1, self.seq_length, dtype=int)
            return keypoints[indices]
        else:
            # Interpolate to upsample to target length
            original_indices = np.linspace(0, 1, num_frames)
            target_indices = np.linspace(0, 1, self.seq_length)
            
            # Flatten spatial dims, interpolate, reshape back
            original_shape = keypoints.shape  # (num_frames, joints, dim)
            flat = keypoints.reshape(num_frames, -1)  # (num_frames, joints*dim)
            
            interpolated = np.zeros((self.seq_length, flat.shape[1]), dtype=keypoints.dtype)
            for feat_idx in range(flat.shape[1]):
                interpolated[:, feat_idx] = np.interp(target_indices, original_indices, flat[:, feat_idx])
            
            return interpolated.reshape(self.seq_length, *original_shape[1:])

    def reshape_to_joints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Reshape flat feature array to (num_frames, num_joints, joint_dim).
        
        IntelliRehab: (num_frames, 75) → (num_frames, 25, 3)
        COCO/YOLO:    (num_frames, 34) → (num_frames, 17, 2)
        """
        if keypoints.ndim == 2:
            num_frames, num_features = keypoints.shape
            joint_dim = settings.KEYPOINT_DIM
            num_joints = num_features // joint_dim
            
            if num_features == settings.NUM_KEYPOINTS * settings.KEYPOINT_DIM:
                # IntelliRehab format: 25 × 3 = 75
                return keypoints.reshape(num_frames, settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM)
            elif num_features == settings.COCO_NUM_KEYPOINTS * settings.COCO_KEYPOINT_DIM:
                # COCO format: 17 × 2 = 34
                return keypoints.reshape(num_frames, settings.COCO_NUM_KEYPOINTS, settings.COCO_KEYPOINT_DIM)
            else:
                # Try to infer: assume 3D if divisible by 3, else 2D
                if num_features % 3 == 0:
                    return keypoints.reshape(num_frames, num_features // 3, 3)
                elif num_features % 2 == 0:
                    return keypoints.reshape(num_frames, num_features // 2, 2)
                else:
                    raise ValueError(
                        f"Cannot reshape features ({num_features}) into joints. "
                        f"Expected {settings.NUM_KEYPOINTS}×{settings.KEYPOINT_DIM}="
                        f"{settings.NUM_KEYPOINTS * settings.KEYPOINT_DIM}"
                    )
        return keypoints  # already 3D

    def process(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            keypoints: (num_frames, num_features) flat, or 
                       (num_frames, num_joints, joint_dim) already shaped
        Returns:
            (seq_length, num_joints, joint_dim) — ready for model input
        """
        # Step 1: Reshape flat features to (num_frames, num_joints, dim)
        keypoints = self.reshape_to_joints(keypoints)
        
        # Step 2: Normalize sequence length (pad or truncate to target length)
        keypoints = self.pad_or_truncate(keypoints)
        
        # Step 3: Z-score normalization
        keypoints = self.normalize(keypoints)
        
        return keypoints.astype(np.float32)