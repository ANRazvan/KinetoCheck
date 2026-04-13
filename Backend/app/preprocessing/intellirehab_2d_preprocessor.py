"""
Preprocessor for IntelliRehab 2D training/inference.

Input format:
- (num_frames, 25, 2) or
- (num_frames, 50)

Output format:
- (seq_length, 25, 2)

Normalization strategy:
- Root-relative translation (SpineBase at origin)
- Sequence-level bbox scaling (same rule used for both train and inference)
"""

from __future__ import annotations

import numpy as np

from config import settings


class IntelliRehab2DPreprocessor:
    """Prepare Kinect-25 2D skeletons for model input."""

    def __init__(self, seq_length: int | None = None):
        self.seq_length = seq_length or settings.SEQUENCE_LENGTH
        self.num_joints = settings.NUM_KEYPOINTS
        self.keypoint_dim = 2
        self.root_joint_index = 0  # SpineBase

    def process(self, keypoints: np.ndarray) -> np.ndarray:
        arr = self._reshape(keypoints)
        arr = self._pad_or_truncate(arr)
        arr = self._normalize_root_bbox(arr)
        return arr.astype(np.float32)

    def _reshape(self, keypoints: np.ndarray) -> np.ndarray:
        arr = np.asarray(keypoints, dtype=np.float32)

        if arr.ndim == 2:
            feature_count = arr.shape[1]
            expected = self.num_joints * self.keypoint_dim
            if feature_count != expected:
                raise ValueError(
                    f"IntelliRehab2DPreprocessor expected {expected} features, got {feature_count}."
                )
            return arr.reshape(arr.shape[0], self.num_joints, self.keypoint_dim)

        if arr.ndim == 3 and arr.shape[1] == self.num_joints and arr.shape[2] == self.keypoint_dim:
            return arr

        raise ValueError(
            "IntelliRehab2DPreprocessor expected shape (N, 50) or (N, 25, 2), "
            f"got {arr.shape}."
        )

    def _pad_or_truncate(self, keypoints: np.ndarray) -> np.ndarray:
        num_frames = keypoints.shape[0]

        if num_frames == self.seq_length:
            return keypoints

        if num_frames > self.seq_length:
            indices = np.linspace(0, num_frames - 1, self.seq_length, dtype=int)
            return keypoints[indices]

        orig_idx = np.linspace(0.0, 1.0, num_frames)
        tgt_idx = np.linspace(0.0, 1.0, self.seq_length)

        flat = keypoints.reshape(num_frames, -1)
        out = np.zeros((self.seq_length, flat.shape[1]), dtype=np.float32)
        for i in range(flat.shape[1]):
            out[:, i] = np.interp(tgt_idx, orig_idx, flat[:, i])

        return out.reshape(self.seq_length, self.num_joints, self.keypoint_dim)

    def _normalize_root_bbox(self, keypoints: np.ndarray) -> np.ndarray:
        # Root-relative translation (SpineBase -> origin)
        root_xy = keypoints[:, self.root_joint_index : self.root_joint_index + 1, :]
        centered = keypoints - root_xy

        # Sequence-level isotropic scaling from the same bbox rule at train/inference.
        x_min = float(np.min(centered[:, :, 0]))
        x_max = float(np.max(centered[:, :, 0]))
        y_min = float(np.min(centered[:, :, 1]))
        y_max = float(np.max(centered[:, :, 1]))

        width = x_max - x_min
        height = y_max - y_min
        scale = max(width, height)
        if scale < 1e-6:
            scale = 1.0

        return centered / scale
