from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePreprocessor(ABC):
    """Base interface for dataset preprocessing pipelines."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    @abstractmethod
    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """Transform one skeleton sequence of shape [T, J, C]."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply preprocessing to a sample dict containing skeleton keypoints."""
        sample = dict(sample)
        sequence = self._extract_sequence(sample)
        transformed = self.transform(sequence)
        sample["keypoints"] = transformed.astype(np.float32, copy=False)
        return sample

    def _extract_sequence(self, sample: dict[str, Any]) -> np.ndarray:
        if "keypoints" not in sample:
            raise KeyError("Sample must contain 'keypoints'.")

        arr = np.asarray(sample["keypoints"], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected keypoints shape [T, J, C], got {arr.shape}."
            )
        return arr


class Standard17Preprocessor(BasePreprocessor):
    """Pipeline for converting raw skeletons to normalized 17-joint tensors."""

    # Generic 33-keypoint -> COCO-like 17-keypoint map.
    _DEFAULT_33_TO_17 = [
        0,   # nose
        2,   # left_eye
        5,   # right_eye
        7,   # left_ear
        8,   # right_ear
        11,  # left_shoulder
        12,  # right_shoulder
        13,  # left_elbow
        14,  # right_elbow
        15,  # left_wrist
        16,  # right_wrist
        23,  # left_hip
        24,  # right_hip
        25,  # left_knee
        26,  # right_knee
        27,  # left_ankle
        28,  # right_ankle
    ]

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)

        joint_cfg = self.cfg.get("joint_mapping", {})
        temporal_cfg = self.cfg.get("temporal", {})
        norm_cfg = self.cfg.get("normalization", {})

        self.joint_indices = joint_cfg.get("indices") or self._DEFAULT_33_TO_17
        self.max_frames = int(temporal_cfg.get("max_frames", 120))
        self.temporal_sampling = str(temporal_cfg.get("sampling", "uniform")).lower()
        self.temporal_padding = str(temporal_cfg.get("padding", "repeat_last")).lower()

        self.norm_enabled = bool(norm_cfg.get("enabled", True))
        self.norm_method = str(norm_cfg.get("method", "pelvis_center_scale")).lower()
        self.epsilon = float(norm_cfg.get("epsilon", 1e-6))

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        sequence = self._map_joints(sequence)

        if self.norm_enabled:
            sequence = self._normalize(sequence)

        sequence = self._fit_temporal_window(sequence)
        return sequence

    def _map_joints(self, sequence: np.ndarray) -> np.ndarray:
        num_joints = sequence.shape[1]
        indices = np.asarray(self.joint_indices, dtype=np.int64)

        if np.any(indices < 0) or np.any(indices >= num_joints):
            raise ValueError(
                "Joint mapping indices are out of bounds for input sequence. "
                f"Input joints={num_joints}, indices={indices.tolist()}"
            )

        return sequence[:, indices, :]

    def _normalize(self, sequence: np.ndarray) -> np.ndarray:
        if self.norm_method != "pelvis_center_scale":
            raise ValueError(f"Unsupported normalization method: {self.norm_method}")

        if sequence.shape[1] < 13:
            raise ValueError("Need at least 13 joints for pelvis normalization.")

        # In the mapped 17 format, left/right hip are 11 and 12.
        left_hip = sequence[:, 11, :]
        right_hip = sequence[:, 12, :]
        pelvis = 0.5 * (left_hip + right_hip)

        centered = sequence - pelvis[:, None, :]

        # In the mapped 17 format, left/right shoulder are 5 and 6.
        left_shoulder = centered[:, 5, :]
        right_shoulder = centered[:, 6, :]

        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=-1)
        scale = np.maximum(shoulder_dist, self.epsilon)

        centered /= scale[:, None, None]
        return centered

    def _fit_temporal_window(self, sequence: np.ndarray) -> np.ndarray:
        num_frames = sequence.shape[0]

        if num_frames == self.max_frames:
            return sequence

        if num_frames > self.max_frames:
            return self._downsample(sequence, self.max_frames)

        return self._pad(sequence, self.max_frames)

    def _downsample(self, sequence: np.ndarray, target_frames: int) -> np.ndarray:
        if self.temporal_sampling != "uniform":
            raise ValueError(f"Unsupported temporal sampling mode: {self.temporal_sampling}")

        idx = np.linspace(0, sequence.shape[0] - 1, target_frames).astype(np.int64)
        return sequence[idx]

    def _pad(self, sequence: np.ndarray, target_frames: int) -> np.ndarray:
        pad_count = target_frames - sequence.shape[0]
        if pad_count <= 0:
            return sequence

        if self.temporal_padding == "repeat_last":
            tail = np.repeat(sequence[-1:, :, :], pad_count, axis=0)
            return np.concatenate([sequence, tail], axis=0)

        if self.temporal_padding == "zeros":
            zeros = np.zeros((pad_count, sequence.shape[1], sequence.shape[2]), dtype=sequence.dtype)
            return np.concatenate([sequence, zeros], axis=0)

        raise ValueError(f"Unsupported temporal padding mode: {self.temporal_padding}")
