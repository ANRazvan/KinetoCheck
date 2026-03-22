"""
Data augmentation utilities for angle and position data.

Designed to handle video input noise:
- Pose extraction error (Gaussian noise on joint angles)
- Frame dropouts (missing detections)
- Temporal jitter (variable frame rates)
- Body scale variation (person distance from camera)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class AugmentationPipeline:
    """
    Augmentation for skeleton angle/position data.

    Maintains temporal and spatial coherence.
    """

    def __init__(
        self,
        angle_noise_std: float = 0.5,
        position_noise_std: float = 2.0,
        frame_dropout_p: float = 0.05,
        scale_jitter_range: tuple[float, float] = (0.95, 1.05),
        enable_angles: bool = True,
    ):
        """
        Args:
            angle_noise_std: Standard deviation of Gaussian noise on joint angles (degrees).
            position_noise_std: Standard deviation of Gaussian noise on positions (mm).
            frame_dropout_p: Probability of dropping each frame (temporal augmentation).
            scale_jitter_range: Factor range to scale positions/angles.
            enable_angles: If True, treat input as angles; else as positions.
        """
        self.angle_noise_std = angle_noise_std
        self.position_noise_std = position_noise_std
        self.frame_dropout_p = frame_dropout_p
        self.scale_jitter_range = scale_jitter_range
        self.enable_angles = enable_angles

    def __call__(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Apply augmentations.

        Args:
            data: Shape (seq_length, num_features)

        Returns:
            Augmented data, same shape and type as input.
        """
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            # Keep augmentation logic in numpy while preserving original tensor device/dtype on return.
            data_np = data.detach().cpu().numpy().copy()
        else:
            data_np = data.copy()

        # Apply augmentations in sequence
        data_np = self._add_noise(data_np)
        data_np = self._temporal_dropout(data_np)
        data_np = self._scale_jitter(data_np)
        data_np = self._clip_values(data_np)

        if is_torch:
            return torch.from_numpy(data_np).to(device=data.device, dtype=data.dtype)
        return data_np

    def _add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate pose extraction error."""
        std = self.angle_noise_std if self.enable_angles else self.position_noise_std
        noise = np.random.normal(0, std, data.shape)
        return data + noise

    def _temporal_dropout(self, data: np.ndarray) -> np.ndarray:
        """Randomly dropout frames to simulate detection failures."""
        if self.frame_dropout_p <= 0:
            return data

        if data.ndim < 2:
            return data

        seq_len = data.shape[0]
        flat = data.reshape(seq_len, -1)

        mask = np.random.rand(seq_len) > self.frame_dropout_p
        if mask.sum() < 2:
            # Keep at least 2 frames
            mask[:2] = True

        frames_to_keep = np.where(mask)[0]
        if len(frames_to_keep) == 0:
            return data

        # Linear interpolation over time for each flattened feature channel.
        out_flat = np.zeros_like(flat)
        x = np.arange(seq_len)
        if len(frames_to_keep) == 1:
            out_flat[:] = flat[frames_to_keep[0]]
        else:
            for j in range(flat.shape[1]):
                out_flat[:, j] = np.interp(x, frames_to_keep, flat[frames_to_keep, j])

        return out_flat.reshape(data.shape)

    def _scale_jitter(self, data: np.ndarray) -> np.ndarray:
        """Scale positions/angles to simulate person distance variation."""
        scale = np.random.uniform(*self.scale_jitter_range)
        return data * scale

    def _clip_values(self, data: np.ndarray) -> np.ndarray:
        """Clip extreme values to maintain stability."""
        if self.enable_angles:
            # Angles: clip to ±360 degrees
            return np.clip(data, -360, 360)
        else:
            # Positions: clip to ±5σ of typical range
            return np.clip(data, -1500, 1500)


class AugmentedDataset(Dataset):
    """
    Wrapper dataset that applies augmentation on-the-fly.

    Loosely couples to any Dataset (UIPRMDDataset, UIPRMDAnglesDataset, etc.).
    """

    def __init__(self, dataset: Dataset, augmentation: AugmentationPipeline | None = None):
        """
        Args:
            dataset: Base dataset to wrap.
            augmentation: Augmentation pipeline. If None, no augmentation applied.
        """
        self.dataset = dataset
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[idx]

        if self.augmentation is not None:
            x = self.augmentation(x)

        return x, y
