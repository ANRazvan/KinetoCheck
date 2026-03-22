"""
Preprocessor for UI-PRMD angles data.

Uses pre-computed joint angles (degrees) instead of positions.
Angles are rotation-invariant and more robust to pose extraction noise.

Joint angles are derived from the 17 anatomical landmarks and represent
relationships between connected body segments (e.g., knee bending angle).
"""

from __future__ import annotations

import numpy as np

from config import settings


class UIPRMDAnglesPreprocessor:
    """
    Normalize and prepare UI-PRMD angle sequences for model input.

    The output shape is ``(seq_length, num_angles)`` where num_angles
    is determined by the actual data (typically ~130-150 angles derived
    from 17 joints).

    Angles are already in degrees (0-360 or -180 to 180 range).
    """

    def __init__(self, seq_length: int | None = None, target_dim: int | None = None):
        self.seq_length = seq_length or settings.SEQUENCE_LENGTH
        self.num_angles: int | None = None
        self.target_dim = target_dim

    # ── public interface ─────────────────────────────────────────────

    def process(self, angles: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline: reshape → pad/truncate → normalize.

        Args:
            angles: ``(num_frames, num_angles)`` or similar flat structure
                   from a UI-PRMD angles file.

        Returns:
            ``np.ndarray`` of shape ``(seq_length, num_angles)``, dtype float32.
        """
        angles = self._validate_shape(angles)
        angles = self._align_feature_dim(angles)
        angles = self._pad_or_truncate(angles)
        angles = self._normalize(angles)
        # STGAT expects (seq, num_keypoints, keypoint_dim). For angles keypoint_dim=1.
        angles = np.expand_dims(angles, axis=-1)
        return angles.astype(np.float32)

    # ── internals ────────────────────────────────────────────────────

    def _validate_shape(self, angles: np.ndarray) -> np.ndarray:
        """Ensure shape is (num_frames, num_angles)."""
        if angles.ndim == 1:
            raise ValueError(
                f"UIPRMDAnglesPreprocessor: expected 2D array, got 1D with shape {angles.shape}. "
                "Angles file may be malformed."
            )
        if angles.ndim != 2:
            raise ValueError(
                f"UIPRMDAnglesPreprocessor: unexpected shape {angles.shape}. "
                "Expected (num_frames, num_angles)."
            )

        inferred_angles = angles.shape[1]
        if self.num_angles is None:
            self.num_angles = inferred_angles
        elif inferred_angles != self.num_angles:
            raise ValueError(
                f"UIPRMDAnglesPreprocessor: inconsistent angle count. "
                f"Expected {self.num_angles}, got {inferred_angles}."
            )
        return angles

    def _align_feature_dim(self, angles: np.ndarray) -> np.ndarray:
        """Align angle feature dimension to target_dim via truncate/pad when requested."""
        if self.target_dim is None:
            return angles

        curr_dim = angles.shape[1]
        if curr_dim == self.target_dim:
            return angles
        if curr_dim > self.target_dim:
            return angles[:, : self.target_dim]

        pad = np.zeros((angles.shape[0], self.target_dim - curr_dim), dtype=angles.dtype)
        return np.concatenate([angles, pad], axis=1)

    def _pad_or_truncate(self, angles: np.ndarray) -> np.ndarray:
        """Resample to exactly ``self.seq_length`` frames via linear interpolation."""
        num_frames = angles.shape[0]
        if num_frames == self.seq_length:
            return angles

        if num_frames > self.seq_length:
            # Downsample: take evenly spaced indices
            indices = np.linspace(0, num_frames - 1, self.seq_length, dtype=int)
            return angles[indices]

        # Upsample: interpolate each angle channel
        orig_idx = np.linspace(0, 1, num_frames)
        tgt_idx = np.linspace(0, 1, self.seq_length)
        out = np.zeros((self.seq_length, angles.shape[1]), dtype=np.float32)
        for i in range(angles.shape[1]):
            out[:, i] = np.interp(tgt_idx, orig_idx, angles[:, i])
        return out

    def _normalize(self, angles: np.ndarray) -> np.ndarray:
        """Z-score normalization across the whole sequence."""
        mean = np.mean(angles)
        std = np.std(angles)
        if std > 0:
            angles = (angles - mean) / std
        # Clip outliers to prevent extreme values from dominatin after augmentation
        angles = np.clip(angles, -3.0, 3.0)  # ~99.7% of normal distribution
        return angles
