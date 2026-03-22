"""
Preprocessor for the UI-PRMD (University of Idaho - Physical Rehabilitation
Movement Dataset) skeleton format.

UI-PRMD joint model
-------------------
The dataset provides 3-D positions for **17 anatomical landmarks** captured
by a Vicon motion-capture system.  Joint ordering (0-indexed):

    0  Pelvis              9  RightUpArm
    1  L5                  10 RightForeArm
    2  L3                  11 RightHand
    3  T12                 12 LeftUpArm
    4  T8                  13 LeftForeArm
    5  Neck                14 LeftHand
    6  Head                15 RightUpLeg
    7  RightShoulder       16 LeftUpLeg
    8  LeftShoulder

Each frame is stored as a flat row of 51 values (17 joints × 3 coords).

The preprocessor exposes the same ``process()`` interface as
``SkeletonPreprocessor`` so it is a drop-in product for
``AbstractTrainingFactory.create_preprocessor()``.
"""

from __future__ import annotations

import numpy as np

from config import settings

# UI-PRMD coordinates are 3D (x, y, z). Number of joints can vary by export.
UIPRMD_JOINT_DIM: int = 3


class UIPRMDPreprocessor:
    """
    Normalize and prepare UI-PRMD skeleton sequences for model input.

    The output shape is ``(seq_length, num_joints, 3)`` — identical to what
    ``SkeletonPreprocessor`` produces for IntelliRehab data, so that both
    datasets can feed the same ST-GAT model architecture.

    Note: The model's ``NUM_KEYPOINTS`` config is *overridden* by
    ``UIPRMD_NUM_JOINTS`` here; make sure the model ``build()`` call
    receives ``num_keypoints=17`` when using this preprocessor.
    """

    def __init__(self, seq_length: int | None = None):
        self.seq_length = seq_length or settings.SEQUENCE_LENGTH
        self.num_joints: int | None = None

    # ── public interface (mirrors SkeletonPreprocessor) ─────────────

    def process(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline: reshape → pad/truncate → normalize.

        Args:
            keypoints: ``(num_frames, 51)`` flat row from a UI-PRMD file,
                       or already shaped ``(num_frames, 17, 3)``.

        Returns:
            ``np.ndarray`` of shape ``(seq_length, 17, 3)``, dtype float32.
        """
        keypoints = self._reshape(keypoints)
        keypoints = self._pad_or_truncate(keypoints)
        keypoints = self._normalize(keypoints)
        return keypoints.astype(np.float32)

    # ── internals ───────────────────────────────────────────────────

    def _reshape(self, keypoints: np.ndarray) -> np.ndarray:
        """Reshape flat arrays to ``(num_frames, num_joints, 3)`` with inferred joints."""
        if keypoints.ndim == 2:
            feature_count = keypoints.shape[1]
            if feature_count % UIPRMD_JOINT_DIM != 0:
                raise ValueError(
                    f"UIPRMDPreprocessor: feature count {feature_count} is not divisible by "
                    f"{UIPRMD_JOINT_DIM}."
                )
            inferred_joints = feature_count // UIPRMD_JOINT_DIM
            if self.num_joints is None:
                self.num_joints = inferred_joints
            elif inferred_joints != self.num_joints:
                raise ValueError(
                    f"UIPRMDPreprocessor: inconsistent joint count. "
                    f"Expected {self.num_joints}, got {inferred_joints}."
                )
            return keypoints.reshape(keypoints.shape[0], self.num_joints, UIPRMD_JOINT_DIM)

        if keypoints.ndim == 3 and keypoints.shape[2] == UIPRMD_JOINT_DIM:
            inferred_joints = keypoints.shape[1]
            if self.num_joints is None:
                self.num_joints = inferred_joints
            elif inferred_joints != self.num_joints:
                raise ValueError(
                    f"UIPRMDPreprocessor: inconsistent joint count. "
                    f"Expected {self.num_joints}, got {inferred_joints}."
                )
            return keypoints

        raise ValueError(
            f"UIPRMDPreprocessor: unexpected keypoints shape {keypoints.shape}. "
            f"Expected (N, joints*{UIPRMD_JOINT_DIM}) or (N, joints, {UIPRMD_JOINT_DIM})."
        )

    def _pad_or_truncate(self, keypoints: np.ndarray) -> np.ndarray:
        """Resample to exactly ``self.seq_length`` frames via linear interpolation."""
        num_frames = keypoints.shape[0]
        if num_frames == self.seq_length:
            return keypoints

        if num_frames > self.seq_length:
            indices = np.linspace(0, num_frames - 1, self.seq_length, dtype=int)
            return keypoints[indices]

        # Upsample: interpolate each feature channel
        orig_idx = np.linspace(0, 1, num_frames)
        tgt_idx = np.linspace(0, 1, self.seq_length)
        flat = keypoints.reshape(num_frames, -1)
        out = np.zeros((self.seq_length, flat.shape[1]), dtype=np.float32)
        for i in range(flat.shape[1]):
            out[:, i] = np.interp(tgt_idx, orig_idx, flat[:, i])
        return out.reshape(self.seq_length, keypoints.shape[1], keypoints.shape[2])

    def _normalize(self, keypoints: np.ndarray) -> np.ndarray:
        """Z-score normalization across the whole sequence."""
        mean = np.mean(keypoints)
        std = np.std(keypoints)
        if std > 0:
            keypoints = (keypoints - mean) / std
        return keypoints
