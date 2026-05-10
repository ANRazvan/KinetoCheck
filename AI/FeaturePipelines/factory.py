from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _safe_norm(v: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    return np.linalg.norm(v, axis=axis) + eps


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute angle ABC over time in degrees for inputs shaped (T, 3)."""
    ba = a - b
    bc = c - b
    cosang = (ba * bc).sum(axis=-1) / (_safe_norm(ba) * _safe_norm(bc))
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang)).astype(np.float32)


class MotionFeatureExtractor(ABC):
    """Strategy product: converts (T, 17, 3) aligned sequence into (C, T, 17)."""

    name: str
    in_channels: int

    @abstractmethod
    def build(self, processed: np.ndarray) -> np.ndarray:
        pass


class BaselineXYZFeatureExtractor(MotionFeatureExtractor):
    name = "baseline_xyz"
    in_channels = 9

    def build(self, processed: np.ndarray) -> np.ndarray:
        velocity = np.diff(processed, axis=0, prepend=processed[:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
        features = np.concatenate([processed, velocity, acceleration], axis=-1)  # (T, 17, 9)
        return np.transpose(features, (2, 0, 1)).astype(np.float32, copy=True)


class AngleAugmentedFeatureExtractor(MotionFeatureExtractor):
    """
    Adds interpretable angle channels without changing graph nodes.
    Channels: xyz + vel + acc + angle + angle_vel + angle_acc => 12
    """

    name = "angles_v1"
    in_channels = 12

    def _compute_angle_map(self, processed: np.ndarray) -> np.ndarray:
        # Joint order is the project COCO17 order:
        # 0:nose,1:l_shoulder,2:r_shoulder,3:l_elbow,4:r_elbow,5:l_wrist,6:r_wrist,
        # 7:l_hip,8:r_hip,9:l_knee,10:r_knee,11:l_ankle,12:r_ankle,13:l_heel,14:r_heel,15:l_foot,16:r_foot
        t = processed.shape[0]
        angle_map = np.zeros((t, 17, 1), dtype=np.float32)

        ls, rs = 1, 2
        le, re = 3, 4
        lw, rw = 5, 6
        lh, rh = 7, 8
        lk, rk = 9, 10
        la, ra = 11, 12

        # Elbow flexion
        angle_map[:, le, 0] = _angle_deg(processed[:, ls], processed[:, le], processed[:, lw])
        angle_map[:, re, 0] = _angle_deg(processed[:, rs], processed[:, re], processed[:, rw])

        # Knee flexion
        angle_map[:, lk, 0] = _angle_deg(processed[:, lh], processed[:, lk], processed[:, la])
        angle_map[:, rk, 0] = _angle_deg(processed[:, rh], processed[:, rk], processed[:, ra])

        # Hip flexion proxy
        angle_map[:, lh, 0] = _angle_deg(processed[:, ls], processed[:, lh], processed[:, lk])
        angle_map[:, rh, 0] = _angle_deg(processed[:, rs], processed[:, rh], processed[:, rk])

        # Trunk lean proxy duplicated to both hips (keeps channels node-local)
        shoulder_mid = 0.5 * (processed[:, ls] + processed[:, rs])
        hip_mid = 0.5 * (processed[:, lh] + processed[:, rh])
        up = shoulder_mid + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        trunk = _angle_deg(up, hip_mid, shoulder_mid)
        angle_map[:, lh, 0] = 0.5 * (angle_map[:, lh, 0] + trunk)
        angle_map[:, rh, 0] = 0.5 * (angle_map[:, rh, 0] + trunk)

        return angle_map

    def build(self, processed: np.ndarray) -> np.ndarray:
        velocity = np.diff(processed, axis=0, prepend=processed[:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])

        angle_map = self._compute_angle_map(processed)
        angle_vel = np.diff(angle_map, axis=0, prepend=angle_map[:1])
        angle_acc = np.diff(angle_vel, axis=0, prepend=angle_vel[:1])

        features = np.concatenate([processed, velocity, acceleration, angle_map, angle_vel, angle_acc], axis=-1)  # (T, 17, 12)
        return np.transpose(features, (2, 0, 1)).astype(np.float32, copy=True)


class FeatureExtractorFactory(ABC):
    """Abstract factory for feature methods."""

    @staticmethod
    def create(method: str) -> MotionFeatureExtractor:
        normalized = method.strip().lower()
        if normalized in {"baseline", "baseline_xyz", "xyz"}:
            return BaselineXYZFeatureExtractor()
        if normalized in {"angles", "angles_v1", "angle"}:
            return AngleAugmentedFeatureExtractor()
        raise ValueError(f"Unknown feature method: {method}")
