"""
UI-PRMD preprocessor.

Implements the same core data treatment used in the data visualization notebook:
1) Keep native Vicon joint ordering from file columns (no reindexing)
2) Reshape flat features to (frames, joints, dims)
3) Resample to fixed sequence length with linear interpolation
4) Normalize with z-score across the sequence

Supports both:
- 39-joint Vicon sequences (117 features in 3D)
- 17-joint variants used by parts of the existing pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
	from config import settings
except ModuleNotFoundError:
	# Allow running this file directly while still resolving Backend/config.py.
	backend_root = Path(__file__).resolve().parents[2]
	backend_root_str = str(backend_root)
	if backend_root_str not in sys.path:
		sys.path.insert(0, backend_root_str)
	from config import settings


class UIPRMDPreprocessor:
	"""
	Normalize and standardize UI-PRMD skeleton sequences.

	Notes on joint order:
	- This preprocessor intentionally preserves the incoming joint order.
	- For Vicon files, this means the same 39-joint column order from the raw
	  dataset is kept end-to-end for training/inference consistency.
	"""

	_instance: "UIPRMDPreprocessor | None" = None

	# 39-joint Vicon names aligned to the canonical UI-PRMD column order.
	# These are metadata-only and are not used to reorder the data.
	VICON_39_JOINT_NAMES = [
		"LFHD", "RFHD", "LBHD", "RBHD",
		"C7", "T10", "CLAV", "STRN", "RBAK",
		"LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
		"RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
		"LASI", "RASI", "LPSI", "RPSI",
		"LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
		"RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
	]

	def __new__(cls, seq_length: int | None = None, keypoint_dim: int | None = None):
		# Mirror existing preprocessor behavior: singleton only for default config.
		if seq_length is None and keypoint_dim is None:
			if cls._instance is None:
				cls._instance = super().__new__(cls)
			return cls._instance
		return super().__new__(cls)

	def __init__(self, seq_length: int | None = None, keypoint_dim: int | None = None):
		if hasattr(self, "seq_length") and seq_length is None and keypoint_dim is None:
			return

		self.seq_length = seq_length or settings.SEQUENCE_LENGTH
		self.keypoint_dim = keypoint_dim or settings.UIPRMD_KEYPOINT_DIM

	def align_vicon_to_mediapipe(self, vicon_data: np.ndarray) -> np.ndarray:
		"""
		Align Vicon markers to the 17 MediaPipe-compatible landmarks used by training.

		Accepted input shapes:
		- (frames, 39, 3)
		- (frames, 117)
		- (frames, 17, 3)
		- (frames, 51)
		- (39, 3) for a single frame
		- (17, 3) for a single frame

		Returns:
		- (frames, 17, 3) in this order:
		  [MP0, MP11, MP12, MP13, MP14, MP15, MP16, MP23, MP24,
		   MP25, MP26, MP27, MP28, MP29, MP30, MP31, MP32]

		Normalization:
		- Translates all landmarks per frame so midpoint(MP23, MP24) is at (0, 0, 0).
		"""
		arr = np.asarray(vicon_data, dtype=np.float32)

		if arr.ndim == 2 and arr.shape in {(39, 3), (17, 3)}:
			arr = arr[np.newaxis, ...]
		elif arr.ndim == 2 and arr.shape[1] in {117, 51}:
			arr = arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)
		elif arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[1] in {17, 39}:
			pass
		else:
			raise ValueError(
				"Expected Vicon data with shape (frames, 39, 3), (frames, 117), "
				"(frames, 17, 3), (frames, 51), (39, 3), or (17, 3). "
				f"Got {arr.shape}."
			)

		def marker(vicon_idx_1_based: int) -> np.ndarray:
			return arr[:, vicon_idx_1_based - 1, :]

		def midpoint(idx_a: int, idx_b: int) -> np.ndarray:
			return (marker(idx_a) + marker(idx_b)) * 0.5

		if arr.shape[1] == 39:
			aligned = np.stack(
				[
					midpoint(1, 2),      # MP0  Nose
					marker(10),          # MP11 L Shoulder
					marker(17),          # MP12 R Shoulder
					marker(12),          # MP13 L Elbow
					marker(19),          # MP14 R Elbow
					midpoint(14, 15),    # MP15 L Wrist
					midpoint(21, 22),    # MP16 R Wrist
					midpoint(24, 26),    # MP23 L Hip
					midpoint(25, 27),    # MP24 R Hip
					marker(29),          # MP25 L Knee
					marker(35),          # MP26 R Knee
					marker(31),          # MP27 L Ankle
					marker(37),          # MP28 R Ankle
					marker(32),          # MP29 L Heel
					marker(38),          # MP30 R Heel
					marker(33),          # MP31 L Foot Index
					marker(39),          # MP32 R Foot Index
				],
				axis=1,
			)
		else:
			# Already in 17-joint MediaPipe-compatible layout.
			aligned = arr

		# Hip-centered translation: midpoint of MP23 and MP24 is the origin.
		hip_center = (aligned[:, 7, :] + aligned[:, 8, :]) * 0.5
		aligned = aligned - hip_center[:, np.newaxis, :]

		return aligned.astype(np.float32)

	def normalize(self, keypoints: np.ndarray) -> np.ndarray:
		"""
		Apply z-score normalization over the full sequence tensor.
		"""
		mean = float(np.mean(keypoints))
		std = float(np.std(keypoints))
		if std > 0.0:
			keypoints = (keypoints - mean) / std
		return keypoints

	def _resample_linear(self, keypoints: np.ndarray, target_len: int) -> np.ndarray:
		"""
		Resample sequence length with feature-wise linear interpolation.
		"""
		num_frames = keypoints.shape[0]
		if num_frames == target_len:
			return keypoints

		# Degenerate case: repeat a single frame.
		if num_frames == 1:
			return np.repeat(keypoints, target_len, axis=0)

		original_shape = keypoints.shape
		flat = keypoints.reshape(num_frames, -1)

		x_old = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
		x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)

		out = np.zeros((target_len, flat.shape[1]), dtype=np.float32)
		for feat_idx in range(flat.shape[1]):
			out[:, feat_idx] = np.interp(x_new, x_old, flat[:, feat_idx])

		return out.reshape(target_len, *original_shape[1:])

	def pad_or_truncate(self, keypoints: np.ndarray) -> np.ndarray:
		"""
		Match target sequence length via interpolation.
		"""
		return self._resample_linear(keypoints, self.seq_length)

	def _reshape_from_flat(self, keypoints: np.ndarray) -> np.ndarray:
		"""
		Convert flat frame features to (frames, joints, dims).

		Accepted examples:
		- (T, 117) with keypoint_dim=3 -> (T, 39, 3)
		- (T, 51) with keypoint_dim=3 -> (T, 17, 3)
		- (T, 34) with keypoint_dim=2 -> (T, 17, 2)
		"""
		if keypoints.ndim != 2:
			raise ValueError("Expected a 2D flat array for _reshape_from_flat.")

		num_frames, num_features = keypoints.shape
		if num_features % self.keypoint_dim != 0:
			raise ValueError(
				f"Cannot reshape {num_features} features using keypoint_dim={self.keypoint_dim}."
			)

		num_joints = num_features // self.keypoint_dim
		return keypoints.reshape(num_frames, num_joints, self.keypoint_dim)

	def reshape_to_joints(self, keypoints: np.ndarray) -> np.ndarray:
		"""
		Ensure output shape is always (frames, joints, keypoint_dim).
		"""
		arr = np.asarray(keypoints, dtype=np.float32)

		if arr.ndim == 2:
			return self._reshape_from_flat(arr)

		if arr.ndim != 3:
			raise ValueError(
				f"Expected keypoints with 2 or 3 dims, got shape={arr.shape}."
			)

		# Already in (T, J, D); convert D to requested keypoint_dim when feasible.
		if arr.shape[2] == self.keypoint_dim:
			return arr

		if arr.shape[2] == 3 and self.keypoint_dim == 2:
			return arr[:, :, :2]

		if arr.shape[2] == 2 and self.keypoint_dim == 3:
			return np.pad(arr, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)

		raise ValueError(
			f"Unsupported keypoint dim conversion from {arr.shape[2]} to {self.keypoint_dim}."
		)

	def center_by_pelvis(self, keypoints: np.ndarray) -> np.ndarray:
		"""
		Optional pelvis-centering used in notebook exploration for 39-joint data.

		This is provided for parity with notebook experiments and is intentionally
		not forced in `process()` to avoid changing existing training behavior.
		"""
		if keypoints.ndim != 3 or keypoints.shape[2] != 3:
			return keypoints

		# Notebook anchor: mean of joints [0, 23, 31] for 39-joint Vicon layout.
		if keypoints.shape[1] >= 32:
			pelvis_center = (keypoints[:, 0] + keypoints[:, 23] + keypoints[:, 31]) / 3.0
			return keypoints - pelvis_center[:, np.newaxis, :]

		return keypoints

	def process(self, keypoints: np.ndarray) -> np.ndarray:
		"""
		Full preprocessing pipeline for UI-PRMD.

		Steps:
		1) Reshape to (T, J, D)
		2) Resample sequence to configured length
		3) Z-score normalization
		"""
		seq = self.reshape_to_joints(keypoints)
		seq = self.pad_or_truncate(seq)
		seq = self.normalize(seq)
		return seq.astype(np.float32)