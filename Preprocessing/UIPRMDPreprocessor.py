import Config.config as cfg
import numpy as np

# ---------------------------------------------------------------------------
# COCO-17 joint indices:
#  0:nose  1:l_shoulder  2:r_shoulder  3:l_elbow   4:r_elbow
#  5:l_wrist  6:r_wrist  7:l_hip  8:r_hip  9:l_knee  10:r_knee
#  11:l_ankle 12:r_ankle 13:l_heel 14:r_heel 15:l_foot 16:r_foot
# ---------------------------------------------------------------------------

# (parent, vertex, child) — angle computed AT the vertex joint
ANGLE_TRIPLETS = [
    (7,  9,  11),   # l_hip  → l_knee  → l_ankle   (left  knee flexion)
    (8,  10, 12),   # r_hip  → r_knee  → r_ankle   (right knee flexion)
    (9,  7,  1),    # l_knee → l_hip   → l_shoulder (left  hip flexion)
    (10, 8,  2),    # r_knee → r_hip   → r_shoulder (right hip flexion)
    (11, 9,  7),    # l_ankle→ l_knee  → l_hip
    (12, 10, 8),    # r_ankle→ r_knee  → r_hip
    (1,  3,  5),    # l_shoulder → l_elbow → l_wrist
    (2,  4,  6),    # r_shoulder → r_elbow → r_wrist
    (3,  1,  7),    # l_elbow    → l_shoulder → l_hip
    (4,  2,  8),    # r_elbow    → r_shoulder → r_hip
    (7,  8,  10),   # l_hip → r_hip → r_knee  (pelvic tilt proxy)
    (1,  0,  2),    # l_shoulder → nose → r_shoulder
]

BONE_PAIRS = [
    (0, 1), (0, 2),
    (1, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11),
    (8, 10),(10, 12),
    (11,13),(12,14),
]


def compute_joint_angles(seq: np.ndarray) -> np.ndarray:
    """
    Input : (T, 17, 3)
    Output: (T, 17)  — angle in radians at each joint.

    Each triplet's angle is accumulated at the vertex joint index.
    Joints with multiple triplets get the mean. Joints with none stay 0.
    Always returns shape (T, 17) so it concatenates cleanly with (T, 17, 3).
    """
    T = seq.shape[0]
    angles      = np.zeros((T, 17), dtype=np.float32)
    angle_count = np.zeros(17, dtype=np.int32)

    for (a, v, b) in ANGLE_TRIPLETS:
        vec1  = seq[:, a, :] - seq[:, v, :]
        vec2  = seq[:, b, :] - seq[:, v, :]
        n1    = np.linalg.norm(vec1, axis=-1)
        n2    = np.linalg.norm(vec2, axis=-1)
        cos_a = (vec1 * vec2).sum(-1) / (n1 * n2 + 1e-6)
        angles[:, v] += np.arccos(np.clip(cos_a, -1.0, 1.0))
        angle_count[v] += 1

    for j in range(17):
        if angle_count[j] > 1:
            angles[:, j] /= angle_count[j]

    return angles   # (T, 17)


def compute_bone_lengths_normalized(seq: np.ndarray) -> np.ndarray:
    """
    Input : (T, 17, 3)
    Output: (T, 16)  bone lengths divided by torso length (scale-invariant).
    """
    torso = np.linalg.norm(seq[:, 7, :] - seq[:, 1, :], axis=-1, keepdims=True)
    torso = np.maximum(torso, 1e-4)
    lengths = np.stack(
        [np.linalg.norm(seq[:, a, :] - seq[:, b, :], axis=-1) for a, b in BONE_PAIRS],
        axis=-1,
    )
    return lengths / torso   # (T, 16)


def rom_normalize(seq: np.ndarray) -> np.ndarray:
    """
    Per-sequence, per-joint Range-of-Motion normalization.

    Input : (T, J, D)
    Output: (T, J, D)  — each joint's trajectory rescaled to [0, 1] within
            its own min/max for this sequence.

    "Full squat depth" becomes 1.0 for everyone regardless of height or
    coordinate system. Joints that barely move (range < 1e-4) are set to 0.5.
    """
    T, J, D = seq.shape
    out = np.empty_like(seq)
    for j in range(J):
        for d in range(D):
            traj = seq[:, j, d]
            lo, hi = float(traj.min()), float(traj.max())
            if hi - lo > 1e-4:
                out[:, j, d] = (traj - lo) / (hi - lo)
            else:
                out[:, j, d] = 0.5
    return out.astype(np.float32)


def build_features_from_aligned(processed: np.ndarray) -> np.ndarray:
    """
    Convert a preprocessed (T, 17, 3) sequence into a (12, T, 17) model tensor.

    Channels 0-2  : ROM-normalised, z-scored XYZ positions
    Channels 3-5  : velocity  (delta position per frame)
    Channels 6-8  : acceleration
    Channel  9    : joint angle in radians (geometry-invariant)
    Channel  10   : angular velocity
    Channel  11   : bone-length ratio normalised by torso length

    `processed` must be the output of preprocessor.process() — already
    ROM-normalised and z-scored, shape (T, 17, 3).

    This is the single source of truth for feature extraction used by both
    training and inference. Import it from here in both places.
    """
    velocity     = np.diff(processed, axis=0, prepend=processed[:1])   # (T,17,3)
    acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])    # (T,17,3)

    angles  = compute_joint_angles(processed)[:, :, np.newaxis]        # (T,17,1)
    ang_vel = np.diff(angles, axis=0, prepend=angles[:1])              # (T,17,1)

    bone = compute_bone_lengths_normalized(processed)                   # (T,16)
    bone = np.concatenate([bone, bone[:, -1:]], axis=-1)[:, :, np.newaxis]  # (T,17,1)

    features = np.concatenate(
        [processed, velocity, acceleration, angles, ang_vel, bone], axis=-1
    )                                                                   # (T,17,12)
    return np.transpose(features, (2, 0, 1)).copy().astype(np.float32) # (12,T,17)


class UIPRMDPreprocessor:

    _instance : 'UIPRMDPreprocessor | None ' = None

    VICON_39_JOINT_NAMES = cfg.VICON_39_JOINT_NAMES

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

        self.seq_length = seq_length or cfg.SEQUENCE_LENGTH
        self.keypoint_dim = keypoint_dim or cfg.UIPRMD_KEYPOINT_DIM
    
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

        aligned = aligned.astype(np.float32)

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
        3) ROM normalization — rescale each joint to [0,1] within its own
           min/max so "full squat depth" = 1.0 for everyone regardless of
           height, coordinate system, or camera distance.
        4) Z-score normalization
        """
        seq = self.reshape_to_joints(keypoints)
        seq = self.pad_or_truncate(seq)
        seq = rom_normalize(seq)
        seq = self.normalize(seq)
        return seq.astype(np.float32)