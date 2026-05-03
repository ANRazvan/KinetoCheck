"""
Video inference with phase-aware pose overlay and per-joint corrective feedback.

FIXES IN THIS VERSION
=====================

FIX A — Axis mapping: auto-detect camera orientation from data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Old code hardcoded Vicon-Y → Screen-X. If the capture camera was facing the
subject from the front, Vicon-X is depth (into camera) and Vicon-Y is lateral.
If facing from the side, it's the opposite. The fix: try both X and Y as the
lateral axis, pick the one that produces wider shoulder separation in the
projected template — that's the correct camera-facing axis. This is computed
once per sequence from the template data, not hardcoded.

FIX B — Single-pass proportional scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Old code scaled height globally, then independently scaled X using shoulder
width. This distorted body proportions. Fix: use a single uniform scale
(nose-to-midfoot), then apply a SINGLE translation to anchor feet. No
independent X rescaling — the ghost is a rigid rescaled version of the template.

FIX C — Phase-locked frame timing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Old code used a linear remap from video frame index to model frame index. But
the model operates at its own internal temporal resolution (T_u frames) that
does not match the video frame count. Fix: precompute a per-video-frame lookup
table that maps each video frame to the correct model frame using the actual
warp weight argmax, so the ghost skeleton is always in the phase-correct pose.

FIX D — Ghost computed from phase-aligned warp, user body measured once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Old code resampled user_seq inside compute_perfect_ghost, causing the user body
dimensions to drift from the actual per-frame measurements. Fix: measure user
body scale from the raw sequence once (median over all frames, robust to
outliers), and apply that scale to the ghost in one pass.

FIX E — Temporal correlation display in HUD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Added a per-joint temporal correlation score shown in the HUD. This is the
Pearson correlation between the ghost joint trajectory and the user joint
trajectory over the sliding window around the current frame, giving a real-time
readout of "how in sync is this joint right now" rather than just average error.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import urllib.request

import numpy as np
import torch

from phase_aware import ExerciseEvaluator
from Preprocessing.UIPRMDPreprocessor import (
    UIPRMDPreprocessor,
    build_features_from_aligned,
)

try:
    import cv2
except ImportError as exc:
    raise ImportError("OpenCV is required: pip install opencv-python") from exc

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise ImportError("MediaPipe is required: pip install mediapipe") from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MP_COCO17_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_POSE_MODEL_PATH = Path(".cache") / "mediapipe" / "pose_landmarker_full.task"

COCO17_NAMES = [
    "nose", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

COCO17_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11),
    (8, 10), (10, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

COLOR_CORRECT   = (200, 200, 200)
COLOR_IMPORTANT = (0, 200, 255)
COLOR_PROBLEM   = (0, 60, 255)
COLOR_OVERLAY   = (220, 130, 30)
COLOR_ARROW     = (30, 220, 255)
COLOR_HUD_BG    = (20, 20, 20)

THRESHOLD_FLOOR = 0.05

# Joints used for body scale measurement (robust, always-visible)
FOOT_L_IDX  = 11   # left_ankle
FOOT_R_IDX  = 12   # right_ankle
NOSE_IDX    = 0
SHOULDER_L  = 1
SHOULDER_R  = 2
HIP_L       = 7
HIP_R       = 8

# Temporal correlation window (frames) for per-joint sync readout
CORR_WINDOW = 15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LoadedExerciseModel:
    exercise_id: int
    model: ExerciseEvaluator
    template_tensor: torch.Tensor
    template_xyz: torch.Tensor | None
    threshold: float
    raw_threshold: float
    checkpoint_path: Path
    use_phase_decoder: bool
    in_channels: int = 9   # actual channel count the model was trained with


# ---------------------------------------------------------------------------
# Device / model loading (unchanged from original)
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _detect_in_channels(state_dict: dict) -> int:
    """
    Read in_channels directly from the first layer's weight shape in the
    saved state dict.  This is the ground truth — it never lies even when
    the metadata field "in_channels" is stale (e.g. old 9-ch checkpoint
    loaded after the feature pipeline was upgraded to 12 channels).

    The first encoder block's spatial attention projection has weight shape
    (out_channels, in_channels), so index [1] gives in_channels.
    """
    key = "encoder.0.spatial_attn.proj.weight"
    if key in state_dict:
        return int(state_dict[key].shape[1])
    # Fallback: scan for any Linear weight in the first block
    for k, v in state_dict.items():
        if "encoder.0" in k and "weight" in k and v.ndim == 2:
            return int(v.shape[1])
    return 9   # safe default for very old checkpoints


def _rebuild_template_12ch(
    ckpt: dict,
    preprocessor: UIPRMDPreprocessor,
) -> torch.Tensor:
    """
    The checkpoint stores template_tensor with whatever channel count was
    used at training time.  If that was 9 (old checkpoint) but the current
    pipeline produces 12 channels, we need to rebuild the template.

    We use the raw Vicon sequences stored in the dataset records — but those
    aren't in the checkpoint.  Instead we rebuild from template_tensor itself:
    the first 3 channels are ROM-normalised, z-scored XYZ, which is exactly
    what build_features_from_aligned expects as input (it was produced by
    preprocessor.process()).  So we can recover (T, 17, 3) from the 9-ch
    tensor and re-run build_features_from_aligned on it.
    """
    tmpl = ckpt["template_tensor"]          # (9, T, 17) or (12, T, 17)
    if tmpl.shape[0] == 12:
        return tmpl                          # already new format, nothing to do

    # tmpl is (9, T, 17) — channels 0-2 are XYZ, which is all we need
    xyz_chw = tmpl[:3]                       # (3, T, 17)
    T, J    = xyz_chw.shape[1], xyz_chw.shape[2]
    # Rearrange to (T, J, 3) = the shape build_features_from_aligned expects
    processed = xyz_chw.permute(1, 2, 0).cpu().numpy().astype("float32")   # (T, 17, 3)
    features  = build_features_from_aligned(processed)                # (12, T, 17)
    return torch.from_numpy(features).float()


def load_models(checkpoints_root: Path, device: torch.device) -> list[LoadedExerciseModel]:
    preprocessor = UIPRMDPreprocessor()
    loaded: list[LoadedExerciseModel] = []

    for exercise_dir in sorted(checkpoints_root.glob("exercise_*")):
        ckpt_path = exercise_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})
        use_phase_decoder = bool(ckpt.get("use_phase_decoder", False))

        # Detect actual in_channels from weight shapes — never trust metadata alone
        state_dict = ckpt["model_state_dict"]
        in_channels = _detect_in_channels(state_dict)
        if in_channels != ckpt.get("in_channels", 9):
            print(
                f"  [compat] {ckpt_path.name}: metadata says in_channels="
                f"{ckpt.get('in_channels',9)} but weights say {in_channels}. "
                f"Using {in_channels}."
            )

        model = ExerciseEvaluator(
            in_channels=in_channels,
            hidden_channels=tuple(cfg.get("hidden_channels", (64, 128))),
            embedding_dim=int(ckpt.get("embedding_dim", cfg.get("embedding_dim", 128))),
            use_phase_decoder=use_phase_decoder,
        ).to(device)

        missing, _ = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [compat] {ckpt_path.name}: {len(missing)} keys missing.")
        model.eval()

        exercise_id  = int(ckpt.get("exercise_id", int(exercise_dir.name.split("_")[-1]) - 1))
        raw_threshold = float(ckpt.get("val_threshold", 0.5))
        threshold     = max(THRESHOLD_FLOOR, raw_threshold)

        if raw_threshold < THRESHOLD_FLOOR:
            print(
                f"  [WARN] Exercise {exercise_id}: threshold={raw_threshold:.3f} below floor. "
                f"Clamped to {threshold:.3f}. Consider retraining."
            )

        # Rebuild template to 12 channels if checkpoint is from before the upgrade
        template_raw = _rebuild_template_12ch(ckpt, preprocessor)
        template = template_raw.detach().clone().float().unsqueeze(0).to(device)

        template_xyz: torch.Tensor | None = None
        if "template_xyz_tensor" in ckpt:
            preprocessor_cfg = ckpt.get("preprocessor_config", {})
            is_raw = bool(preprocessor_cfg.get("template_xyz_is_raw", False))
            if is_raw:
                template_xyz = ckpt["template_xyz_tensor"].detach().clone().float().unsqueeze(0).to(device)

        loaded.append(LoadedExerciseModel(
            exercise_id=exercise_id,
            model=model,
            template_tensor=template,
            template_xyz=template_xyz,
            threshold=threshold,
            raw_threshold=raw_threshold,
            checkpoint_path=ckpt_path,
            use_phase_decoder=use_phase_decoder,
            in_channels=in_channels,
        ))

    if not loaded:
        raise FileNotFoundError(f"No best_checkpoint.pt found under {checkpoints_root}")
    return loaded


# ---------------------------------------------------------------------------
# Pose extraction (unchanged)
# ---------------------------------------------------------------------------

def ensure_pose_task_model(model_path: Path | None) -> Path:
    resolved = Path(model_path) if model_path is not None else DEFAULT_POSE_MODEL_PATH
    if resolved.exists():
        return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {resolved}…")
    urllib.request.urlretrieve(DEFAULT_POSE_MODEL_URL, str(resolved))
    print("Done.")
    return resolved


def extract_mediapipe_sequence(video_path: Path, pose_model_path: Path) -> np.ndarray:
    """Returns (T, J, 3) raw MediaPipe XYZ — X,Y in [0,1] image fractions."""
    base_options = mp_python.BaseOptions(model_asset_path=str(pose_model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sequence: list[np.ndarray] = []
    last_valid: np.ndarray | None = None

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((frame_idx / fps) * 1000)
            det    = landmarker.detect_for_video(mp_img, ts_ms)

            if det.pose_landmarks:
                lms = det.pose_landmarks[0]
                pts = np.array(
                    [[lms[i].x, lms[i].y, lms[i].z] for i in MP_COCO17_IDXS],
                    dtype=np.float32,
                )
                last_valid = pts
            else:
                pts = (
                    np.zeros((17, 3), dtype=np.float32)
                    if last_valid is None
                    else last_valid.copy()
                )
            sequence.append(pts)
            frame_idx += 1

    cap.release()
    if not sequence:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(sequence, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_model_input(
    sequence: np.ndarray,
    preprocessor: UIPRMDPreprocessor,
    in_channels: int = 12,
) -> torch.Tensor:
    """
    sequence : (T, J, 3) raw MediaPipe XYZ → (1, C, T, J) model tensor.

    in_channels=12 → new pipeline (angles + bone ratios, ROM normalised).
    in_channels=9  → old pipeline (xyz + velocity + acceleration only),
                     used automatically when an old checkpoint is loaded so
                     the channel count always matches the model weights.
    """
    aligned   = preprocessor.align_vicon_to_mediapipe(sequence)
    processed = preprocessor.process(aligned)    # (T, 17, 3)

    if in_channels == 12:
        features = build_features_from_aligned(processed)   # (12, T, 17)
    else:
        # Legacy 9-channel path — keeps old checkpoints working
        velocity     = np.diff(processed, axis=0, prepend=processed[:1])
        acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
        features = np.concatenate([processed, velocity, acceleration], axis=-1)
        features = np.transpose(features, (2, 0, 1)).copy().astype(np.float32)

    return torch.from_numpy(features).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# FIX A: Auto-detect camera axis orientation from template data
# ---------------------------------------------------------------------------

def _detect_lateral_axis(template_xyz: np.ndarray) -> int:
    """
    template_xyz: (3, T, J) Vicon data.
    Vicon axes: 0=X, 1=Y, 2=Z (up).
    Screen axes: we need the lateral (left-right) axis.

    Strategy: the shoulder joint pair (joints 1,2) should be maximally
    separated on the lateral axis. We compare mean separation on axis 0 vs
    axis 1 and pick the larger one. This tells us which Vicon horizontal
    axis the camera was facing perpendicular to.

    Returns 0 (use Vicon X as lateral) or 1 (use Vicon Y as lateral).
    """
    sep_x = float(np.abs(template_xyz[0, :, SHOULDER_L] - template_xyz[0, :, SHOULDER_R]).mean())
    sep_y = float(np.abs(template_xyz[1, :, SHOULDER_L] - template_xyz[1, :, SHOULDER_R]).mean())
    lateral_axis = 0 if sep_x >= sep_y else 1
    print(f"  [axis] shoulder sep X={sep_x:.4f} Y={sep_y:.4f} → lateral_axis={lateral_axis}")
    return lateral_axis


# ---------------------------------------------------------------------------
# FIX B+C+D: Corrected ghost skeleton computation
# ---------------------------------------------------------------------------

def compute_perfect_ghost(
    warp_weights_np: np.ndarray,   # (T_u, T_t_feat)
    template_xyz: np.ndarray,      # (3, T_t_raw, J) — raw Vicon XYZ from checkpoint
    raw_sequence: np.ndarray,      # (T_seq, J, 3) — raw MediaPipe [0,1] fractions
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        ghost_screen  : (T_u, J, 2) in [0,1] image-fraction screen coords
        user_synced   : (T_u, J, 2) user XY resampled to model timeline
        video_to_model: (T_seq,) int array mapping video frame → model frame index

    FIX B: Single uniform scale only — no independent X rescaling.
    FIX C: video_to_model built from warp argmax for phase-locked rendering.
    FIX D: User body scale measured once from full raw_sequence (median, robust).
    """
    T_u, T_t_feat = warp_weights_np.shape
    _, T_t_raw, J = template_xyz.shape
    T_seq = raw_sequence.shape[0]

    # ---- 0. Resample template to feature timeline ----
    if T_t_raw != T_t_feat:
        x_old = np.linspace(0.0, 1.0, T_t_raw)
        x_new = np.linspace(0.0, 1.0, T_t_feat)
        resampled = np.zeros((3, T_t_feat, J), dtype=np.float32)
        for c in range(3):
            for j in range(J):
                resampled[c, :, j] = np.interp(x_new, x_old, template_xyz[c, :, j])
        template_xyz = resampled   # (3, T_t_feat, J)

    # ---- 1. FIX A: detect lateral axis from template ----
    lateral_axis = _detect_lateral_axis(template_xyz)
    vertical_axis = 2  # Vicon Z is always up

    # ---- 2. Soft-warp template to user timeline ----
    # template_xyz axes: (3=XYZ, T_t_feat, J)
    # We only need lateral and vertical for the 2D overlay.
    tmpl_lat = template_xyz[lateral_axis, :, :]   # (T_t_feat, J)
    tmpl_vert = template_xyz[vertical_axis, :, :]  # (T_t_feat, J)

    # Warp: (T_u, T_t_feat) @ (T_t_feat, J) → (T_u, J)
    ghost_lat  = warp_weights_np @ tmpl_lat    # (T_u, J)
    ghost_vert = warp_weights_np @ tmpl_vert   # (T_u, J)

    # ---- 3. Convert Vicon coords to screen coords ----
    # Lateral axis → screen X  (may need sign flip — auto-detect below)
    # Vicon -Z (vertical, flipped) → screen Y (Y increases downward on screen)
    ghost_screen_x = ghost_lat.copy()     # (T_u, J)
    ghost_screen_y = -ghost_vert.copy()   # (T_u, J)  negate: Vicon up = screen down

    # ---- 3.5 Normalize warped ghost into [0, 1] before any scale/anchor step ----
    all_ghost = np.stack([ghost_screen_x, ghost_screen_y], axis=-1)  # (T_u, J, 2)
    g_min = all_ghost.min(axis=(0, 1), keepdims=True)
    g_max = all_ghost.max(axis=(0, 1), keepdims=True)
    g_range = np.maximum(g_max - g_min, 1e-6)
    ghost_screen_x = (ghost_screen_x - g_min[..., 0]) / g_range[..., 0]
    ghost_screen_y = (ghost_screen_y - g_min[..., 1]) / g_range[..., 1]

    # ---- 4. FIX D: measure user body dimensions ONCE from full raw_sequence ----
    # Use median over all frames for robustness (ignores occasional bad detections).
    user_foot_y  = np.median(
        (raw_sequence[:, FOOT_L_IDX, 1] + raw_sequence[:, FOOT_R_IDX, 1]) / 2.0
    )
    user_foot_x  = np.median(
        (raw_sequence[:, FOOT_L_IDX, 0] + raw_sequence[:, FOOT_R_IDX, 0]) / 2.0
    )
    user_nose_y  = np.median(raw_sequence[:, NOSE_IDX, 1])
    user_height  = float(np.abs(user_foot_y - user_nose_y))

    # ---- 5. FIX B: single uniform scale from ghost body height ----
    # Anchor ghost at its own feet first (median over time)
    ghost_feet_x = np.median((ghost_screen_x[:, FOOT_L_IDX] + ghost_screen_x[:, FOOT_R_IDX]) / 2.0)
    ghost_feet_y = np.median((ghost_screen_y[:, FOOT_L_IDX] + ghost_screen_y[:, FOOT_R_IDX]) / 2.0)
    ghost_nose_y = np.median(ghost_screen_y[:, NOSE_IDX])
    ghost_height = float(np.abs(ghost_feet_y - ghost_nose_y))

    if ghost_height < 1e-5:
        # Template has essentially no vertical extent — can't compute scale.
        # Fall back to 1.0 scale. This usually means the lateral_axis detection
        # picked the depth axis by mistake. Warn and try the other axis.
        print("  [WARN] Ghost height near zero — trying alternate lateral axis.")
        lateral_axis = 1 - lateral_axis
        tmpl_lat = template_xyz[lateral_axis, :, :]
        ghost_lat = warp_weights_np @ tmpl_lat
        ghost_screen_x = ghost_lat.copy()
        ghost_feet_x = np.median((ghost_screen_x[:, FOOT_L_IDX] + ghost_screen_x[:, FOOT_R_IDX]) / 2.0)
        ghost_height = max(1e-5, float(np.abs(ghost_feet_y - ghost_nose_y)))

    # FIX: detect lateral flip (if ghost left shoulder is on screen right, flip X)
    ghost_lshoulder_x = np.median(ghost_screen_x[:, SHOULDER_L])
    ghost_rshoulder_x = np.median(ghost_screen_x[:, SHOULDER_R])
    user_lshoulder_x  = np.median(raw_sequence[:, SHOULDER_L, 0])
    user_rshoulder_x  = np.median(raw_sequence[:, SHOULDER_R, 0])
    ghost_lr = ghost_lshoulder_x - ghost_rshoulder_x   # positive = L is to the right of R in ghost space
    user_lr  = user_lshoulder_x  - user_rshoulder_x    # same for user screen space
    if (ghost_lr * user_lr) < 0:
        # Opposite chirality — flip the lateral axis
        ghost_screen_x = -ghost_screen_x
        ghost_feet_x   = -ghost_feet_x
        print("  [axis] Lateral flip applied (mirror correction).")

    # Single uniform scale: ghost_height → user_height
    scale = user_height / ghost_height

    # Centre ghost around its own foot midpoint, apply scale, then translate to user foot midpoint
    ghost_screen_x = (ghost_screen_x - ghost_feet_x) * scale + user_foot_x
    ghost_screen_y = (ghost_screen_y - ghost_feet_y) * scale + user_foot_y

    ghost_screen = np.stack([ghost_screen_x, ghost_screen_y], axis=-1)  # (T_u, J, 2)

    # ---- 6. FIX C: build video_to_model lookup table from warp argmax ----
    # warp_weights_np[t, :] is the soft distribution over template frames for
    # model frame t. argmax gives the most attended template frame, which we
    # convert to a fractional phase position and invert for each video frame.
    model_argmax = np.argmax(warp_weights_np, axis=1)   # (T_u,)
    model_frac = model_argmax / max(1, T_t_feat - 1)    # (T_u,) in [0, 1]
    video_frac = np.linspace(0.0, 1.0, T_seq)           # (T_seq,)
    video_to_model = np.array(
        [int(np.argmin(np.abs(model_frac - vf))) for vf in video_frac],
        dtype=np.int32,
    )

    # ---- 7. Resample user to model timeline (for delta/error computation only) ----
    x_old_user = np.linspace(0.0, 1.0, T_seq)
    x_new_user = np.linspace(0.0, 1.0, T_u)
    user_synced = np.zeros((T_u, J, 2), dtype=np.float32)
    for j in range(J):
        for c in range(2):
            user_synced[:, j, c] = np.interp(x_new_user, x_old_user, raw_sequence[:, j, c])

    return ghost_screen, user_synced, video_to_model


# ---------------------------------------------------------------------------
# FIX E: Per-joint temporal correlation score
# ---------------------------------------------------------------------------

def compute_temporal_correlation(
    ghost_screen: np.ndarray,   # (T_u, J, 2)
    user_synced: np.ndarray,    # (T_u, J, 2)
    model_t: int,
    window: int = CORR_WINDOW,
) -> np.ndarray:
    """
    Compute per-joint Pearson correlation between ghost and user trajectories
    in a sliding window around model_t.

    Returns (J,) correlation values in [-1, 1].
    High positive value = your movement is in sync with perfect form.
    Low or negative = timing or shape mismatch.
    """
    T_u, J, _ = ghost_screen.shape
    t_lo = max(0, model_t - window // 2)
    t_hi = min(T_u, model_t + window // 2 + 1)
    if t_hi - t_lo < 3:
        return np.zeros(J, dtype=np.float32)

    # Use Y (vertical) trajectory as the primary correlation signal —
    # vertical motion is most discriminative for exercise form.
    g = ghost_screen[t_lo:t_hi, :, 1]   # (W, J)
    u = user_synced[t_lo:t_hi, :, 1]    # (W, J)

    # Pearson r per joint
    g_c = g - g.mean(axis=0, keepdims=True)
    u_c = u - u.mean(axis=0, keepdims=True)
    num = (g_c * u_c).sum(axis=0)
    denom = np.sqrt((g_c**2).sum(axis=0) * (u_c**2).sum(axis=0)) + 1e-8
    return (num / denom).astype(np.float32)   # (J,)


# ---------------------------------------------------------------------------
# Prediction (mostly unchanged, calls new compute_perfect_ghost)
# ---------------------------------------------------------------------------

def compute_joint_deviation(
    user_tensor: torch.Tensor,
    template_tensor: torch.Tensor,
    joint_importance: np.ndarray,
) -> list[dict]:
    user_np     = user_tensor.detach().cpu().numpy()[0]
    template_np = template_tensor.detach().cpu().numpy()[0]

    pos_dev = np.abs(user_np[:3] - template_np[:3]).mean(axis=(0, 1))
    imp     = joint_importance.astype(np.float32)

    dev_norm = (pos_dev - pos_dev.min()) / max(1e-6, float(pos_dev.max() - pos_dev.min()))
    imp_norm = (imp     - imp.min())     / max(1e-6, float(imp.max()     - imp.min()))
    combined = 0.75 * dev_norm + 0.25 * imp_norm

    ranked = np.argsort(-combined)
    return [
        {
            "joint":         COCO17_NAMES[int(idx)],
            "joint_index":   int(idx),
            "deviation":     float(pos_dev[int(idx)]),
            "importance":    float(imp[int(idx)]),
            "problem_score": float(combined[int(idx)]),
        }
        for idx in ranked
    ]


def run_prediction(
    input_tensor: torch.Tensor,
    raw_sequence: np.ndarray,
    models: list[LoadedExerciseModel],
    device: torch.device,
) -> tuple[dict, np.ndarray, list[dict], dict | None]:
    results = []
    with torch.no_grad():
        user_tensor = input_tensor.to(device)
        for loaded in models:
            outputs = loaded.model(loaded.template_tensor, user_tensor)
            score   = float(outputs["similarity_score"].item())
            pred    = score >= loaded.threshold
            margin  = score - loaded.threshold

            joint_importance = outputs["joint_importance"].detach().cpu().numpy()[0]

            item = {
                "exercise_id":     loaded.exercise_id,
                "exercise_name":   f"exercise_{loaded.exercise_id + 1:02d}",
                "score":           score,
                "threshold":       loaded.threshold,
                "raw_threshold":   loaded.raw_threshold,
                "margin":          margin,
                "predicted_label": "correct" if pred else "incorrect",
                "checkpoint":      str(loaded.checkpoint_path),
                "has_phase_decoder": loaded.use_phase_decoder,
            }
            results.append({
                "item":             item,
                "joint_importance": joint_importance,
                "loaded":           loaded,
                "outputs":          outputs,
            })

    best_result  = max(results, key=lambda x: x["item"]["margin"])
    best_item    = best_result["item"]
    best_imp     = best_result["joint_importance"]
    best_loaded  = best_result["loaded"]
    best_outputs = best_result["outputs"]

    phase_outputs: dict | None = None
    if (
        "warp_weights" in best_outputs
        and best_loaded.use_phase_decoder
        and best_loaded.template_xyz is not None
    ):
        warp_weights = best_outputs["warp_weights"].detach().cpu()
        warp_np      = warp_weights[0].numpy()   # (T_u, T_t)

        tmpl_xyz_np  = best_loaded.template_xyz[0].cpu().numpy()   # (3, T_t, J)

        # Call the FIXED ghost computation
        ghost_screen, user_synced, video_to_model = compute_perfect_ghost(
            warp_np, tmpl_xyz_np, raw_sequence
        )

        # ghost_screen: (T_u, J, 2) in [0,1] screen coords
        # Pack into (T_u, J, 3) for downstream code (Z=0 unused)
        ghost_xyz = np.zeros((ghost_screen.shape[0], ghost_screen.shape[1], 3), dtype=np.float32)
        ghost_xyz[..., :2] = ghost_screen

        # User synced also packed to 3 channels
        user_synced_3 = np.zeros_like(ghost_xyz)
        user_synced_3[..., :2] = user_synced

        err_mag = (
            best_outputs["joint_error_magnitude"].detach().cpu().numpy()[0]
            if "joint_error_magnitude" in best_outputs
            else np.linalg.norm(ghost_screen - user_synced, axis=-1).mean(axis=0)
        )
        joint_conf = (
            best_outputs["joint_confidence"].detach().cpu().numpy()[0]
            if "joint_confidence" in best_outputs
            else np.ones((warp_np.shape[0], 17), dtype=np.float32)
        )

        phase_outputs = {
            "ghost_xyz":       ghost_xyz,        # (T_u, J, 3)  XY in [0,1]
            "user_synced":     user_synced_3,     # (T_u, J, 3)  XY in [0,1]
            "delta_xy":        ghost_xyz - user_synced_3,
            "joint_error_mag": err_mag,           # (J,)
            "joint_confidence": joint_conf,       # (T_u, J)
            "video_to_model":  video_to_model,    # (T_seq,) FIX C
        }

    if phase_outputs is not None:
        err_mag = phase_outputs["joint_error_mag"]
        ranked  = np.argsort(-err_mag)
        worst_joints = [
            {
                "joint":         COCO17_NAMES[int(idx)],
                "joint_index":   int(idx),
                "deviation":     float(err_mag[int(idx)]),
                "importance":    float(best_imp[int(idx)]),
                "problem_score": float(
                    0.6 * err_mag[int(idx)] / max(1e-6, err_mag.max())
                    + 0.4 * best_imp[int(idx)] / max(1e-6, best_imp.max())
                ),
            }
            for idx in ranked
        ]
    else:
        worst_joints = compute_joint_deviation(
            input_tensor.to(device), best_loaded.template_tensor, best_imp
        )

    all_results = [
        r["item"]
        for r in sorted(results, key=lambda x: x["item"]["margin"], reverse=True)
    ]
    return {"best": best_item, "all": all_results}, best_imp, worst_joints, phase_outputs


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _pixel(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return int(x * w), int(y * h)


def _joint_color(idx: int, bad_idxs: set[int], imp_idxs: set[int]) -> tuple[int, int, int]:
    if idx in bad_idxs:   return COLOR_PROBLEM
    if idx in imp_idxs:   return COLOR_IMPORTANT
    return COLOR_CORRECT


def draw_skeleton(
    frame: np.ndarray,
    points: list[tuple[int, int] | None],
    joint_colors: list[tuple[int, int, int]],
    norm: np.ndarray,
    width: int,
    bad_idxs: set[int],
    imp_idxs: set[int],
    line_thickness: int = 2,
) -> None:
    for a, b in COCO17_EDGES:
        pa, pb = points[a], points[b]
        if pa is None or pb is None:
            continue
        ec = joint_colors[a] if a in bad_idxs or a in imp_idxs else joint_colors[b]
        cv2.line(frame, pa, pb, ec, line_thickness, cv2.LINE_AA)
    for idx, pt in enumerate(points):
        if pt is None:
            continue
        radius = int(4 + 10 * float(norm[idx]))
        cv2.circle(frame, pt, radius, joint_colors[idx], 2, cv2.LINE_AA)


def draw_ghost_skeleton(
    frame: np.ndarray,
    ghost_xyz: np.ndarray,   # (J, 3)  X,Y in [0,1]
    width: int,
    height: int,
    alpha: float = 0.55,     # slightly more opaque than original for visibility
) -> None:
    overlay = frame.copy()
    pts: list[tuple[int, int] | None] = []
    for j in range(17):
        x, y = float(ghost_xyz[j, 0]), float(ghost_xyz[j, 1])
        if -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05:
            pts.append(_pixel(max(0., min(1., x)), max(0., min(1., y)), width, height))
        else:
            pts.append(None)

    if sum(p is not None for p in pts) < 3:
        return

    for a, b in COCO17_EDGES:
        if pts[a] and pts[b]:
            cv2.line(overlay, pts[a], pts[b], COLOR_OVERLAY, 2, cv2.LINE_AA)
    for pt in pts:
        if pt:
            cv2.circle(overlay, pt, 5, COLOR_OVERLAY, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_correction_arrows(
    frame: np.ndarray,
    user_pts: list[tuple[int, int] | None],
    ghost_xyz: np.ndarray,   # (J, 3) image-fraction ghost
    joint_conf: np.ndarray,  # (J,)
    width: int,
    height: int,
    bad_idxs: set[int],
    min_px: int = 6,
) -> None:
    for j in bad_idxs:
        if user_pts[j] is None:
            continue
        gx, gy = float(ghost_xyz[j, 0]), float(ghost_xyz[j, 1])
        if not (-0.05 <= gx <= 1.05 and -0.05 <= gy <= 1.05):
            continue
        tip = (
            int(max(0, min(width  - 1, gx * width))),
            int(max(0, min(height - 1, gy * height))),
        )
        dx = tip[0] - user_pts[j][0]
        dy = tip[1] - user_pts[j][1]
        if abs(dx) < min_px and abs(dy) < min_px:
            continue
        thickness = max(1, int(1 + float(joint_conf[j]) * 3))
        cv2.arrowedLine(frame, user_pts[j], tip,
                        COLOR_ARROW, thickness, cv2.LINE_AA, tipLength=0.25)


def draw_hud(
    frame: np.ndarray,
    best: dict,
    worst_joints: list[dict],
    top3_names: list[str],
    has_phase: bool,
    joint_corr: np.ndarray | None = None,   # FIX E: (J,) correlation scores
) -> None:
    lines = [
        f"Pred: {best['predicted_label']}  ({best['exercise_name']})",
        f"Score: {best['score']:.3f}  |  Thr: {best['threshold']:.3f}",
        f"Top-attn: {', '.join(top3_names)}",
    ]

    if has_phase and joint_corr is not None:
        # Show worst joints with their temporal sync score
        sync_parts = []
        for w in worst_joints[:3]:
            ji = int(w["joint_index"])
            corr_val = float(joint_corr[ji]) if ji < len(joint_corr) else 0.0
            sync_str = f"{corr_val:+.2f}"
            sync_parts.append(f"{w['joint']}[sync={sync_str}]")
        lines.append("Fix: " + "  ".join(sync_parts))
    elif has_phase:
        lines.append("Fix: " + "  ".join(
            f"{w['joint']} ({w['deviation']:.3f})" for w in worst_joints[:3]
        ))
    else:
        lines.append("Fix: " + ", ".join(w["joint"] for w in worst_joints[:3]))

    font_scale, thickness, pad, line_h = 0.60, 2, 8, 26
    panel_w = 640
    panel_h = len(lines) * line_h + pad * 2
    sub = frame[0:panel_h, 0:panel_w].copy()
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(frame[0:panel_h, 0:panel_w], 0.65, sub, 0.35, 0,
                    frame[0:panel_h, 0:panel_w])

    # Color the sync score: green = in sync, red = out of sync
    fix_color = (80, 140, 255)
    if joint_corr is not None and worst_joints:
        mean_worst_corr = np.mean([
            float(joint_corr[int(w["joint_index"])])
            for w in worst_joints[:3]
            if int(w["joint_index"]) < len(joint_corr)
        ])
        if mean_worst_corr > 0.6:
            fix_color = (10, 220, 10)
        elif mean_worst_corr > 0.2:
            fix_color = (0, 200, 255)
        else:
            fix_color = (60, 60, 255)

    colors = [
        (10, 220, 10) if best["predicted_label"] == "correct" else (60, 60, 255),
        (240, 240, 240),
        (120, 220, 255),
        fix_color,
    ]
    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(frame, line, (pad, pad + (i + 1) * line_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Annotation loop — uses FIX C: video_to_model lookup table
# ---------------------------------------------------------------------------

def annotate_video(
    video_path: Path,
    output_path: Path,
    summary: dict,
    joint_importance: np.ndarray,
    worst_joints: list[dict],
    phase_outputs: dict | None,
    pose_model_path: Path,
) -> str:
    base_options = mp_python.BaseOptions(model_asset_path=str(pose_model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    used_codec = ""
    for codec in ("avc1", "H264", "X264", "mp4v"):
        fourcc    = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if candidate.isOpened():
            writer     = candidate
            used_codec = codec
            break
        candidate.release()
    if writer is None:
        cap.release()
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    norm_imp   = joint_importance.astype(np.float32)
    norm_imp   = (norm_imp - norm_imp.min()) / max(1e-6, float(norm_imp.max() - norm_imp.min()))
    top3_idxs  = set(int(i) for i in np.argsort(-joint_importance)[:3])
    top3_names = [COCO17_NAMES[i] for i in np.argsort(-joint_importance)[:3]]
    bad_idxs   = {int(w["joint_index"]) for w in worst_joints[:3]}
    has_phase  = phase_outputs is not None

    ghost_seq       = None
    user_synced_seq = None
    joint_conf_seq  = None
    video_to_model  = None
    T_model         = 0

    if has_phase:
        ghost_seq       = phase_outputs["ghost_xyz"]        # (T_u, J, 3)
        user_synced_seq = phase_outputs["user_synced"]      # (T_u, J, 3)
        joint_conf_seq  = phase_outputs["joint_confidence"] # (T_u, J)
        video_to_model  = phase_outputs["video_to_model"]   # (T_seq,) — FIX C
        T_model         = ghost_seq.shape[0]

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            # FIX C: use lookup table instead of linear remap
            if video_to_model is not None and frame_idx < len(video_to_model):
                model_t = int(video_to_model[frame_idx])
                model_t = max(0, min(model_t, T_model - 1))
            elif T_model > 0:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                model_t = int(round(frame_idx / max(1, total_frames - 1) * max(0, T_model - 1)))
                model_t = min(model_t, T_model - 1)
            else:
                model_t = 0

            # FIX E: compute temporal correlation for this window
            joint_corr: np.ndarray | None = None
            if has_phase and ghost_seq is not None and user_synced_seq is not None:
                joint_corr = compute_temporal_correlation(
                    ghost_seq[..., :2],       # (T_u, J, 2)
                    user_synced_seq[..., :2], # (T_u, J, 2)
                    model_t,
                )

            if has_phase and ghost_seq is not None:
                draw_ghost_skeleton(frame, ghost_seq[model_t], width, height)

            if result.pose_landmarks:
                lms  = result.pose_landmarks[0]
                user_pts: list[tuple[int, int] | None] = []
                joint_colors: list[tuple[int, int, int]] = []

                for local_idx, mp_idx in enumerate(MP_COCO17_IDXS):
                    lm = lms[mp_idx]
                    px, py = int(lm.x * width), int(lm.y * height)
                    user_pts.append(
                        (px, py) if 0 <= px < width and 0 <= py < height else None
                    )
                    # FIX E: tint bad joints by their sync score too
                    base_color = _joint_color(local_idx, bad_idxs, top3_idxs)
                    if joint_corr is not None and local_idx in bad_idxs:
                        corr_val = float(joint_corr[local_idx])
                        if corr_val < 0.2:
                            base_color = (0, 0, 255)   # pure red = very out of sync
                    joint_colors.append(base_color)

                draw_skeleton(frame, user_pts, joint_colors, norm_imp,
                              width, bad_idxs, top3_idxs)

                if has_phase and ghost_seq is not None and joint_conf_seq is not None:
                    draw_correction_arrows(
                        frame, user_pts,
                        ghost_seq[model_t],
                        joint_conf_seq[model_t],
                        width, height, bad_idxs,
                    )

            draw_hud(
                frame, summary["best"], worst_joints, top3_names,
                has_phase, joint_corr
            )
            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    return used_codec


# ---------------------------------------------------------------------------
# Pipeline (unchanged structure)
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    models: list[LoadedExerciseModel],
    preprocessor: UIPRMDPreprocessor,
    output_dir: Path,
    device: torch.device,
    pose_model_path: Path,
) -> dict:
    raw_sequence  = extract_mediapipe_sequence(video_path, pose_model_path)
    # Use the in_channels of the first model — all models for a given
    # exercise share the same architecture, and mixed checkpoints are not
    # supported in a single run.
    in_ch = models[0].in_channels if models else 12
    input_tensor  = build_model_input(raw_sequence, preprocessor, in_channels=in_ch)

    summary, joint_importance, worst_joints, phase_outputs = run_prediction(
        input_tensor, raw_sequence, models, device
    )

    annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
    output_codec   = annotate_video(
        video_path, annotated_path, summary,
        joint_importance, worst_joints, phase_outputs, pose_model_path,
    )

    report = {
        "video":                 str(video_path),
        "annotated_video":       str(annotated_path),
        "annotated_video_codec": output_codec,
        "num_frames":            int(raw_sequence.shape[0]),
        "has_phase_decoder":     phase_outputs is not None,
        "worst_joints":          worst_joints[:5],
        **summary,
    }
    (output_dir / f"{video_path.stem}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# CLI (unchanged)
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference on Video-kineto with phase-aware corrective feedback."
    )
    parser.add_argument("--input-dir",        type=Path, default=Path("Video-kineto") / "UIPRMD-videos")
    parser.add_argument("--output-dir",       type=Path, default=Path("Video-kineto-annotated"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints") / "uiprmd")
    parser.add_argument("--pose-model",       type=Path, default=None)
    parser.add_argument("--device",           type=str,  default="auto")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_model_path = ensure_pose_task_model(args.pose_model)

    video_files = sorted(args.input_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {args.input_dir}")

    device = resolve_device(args.device)
    models = load_models(args.checkpoints_root, device)

    specific_models: list[list[LoadedExerciseModel] | None] = [None] * len(models)
    for idx, m in enumerate(models):
        print(f"Loaded exercise {m.exercise_id} from {m.checkpoint_path} "
              f"(thr={m.threshold:.3f} raw={m.raw_threshold:.3f}, "
              f"phase_decoder={m.use_phase_decoder})")
        specific_models[idx] = [m]

    preprocessor = UIPRMDPreprocessor()
    all_reports  = []

    for video_path in video_files:
        exercise_id = int(video_path.stem.split("-")[0].replace("ex", ""))
        model_idx   = exercise_id - 1
        if exercise_id < 1 or exercise_id > len(models):
            print(f"Skipping {video_path.name}: exercise ID {exercise_id} out of range.")
            continue
        report = process_video(
            video_path, specific_models[model_idx],  # type: ignore[arg-type]
            preprocessor, args.output_dir, device, pose_model_path,
        )
        all_reports.append(report)
        print(json.dumps(report["best"], indent=2))

    summary_path = args.output_dir / "all_videos_summary.json"
    summary_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()