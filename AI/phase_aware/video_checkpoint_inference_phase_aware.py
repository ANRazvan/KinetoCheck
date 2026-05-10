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
import csv

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
# Model cache (global, persists across requests)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, list[LoadedExerciseModel]] = {}


def get_cached_models(checkpoints_root: Path, device: torch.device) -> list[LoadedExerciseModel]:
    """
    Load models once and cache them in memory. Subsequent calls return the cached models.
    
    Args:
        checkpoints_root: Path to the checkpoints directory
        device: torch device to load models onto
        
    Returns:
        List of cached LoadedExerciseModel instances
    """
    cache_key = str(checkpoints_root.resolve())
    
    if cache_key not in _MODEL_CACHE:
        print(f"[Cache] Loading models from {checkpoints_root}...")
        _MODEL_CACHE[cache_key] = load_models(checkpoints_root, device)
        print(f"[Cache] Loaded {len(_MODEL_CACHE[cache_key])} models. Cached for future requests.")
    else:
        print(f"[Cache] Using {len(_MODEL_CACHE[cache_key])} cached models.")
    
    return _MODEL_CACHE[cache_key]


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
    in_channels: int = 9  # actual channel count the model was trained with


# ---------------------------------------------------------------------------
# Device / model loading (unchanged from original)
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(checkpoints_root: Path, device: torch.device) -> list[LoadedExerciseModel]:
    loaded: list[LoadedExerciseModel] = []

    for exercise_dir in sorted(checkpoints_root.glob("exercise_*")):
        ckpt_path = exercise_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})
        use_phase_decoder = bool(ckpt.get("use_phase_decoder", False))

        # Detect actual in_channels from weight shape — checkpoint metadata
        # can be stale (9) even after retraining to 12 channels.
        state_dict  = ckpt["model_state_dict"]
        key         = "encoder.0.spatial_attn.proj.weight"
        in_channels = int(state_dict[key].shape[1]) if key in state_dict else int(ckpt.get("in_channels", 9))

        model = ExerciseEvaluator(
            in_channels=in_channels,
            hidden_channels=tuple(cfg.get("hidden_channels", (64, 128))),
            embedding_dim=int(ckpt.get("embedding_dim", cfg.get("embedding_dim", 128))),
            use_phase_decoder=use_phase_decoder,
        ).to(device)

        missing, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
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

        template = ckpt["template_tensor"].detach().clone().float().unsqueeze(0).to(device)

        if "template_xyz_tensor" in ckpt:
            # Load regardless of is_raw — compute_perfect_ghost normalises
            # whatever coordinate system the XYZ is in.
            template_xyz = ckpt["template_xyz_tensor"].detach().clone().float().unsqueeze(0).to(device)
        else:
            # Old checkpoint without template_xyz_tensor — reconstruct from
            # template_tensor channels 0-2 (XYZ positions).
            # template_tensor shape: (C, T, J)  →  we need (1, 3, T, J)
            tmpl_t = ckpt["template_tensor"].detach().clone().float()
            template_xyz = tmpl_t[:3].unsqueeze(0).to(device)  # (1, 3, T, J)
            print(f"  [compat] {ckpt_path.name}: no template_xyz_tensor, "
                  f"reconstructed from template_tensor XYZ channels.")

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
# Feature building (unchanged)
# ---------------------------------------------------------------------------

def build_model_input(
    sequence: np.ndarray,
    preprocessor: UIPRMDPreprocessor,
    in_channels: int = 12,
) -> torch.Tensor:
    """
    sequence    : (T, J, 3) raw MediaPipe XYZ
    in_channels : must match the loaded model (12 for new checkpoints, 9 for old).
                  Passed in from loaded.in_channels so it always matches.

    Returns (1, in_channels, T, J).
    """
    aligned   = preprocessor.align_vicon_to_mediapipe(sequence)
    processed = preprocessor.process(aligned)          # (T, 17, 3)

    if in_channels == 12:
        features = build_features_from_aligned(processed)  # (12, T, 17)
    else:
        # Legacy 9-channel path for old checkpoints
        velocity     = np.diff(processed, axis=0, prepend=processed[:1])
        acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
        feat = np.concatenate([processed, velocity, acceleration], axis=-1)
        features = np.transpose(feat, (2, 0, 1)).copy().astype(np.float32)

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
# Motion-phase ghost skeleton computation
# ---------------------------------------------------------------------------

def _motion_phase_signal(xy_seq):
    # type: (np.ndarray) -> np.ndarray
    """
    Compute a 1-D phase signal from hip vertical position.
    Input : (T, J, 2)  image-fraction XY
    Output: (T,)       smoothed phase in [0, 1]
    Squatting = hip Y rises (Y increases downward) = phase rises.
    """
    hip_mid_y = (xy_seq[:, HIP_L, 1] + xy_seq[:, HIP_R, 1]) / 2.0
    window = max(1, len(hip_mid_y) // 20)
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(hip_mid_y, kernel, mode='same')
    lo, hi = smoothed.min(), smoothed.max()
    if hi - lo < 1e-4:
        return np.linspace(0.0, 1.0, len(smoothed), dtype=np.float32)
    return ((smoothed - lo) / (hi - lo)).astype(np.float32)


def _build_video_to_template(user_phase, tmpl_phase):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    For every video frame, find the template frame with the closest phase.
    Returns (T_seq,) int array.
    """
    mapping = np.zeros(len(user_phase), dtype=np.int32)
    for i, up in enumerate(user_phase):
        mapping[i] = int(np.argmin(np.abs(tmpl_phase - up)))
        print(f"  [mapping] video_frame={i} user_phase={up:.3f} → template_frame={mapping[i]} tmpl_phase={tmpl_phase[mapping[i]]:.3f}")
    return mapping


def compute_perfect_ghost(warp_weights_np, template_xyz, raw_sequence, ghost_anchor: str = "hips"):
    # type: (np.ndarray, np.ndarray, np.ndarray, str) -> tuple
    """
    Build a ghost skeleton that moves WITH the user in real time.

    WHY THE OLD VERSION SAT STILL
    --------------------------------
    Old code drove ghost timing from warp_weights argmax.
    When the model is not well trained, warp weights are nearly uniform so
    argmax returns the same template frame for almost every user frame
    -> ghost is frozen at one pose for the entire video.

    NEW APPROACH: motion-phase tracking
    ------------------------------------
    Use the user's hip Y position as a phase signal (squatting = phase rises).
    Match each video frame to the template frame with the closest phase value.
    This makes the ghost genuinely reactive to the user's movement depth,
    regardless of model quality or training state.

    Per-frame anchoring
    --------------------
    Each frame: centre the template pose on the user's current hip midpoint
    and scale it to match the user's current body height. The ghost always
    sits on the user's body even if they move laterally.

    Returns
    -------
    ghost_screen  : (T_seq, J, 2)  ghost XY in [0,1], one per VIDEO frame
    user_synced   : (T_seq, J, 2)  user raw XY (same timeline)
    video_to_tmpl : (T_seq,) int   template frame index per video frame
    """
    _, T_t_raw, J = template_xyz.shape
    T_seq = raw_sequence.shape[0]

    # # Normalise template XY into [0, 1]
    # tmpl_xy = template_xyz[:2, :, :].copy()          # (2, T_t_raw, J)
    # for c in range(2):
    #     mn = float(tmpl_xy[c].min()); mx = float(tmpl_xy[c].max())
    #     if mx - mn > 1e-4:
    #         tmpl_xy[c] = (tmpl_xy[c] - mn) / (mx - mn)
    #     else:
    #         tmpl_xy[c] = 0.5
    # tmpl_xy_seq = tmpl_xy.transpose(1, 2, 0).astype(np.float32)  # (T_t_raw, J, 2)

    lat_axis = _detect_lateral_axis(template_xyz)

    # Use lateral axis + vertical axis (Z) instead of raw X/Y
    tmpl_xy = np.stack(
        [template_xyz[lat_axis], template_xyz[2]],
        axis=0,
    ).astype(np.float32)

    # Make vertical axis behave like screen Y
    tmpl_xy[1] = -tmpl_xy[1]

    for c in range(2):
        mn = float(np.min(tmpl_xy[c]))
        mx = float(np.max(tmpl_xy[c]))
        tmpl_xy[c] = (tmpl_xy[c] - mn) / max(mx - mn, 1e-6)

    tmpl_xy_seq = tmpl_xy.transpose(1, 2, 0).astype(np.float32)

    # Fix chirality: flip template X if user appears mirrored
    tmpl_lr = (float(np.median(tmpl_xy_seq[:, SHOULDER_L, 0]))
               - float(np.median(tmpl_xy_seq[:, SHOULDER_R, 0])))
    user_lr = (float(np.median(raw_sequence[:, SHOULDER_L, 0]))
               - float(np.median(raw_sequence[:, SHOULDER_R, 0])))
    if (tmpl_lr * user_lr) < 0:
        tmpl_xy_seq[:, :, 0] = 1.0 - tmpl_xy_seq[:, :, 0]
        print("  [ghost] Template X flipped to match user chirality.")

    # Deterministic mapping: linearly resample template timeline to video
    # timeline. We do NOT try to guess phase anymore — user requested that
    # every template frame be represented (interpolated or downsampled)
    # across the video frames.
    T_t = int(tmpl_xy_seq.shape[0])
    if T_t == T_seq:
        frac_positions = np.arange(T_seq, dtype=np.float32)
    else:
        frac_positions = np.linspace(0, T_t - 1, num=T_seq, dtype=np.float32)

    # Nearest integer mapping for external consumers
    video_to_tmpl = np.round(frac_positions).astype(np.int32)
    print(f"  [ghost] linear template mapping range: {video_to_tmpl.min()} -> {video_to_tmpl.max()} (tmpl_T={T_t})")

    # Body heights for per-frame scaling
    user_nose_y = raw_sequence[:, NOSE_IDX, 1]
    user_foot_y = (raw_sequence[:, FOOT_L_IDX, 1] + raw_sequence[:, FOOT_R_IDX, 1]) / 2.0
    user_height = np.abs(user_foot_y - user_nose_y).clip(min=1e-3)

    tmpl_nose_y = tmpl_xy_seq[:, NOSE_IDX, 1]
    tmpl_foot_y = (tmpl_xy_seq[:, FOOT_L_IDX, 1] + tmpl_xy_seq[:, FOOT_R_IDX, 1]) / 2.0
    tmpl_height = np.abs(tmpl_foot_y - tmpl_nose_y).clip(min=1e-3)
    user_height = np.abs(user_foot_y - user_nose_y).clip(min=1e-3)

    tmpl_height_med = max(float(np.median(tmpl_height)), 1e-3)
    user_height_med = max(float(np.median(user_height)), 1e-3)
    global_scale = np.clip(user_height_med / tmpl_height_med, 0.5, 4.0)
    print("tmpl_height:", tmpl_height.min(), tmpl_height.mean(), tmpl_height.max())


    # Build ghost per frame, anchored to selected user anchor position
    ghost_screen = np.zeros((T_seq, J, 2), dtype=np.float32)
    # for i in range(T_seq):
    #     t = int(video_to_tmpl[i])
    #     tmpl_pose = tmpl_xy_seq[t].copy()                          # (J, 2)
    #     tmpl_hip  = (tmpl_pose[HIP_L] + tmpl_pose[HIP_R]) / 2.0  # (2,)
    #     centred   = tmpl_pose - tmpl_hip                           # (J, 2)
    #     scale     = global_scale
    #     # print(f"i={i} t={t} tmpl_height={tmpl_height[t]:.6f} scale={scale:.3e}")
    #     user_hip  = np.array([
    #         (raw_sequence[i, HIP_L, 0] + raw_sequence[i, HIP_R, 0]) / 2.0,
    #         (raw_sequence[i, HIP_L, 1] + raw_sequence[i, HIP_R, 1]) / 2.0,
    #     ], dtype=np.float32)
    #     ghost_screen[i] = centred * scale + user_hip
    print(f"warp weights:", warp_weights_np.shape if warp_weights_np is not None else None)
    print(f"T_seq : {T_seq} tmpl_T : {T_t}")

    warp_resampled = None
    if warp_weights_np is not None:
        T_w, T_warp_cols = warp_weights_np.shape

        # Resample rows -> one row per video frame if needed
        if T_w != T_seq:
            orig_positions = np.linspace(0, T_seq - 1, num=T_w, dtype=np.float32)
            target_positions = np.arange(T_seq, dtype=np.float32)
            warp_resampled = np.zeros((T_seq, T_warp_cols), dtype=np.float32)
            for j in range(T_warp_cols):
                warp_resampled[:, j] = np.interp(target_positions, orig_positions, warp_weights_np[:, j])
            warp_resampled = warp_resampled / (warp_resampled.sum(axis=1, keepdims=True) + 1e-8)
            print(f"  [ghost] resampled warp rows {warp_weights_np.shape} -> {warp_resampled.shape}")
        else:
            warp_resampled = warp_weights_np.astype(np.float32)

        # Ensure warp columns align to template-frame axis (tmpl_xy_seq.shape[0]).
        target_T_t = int(tmpl_xy_seq.shape[0])
        if warp_resampled.shape[1] != target_T_t:
            orig_cols = np.arange(warp_resampled.shape[1], dtype=np.float32)
            target_cols = np.linspace(0, warp_resampled.shape[1] - 1, num=target_T_t, dtype=np.float32)
            warp_cols_resampled = np.zeros((warp_resampled.shape[0], target_T_t), dtype=np.float32)
            for r in range(warp_resampled.shape[0]):
                warp_cols_resampled[r] = np.interp(target_cols, orig_cols, warp_resampled[r])
            warp_cols_resampled = warp_cols_resampled / (warp_cols_resampled.sum(axis=1, keepdims=True) + 1e-8)
            warp_resampled = warp_cols_resampled
            print(f"  [ghost] resampled warp columns -> {warp_resampled.shape} to match tmpl frames={target_T_t}")

        # debug check
    if warp_resampled is not None:
        print("  [ghost] tmpl_xy_seq.shape:", tmpl_xy_seq.shape, "warp_resampled.shape:", warp_resampled.shape)

    # Diagnostics collected for user feedback (returned to caller)
    ghost_debug = {}
    if warp_resampled is not None:
        wr = warp_resampled
        # Representative frames to inspect
        sample_idx = [0, T_seq // 4, T_seq // 2, 3 * T_seq // 4, T_seq - 1]
        sample_idx = [int(max(0, min(T_seq - 1, i))) for i in sample_idx]
        topk_examples = []
        for i in sample_idx:
            row = wr[i]
            topk = np.argsort(-row)[:5]
            topk_examples.append({"frame": int(i), "topk": topk.tolist(), "weights": row[topk].tolist()})

        # Row entropy: low -> very peaked
        row_entropy = -np.sum(wr * np.log(wr + 1e-12), axis=1)
        ent_min, ent_mean, ent_max = float(row_entropy.min()), float(row_entropy.mean()), float(row_entropy.max())

        # Argmax usage stats
        argmax_seq = np.argmax(wr, axis=1)
        unique, counts = np.unique(argmax_seq, return_counts=True)
        most_common = sorted(list(zip(unique.tolist(), counts.tolist())), key=lambda x: -x[1])[:6]

        # Agreement between model argmax and phase mapping
        try:
            arg_eq_ratio = float((argmax_seq == video_to_tmpl).mean())
        except Exception:
            arg_eq_ratio = None

        # Correlation between user phase and template phase at argmax
        try:
            tmpl_at_argmax = tmpl_phase[argmax_seq]
            phase_corr = float(np.corrcoef(user_phase, tmpl_at_argmax)[0, 1])
        except Exception:
            phase_corr = None

        ghost_debug = {
            "warp_shape": tuple(wr.shape),
            "tmpl_shape": tuple(tmpl_xy_seq.shape),
            "sample_topk": topk_examples,
            "entropy": [ent_min, ent_mean, ent_max],
            "argmax_unique_count": int(len(unique)),
            "argmax_most_common": most_common,
            "argmax_eq_ratio": arg_eq_ratio,
            "phase_corr_user_vs_argmax": phase_corr,
        }

        print(f"  [ghost debug] warp rows entropy min/mean/max: {ent_min:.3f}/{ent_mean:.3f}/{ent_max:.3f}")
        print(f"  [ghost debug] argmax unique frames: {len(unique)} most_common: {most_common[:6]}")
        print(f"  [ghost debug] argmax==video_to_tmpl ratio: {arg_eq_ratio:.3f} phase_corr(user,tmpl[argmax])={phase_corr}")
    else:
        # No warp available — summarise phase mapping
        uniq, cnts = np.unique(video_to_tmpl, return_counts=True)
        ghost_debug = {
            "warp_shape": None,
            "tmpl_shape": tuple(tmpl_xy_seq.shape),
            "video_to_tmpl_unique_count": int(len(uniq)),
            "video_to_tmpl_top": sorted(list(zip(uniq.tolist(), cnts.tolist())), key=lambda x: -x[1])[:6],
        }

    # Precompute a per-video-frame template pose via linear interpolation
    # across the template timeline so we deterministically display the
    # template frames (interpolated or downsampled) rather than guessing.
    idx0 = np.floor(frac_positions).astype(int)
    idx1 = np.minimum(idx0 + 1, T_t - 1)
    alpha = (frac_positions - idx0).astype(np.float32)
    tmpl_pose_per_video = (1.0 - alpha)[:, None, None] * tmpl_xy_seq[idx0] + alpha[:, None, None] * tmpl_xy_seq[idx1]

    def _anchor_joint_pair(anchor_mode: str) -> tuple[int, int]:
        if anchor_mode == "heels":
            return 13, 14
        if anchor_mode == "ankles":
            return 11, 12
        if anchor_mode == "foot_index":
            return 15, 16
        return HIP_L, HIP_R

    left_anchor_idx, right_anchor_idx = _anchor_joint_pair(ghost_anchor)
    print(f"  [ghost] anchor mode={ghost_anchor} joints=({left_anchor_idx},{right_anchor_idx})")

    for i in range(T_seq):
        tmpl_pose = tmpl_pose_per_video[i].copy()

        tmpl_anchor = (tmpl_pose[left_anchor_idx] + tmpl_pose[right_anchor_idx]) / 2.0  # (2,)
        centred   = tmpl_pose - tmpl_anchor                          # (J, 2)
        scale     = global_scale
        user_anchor = np.array([
            (raw_sequence[i, left_anchor_idx, 0] + raw_sequence[i, right_anchor_idx, 0]) / 2.0,
            (raw_sequence[i, left_anchor_idx, 1] + raw_sequence[i, right_anchor_idx, 1]) / 2.0,
        ], dtype=np.float32)
        ghost_screen[i] = centred * scale + user_anchor

    user_synced = raw_sequence[:, :, :2].copy()   # (T_seq, J, 2)
    ghost_screen[..., 0] = np.clip(ghost_screen[..., 0], -0.05, 1.05)
    ghost_screen[..., 1] = np.clip(ghost_screen[..., 1], -0.05, 1.05)
    ghost_debug["anchor_mode"] = ghost_anchor
    ghost_debug["anchor_joints"] = [int(left_anchor_idx), int(right_anchor_idx)]
    return ghost_screen, user_synced, video_to_tmpl, ghost_debug



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
    ghost_anchor: str = "hips",
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
    print(f"  [overlay] template_xyz={'present' if best_loaded.template_xyz is not None else 'MISSING'}, "
          f"use_phase_decoder={best_loaded.use_phase_decoder}, "
          f"warp_weights_in_outputs={'warp_weights' in best_outputs}")
    if best_loaded.template_xyz is not None:
        # warp_np is only used as a pass-through argument to compute_perfect_ghost
        # (kept for API compatibility). The new ghost logic ignores it entirely
        # and drives timing from the user's motion phase instead.
        if "warp_weights" in best_outputs:
            warp_np = best_outputs["warp_weights"].detach().cpu()[0].numpy()
        else:
            T_u = best_outputs["similarity_score"].shape[0]
            T_t = best_loaded.template_tensor.shape[2]
            warp_np = np.ones((T_u, T_t), dtype=np.float32) / T_t  # uniform fallback

        tmpl_xyz_np = best_loaded.template_xyz[0].cpu().numpy()   # (3, T_t, J)

        ghost_screen, user_synced, video_to_model, ghost_debug = compute_perfect_ghost(
            warp_np, tmpl_xyz_np, raw_sequence, ghost_anchor=ghost_anchor
        )

        # ghost_screen: (T_u, J, 2) in [0,1] screen coords
        # Pack into (T_u, J, 3) for downstream code (Z=0 unused)
        # ghost_screen is now (T_seq, J, 2) — same length as the video, not T_u
        ghost_xyz = np.zeros((ghost_screen.shape[0], ghost_screen.shape[1], 3), dtype=np.float32)
        ghost_xyz[..., :2] = ghost_screen

        # user_synced is (T_seq, J, 2) — same timeline
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
            "ghost_debug":     ghost_debug,
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
        ghost_seq       = phase_outputs["ghost_xyz"]        # (T_seq, J, 3)
        user_synced_seq = phase_outputs["user_synced"]      # (T_seq, J, 3)
        joint_conf_seq  = phase_outputs["joint_confidence"] # (T_u, J)  model timeline
        video_to_model  = phase_outputs["video_to_model"]   # (T_seq,) template frame index
        T_model         = ghost_seq.shape[0]  # now equals T_seq

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

            # Ghost is now (T_seq, J, 3) — one entry per video frame.
            # Index directly by frame_idx; no remapping needed.
            if T_model > 0:
                model_t = min(frame_idx, T_model - 1)
            else:
                model_t = 0

            # FIX E: compute temporal correlation for this window
            joint_corr: np.ndarray | None = None
            if has_phase and ghost_seq is not None and user_synced_seq is not None:
                # Both signals are now on the video timeline (T_seq frames)
                joint_corr = compute_temporal_correlation(
                    ghost_seq[..., :2],       # (T_seq, J, 2)
                    user_synced_seq[..., :2], # (T_seq, J, 2)
                    model_t,
                )

            if has_phase and ghost_seq is not None:
                if frame_idx == 0:  # print once
                    g = ghost_seq[model_t]
                    print(f"  [ghost debug] frame0 XY min=({g[:,:2].min():.3f}) max=({g[:,:2].max():.3f})")
                    in_range = np.sum((g[:,0] >= -0.05) & (g[:,0] <= 1.05) & (g[:,1] >= -0.05) & (g[:,1] <= 1.05))
                    print(f"  [ghost debug] joints in range: {in_range}/17")
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
                    # joint_conf_seq is on the model timeline (T_u frames)
                    # map frame_idx proportionally
                    T_conf = joint_conf_seq.shape[0]
                    total_frames_approx = max(1, T_model)
                    conf_t = min(int(frame_idx * T_conf / total_frames_approx), T_conf - 1)
                    draw_correction_arrows(
                        frame, user_pts,
                        ghost_seq[model_t],
                        joint_conf_seq[conf_t],
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
    ghost_anchor: str = "hips",
) -> dict:
    raw_sequence  = extract_mediapipe_sequence(video_path, pose_model_path)
    in_ch = models[0].in_channels if models else 12
    input_tensor  = build_model_input(raw_sequence, preprocessor, in_channels=in_ch)

    summary, joint_importance, worst_joints, phase_outputs = run_prediction(
        input_tensor, raw_sequence, models, device, ghost_anchor=ghost_anchor
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
        "ghost_anchor":          ghost_anchor,
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
    parser.add_argument(
        "--ghost-anchor",
        type=str,
        default="hips",
        choices=["hips", "ankles", "heels", "foot_index"],
        help="Anchor ghost overlay to this body reference (default: hips).",
    )
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
    
    # NEW: Create a list to store our rows for the CSV
    evaluation_rows = [] 

    for video_path in video_files:
        exercise_id = int(video_path.stem.split("-")[0].replace("ex", ""))
        model_idx   = exercise_id - 1
        if exercise_id < 1 or exercise_id > len(models):
            print(f"Skipping {video_path.name}: exercise ID {exercise_id} out of range.")
            continue
            
        report = process_video(
            video_path, specific_models[model_idx],  # type: ignore[arg-type]
            preprocessor, args.output_dir, device, pose_model_path,
            ghost_anchor=args.ghost_anchor,
        )
        all_reports.append(report)
        print(json.dumps(report["best"], indent=2))

        # NEW: Parse the expected label from the filename and append the data row
        best = report["best"]
        expected_label = "incorrect" if "inc" in video_path.stem.lower() else "correct"
        actual_label = best["predicted_label"]
        
        evaluation_rows.append({
            "Video Name": video_path.name,
            "Exercise": best["exercise_name"],
            "Score": best["score"],
            "Threshold": best["threshold"],
            "Expected": expected_label,
            "Actual": actual_label,
            "Match": "TRUE" if expected_label == actual_label else "FALSE"
        })

    summary_path = args.output_dir / "all_videos_summary.json"
    summary_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    # NEW: Write the collected evaluation rows to a CSV file
    csv_path = args.output_dir / "evaluation_overview.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["Video Name", "Exercise", "Score", "Threshold", "Expected", "Actual", "Match"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_rows)
        
    print(f"Saved evaluation overview: {csv_path}")


if __name__ == "__main__":
    main()