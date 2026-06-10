from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from Models import ExerciseEvaluator
from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor, build_features_from_aligned

__all__ = [
    "LoadedExerciseModel",
    "annotate_video",
    "build_model_input",
    "compute_joint_deviation",
    "compute_perfect_ghost",
    "compute_temporal_correlation",
    "draw_correction_arrows",
    "draw_ghost_skeleton",
    "draw_hud",
    "draw_skeleton",
    "ensure_pose_task_model",
    "extract_mediapipe_sequence",
    "get_cached_models",
    "load_models",
    "main",
    "parse_args",
    "process_video",
    "resolve_device",
    "run_prediction",
]

BASE_DIR = Path(__file__).resolve().parent
MP_COCO17_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_POSE_MODEL_PATH = Path(".cache") / "mediapipe" / "pose_landmarker_full.task"

COCO17_NAMES = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

COCO17_EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

COLOR_CORRECT = (200, 200, 200)
COLOR_IMPORTANT = (0, 200, 255)
COLOR_PROBLEM = (0, 60, 255)
COLOR_OVERLAY = (220, 130, 30)
COLOR_ARROW = (30, 220, 255)
COLOR_HUD_BG = (20, 20, 20)

THRESHOLD_FLOOR = 0.05
FOOT_L_IDX = 11
FOOT_R_IDX = 12
NOSE_IDX = 0
SHOULDER_L = 1
SHOULDER_R = 2
HIP_L = 7
HIP_R = 8
CORR_WINDOW = 15


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
    in_channels: int = 9

    @property
    def model_path(self) -> Path:
        return self.checkpoint_path


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coerce_int(value: object, default: int | None = None) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            if default is not None:
                return default
            raise ValueError("Cannot coerce an empty list to int")
        value = value[0]
    if value is None:
        if default is not None:
            return default
        raise ValueError("Cannot coerce None to int")
    return int(value)


def load_models(checkpoints_root: Path, device: torch.device) -> list[LoadedExerciseModel]:
    loaded: list[LoadedExerciseModel] = []

    for exercise_dir in sorted(checkpoints_root.glob("exercise_*")):
        ckpt_path = exercise_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model_state_dict = ckpt["model_state_dict"]
        state_key = "encoder.0.spatial_attn.proj.weight"
        in_channels = int(model_state_dict[state_key].shape[1]) if state_key in model_state_dict else int(ckpt.get("in_channels", 9))
        use_phase_decoder = bool(ckpt.get("use_phase_decoder", False))

        model = ExerciseEvaluator(
            in_channels=in_channels,
            hidden_channels=tuple(cfg.get("hidden_channels", (64, 128))),
            embedding_dim=_coerce_int(cfg.get("embedding_dim"), 128),
            use_phase_decoder=use_phase_decoder,
        ).to(device)

        missing, _ = model.load_state_dict(model_state_dict, strict=False)
        if missing:
            print(f"  [compat] {ckpt_path.name}: {len(missing)} keys missing.")
        model.eval()

        exercise_id = _coerce_int(ckpt.get("exercise_id"), int(exercise_dir.name.split("_")[-1]) - 1)
        raw_threshold = float(ckpt.get("val_threshold", 0.5))
        threshold = max(THRESHOLD_FLOOR, raw_threshold)

        if raw_threshold < THRESHOLD_FLOOR:
            print(
                f"  [WARN] Exercise {exercise_id}: threshold={raw_threshold:.3f} below floor. "
                f"Clamped to {threshold:.3f}. Consider retraining."
            )

        template_tensor = ckpt["template_tensor"].detach().clone().float().unsqueeze(0).to(device)
        if "template_xyz_tensor" in ckpt:
            template_xyz = ckpt["template_xyz_tensor"].detach().clone().float().unsqueeze(0).to(device)
        else:
            tmpl_t = ckpt["template_tensor"].detach().clone().float()
            template_xyz = tmpl_t[:3].unsqueeze(0).to(device)
            print(f"  [compat] {ckpt_path.name}: no template_xyz_tensor, reconstructed from template_tensor XYZ channels.")

        loaded.append(
            LoadedExerciseModel(
                exercise_id=exercise_id,
                model=model,
                template_tensor=template_tensor,
                template_xyz=template_xyz,
                threshold=threshold,
                raw_threshold=raw_threshold,
                checkpoint_path=ckpt_path,
                use_phase_decoder=use_phase_decoder,
                in_channels=in_channels,
            )
        )

    if not loaded:
        raise FileNotFoundError(f"No best_checkpoint.pt found under {checkpoints_root}")
    return loaded


@lru_cache(maxsize=8)
def _cached_models(checkpoints_root: str, device_key: str) -> tuple[LoadedExerciseModel, ...]:
    device = torch.device(device_key)
    return tuple(load_models(Path(checkpoints_root), device))


def get_cached_models(checkpoints_root: Path, device: torch.device) -> list[LoadedExerciseModel]:
    return list(_cached_models(str(checkpoints_root.resolve()), str(device)))


def ensure_pose_task_model(model_path: Path | None) -> Path:
    resolved = Path(model_path) if model_path is not None else DEFAULT_POSE_MODEL_PATH
    if resolved.exists():
        return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {resolved}…")
    urllib.request.urlretrieve(DEFAULT_POSE_MODEL_URL, str(resolved))
    print("Done.")
    return resolved


def extract_mediapipe_sequence(
    video_path: Path, 
    pose_model_path: Path,
    min_valid_frames: int = 5,
    min_total_frames: int = 15,
    min_avg_confidence: float = 0.65  # NEW: Strict confidence filter
) -> np.ndarray:
    
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
    valid_frame_count = 0 

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            det = landmarker.detect_for_video(mp_img, ts_ms)

            is_frame_valid = False

            if det.pose_landmarks:
                lms = det.pose_landmarks[0]
                
                # --- GHOST SKELETON FILTER ---
                # Calculate the average visibility of our 17 target joints
                confidences = [lms[i].visibility for i in MP_COCO17_IDXS if hasattr(lms[i], 'visibility')]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Only accept the skeleton if MediaPipe is highly confident
                if avg_conf >= min_avg_confidence:
                    is_frame_valid = True
                # -----------------------------

            if is_frame_valid:
                pts = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in MP_COCO17_IDXS], dtype=np.float32)
                last_valid = pts
                valid_frame_count += 1
            else:
                # Treat ghost skeletons as empty frames
                pts = np.zeros((17, 3), dtype=np.float32) if last_valid is None else last_valid.copy()
            
            sequence.append(pts)
            frame_idx += 1

    cap.release()

    # --- ERROR HANDLING ---
    if not sequence:
        raise RuntimeError(f"No frames read from {video_path}")

    if len(sequence) < min_total_frames:
        raise ValueError("VIDEO_TOO_SHORT")

    if valid_frame_count < min_valid_frames:
        raise ValueError("NO_HUMAN_DETECTED")

    return np.stack(sequence, axis=0).astype(np.float32)


def build_model_input(sequence: np.ndarray, preprocessor: UIPRMDPreprocessor, in_channels: int = 12) -> torch.Tensor:
    aligned = preprocessor.align_vicon_to_mediapipe(sequence)
    processed = preprocessor.process(aligned)

    if in_channels == 12:
        features = build_features_from_aligned(processed)
    else:
        velocity = np.diff(processed, axis=0, prepend=processed[:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
        feat = np.concatenate([processed, velocity, acceleration], axis=-1)
        features = np.transpose(feat, (2, 0, 1)).copy().astype(np.float32)

    return torch.from_numpy(features).float().unsqueeze(0)


def _detect_lateral_axis(template_xyz: np.ndarray) -> int:
    sep_x = float(np.abs(template_xyz[0, :, SHOULDER_L] - template_xyz[0, :, SHOULDER_R]).mean())
    sep_y = float(np.abs(template_xyz[1, :, SHOULDER_L] - template_xyz[1, :, SHOULDER_R]).mean())
    return 0 if sep_x >= sep_y else 1


def compute_perfect_ghost(warp_weights_np, template_xyz, raw_sequence, ghost_anchor: str = "hips"):
    _, _, J = template_xyz.shape
    T_seq = raw_sequence.shape[0]

    lat_axis = _detect_lateral_axis(template_xyz)
    tmpl_xy = np.stack([template_xyz[lat_axis], template_xyz[2]], axis=0).astype(np.float32)
    tmpl_xy[1] = -tmpl_xy[1]
    for c in range(2):
        mn = float(np.min(tmpl_xy[c]))
        mx = float(np.max(tmpl_xy[c]))
        tmpl_xy[c] = (tmpl_xy[c] - mn) / max(mx - mn, 1e-6)

    tmpl_xy_seq = tmpl_xy.transpose(1, 2, 0).astype(np.float32)
    tmpl_lr = float(np.median(tmpl_xy_seq[:, SHOULDER_L, 0])) - float(np.median(tmpl_xy_seq[:, SHOULDER_R, 0]))
    user_lr = float(np.median(raw_sequence[:, SHOULDER_L, 0])) - float(np.median(raw_sequence[:, SHOULDER_R, 0]))
    if (tmpl_lr * user_lr) < 0:
        tmpl_xy_seq[:, :, 0] = 1.0 - tmpl_xy_seq[:, :, 0]

    T_t = int(tmpl_xy_seq.shape[0])
    frac_positions = np.arange(T_seq, dtype=np.float32) if T_t == T_seq else np.linspace(0, T_t - 1, num=T_seq, dtype=np.float32)
    video_to_tmpl = np.round(frac_positions).astype(np.int32)

    user_nose_y = raw_sequence[:, NOSE_IDX, 1]
    user_foot_y = (raw_sequence[:, FOOT_L_IDX, 1] + raw_sequence[:, FOOT_R_IDX, 1]) / 2.0
    user_height = np.abs(user_foot_y - user_nose_y).clip(min=1e-3)

    tmpl_nose_y = tmpl_xy_seq[:, NOSE_IDX, 1]
    tmpl_foot_y = (tmpl_xy_seq[:, FOOT_L_IDX, 1] + tmpl_xy_seq[:, FOOT_R_IDX, 1]) / 2.0
    tmpl_height = np.abs(tmpl_foot_y - tmpl_nose_y).clip(min=1e-3)

    tmpl_height_med = max(float(np.median(tmpl_height)), 1e-3)
    user_height_med = max(float(np.median(user_height)), 1e-3)
    global_scale = np.clip(user_height_med / tmpl_height_med, 0.5, 4.0)

    ghost_screen = np.zeros((T_seq, J, 2), dtype=np.float32)
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
    for i in range(T_seq):
        tmpl_pose = tmpl_pose_per_video[i].copy()
        tmpl_anchor = (tmpl_pose[left_anchor_idx] + tmpl_pose[right_anchor_idx]) / 2.0
        centred = tmpl_pose - tmpl_anchor
        user_anchor = np.array([
            (raw_sequence[i, left_anchor_idx, 0] + raw_sequence[i, right_anchor_idx, 0]) / 2.0,
            (raw_sequence[i, left_anchor_idx, 1] + raw_sequence[i, right_anchor_idx, 1]) / 2.0,
        ], dtype=np.float32)
        ghost_screen[i] = centred * global_scale + user_anchor

    user_synced = raw_sequence[:, :, :2].copy()
    ghost_screen[..., 0] = np.clip(ghost_screen[..., 0], -0.05, 1.05)
    ghost_screen[..., 1] = np.clip(ghost_screen[..., 1], -0.05, 1.05)
    ghost_debug = {"tmpl_shape": tuple(tmpl_xy_seq.shape), "anchor_mode": ghost_anchor}
    return ghost_screen, user_synced, video_to_tmpl, ghost_debug


def compute_temporal_correlation(ghost_screen: np.ndarray, user_synced: np.ndarray, model_t: int, window: int = CORR_WINDOW) -> np.ndarray:
    T_u, J, _ = ghost_screen.shape
    t_lo = max(0, model_t - window // 2)
    t_hi = min(T_u, model_t + window // 2 + 1)
    if t_hi - t_lo < 3:
        return np.zeros(J, dtype=np.float32)
    g = ghost_screen[t_lo:t_hi, :, 1]
    u = user_synced[t_lo:t_hi, :, 1]
    g_c = g - g.mean(axis=0, keepdims=True)
    u_c = u - u.mean(axis=0, keepdims=True)
    num = (g_c * u_c).sum(axis=0)
    denom = np.sqrt((g_c**2).sum(axis=0) * (u_c**2).sum(axis=0)) + 1e-8
    return (num / denom).astype(np.float32)


def compute_joint_deviation(user_tensor: torch.Tensor, template_tensor: torch.Tensor, joint_importance: np.ndarray) -> list[dict]:
    user_np = user_tensor.detach().cpu().numpy()[0]
    template_np = template_tensor.detach().cpu().numpy()[0]
    pos_dev = np.abs(user_np[:3] - template_np[:3]).mean(axis=(0, 1))
    imp = joint_importance.astype(np.float32)
    dev_norm = (pos_dev - pos_dev.min()) / max(1e-6, float(pos_dev.max() - pos_dev.min()))
    imp_norm = (imp - imp.min()) / max(1e-6, float(imp.max() - imp.min()))
    combined = 0.75 * dev_norm + 0.25 * imp_norm
    ranked = np.argsort(-combined)
    return [
        {
            "joint": COCO17_NAMES[int(idx)],
            "joint_index": int(idx),
            "deviation": float(pos_dev[int(idx)]),
            "importance": float(imp[int(idx)]),
            "problem_score": float(combined[int(idx)]),
        }
        for idx in ranked
    ]


def run_prediction(input_tensor: torch.Tensor, raw_sequence: np.ndarray, models: list[LoadedExerciseModel], device: torch.device, ghost_anchor: str = "hips") -> tuple[dict, np.ndarray, list[dict], dict | None]:
    results = []
    with torch.no_grad():
        user_tensor = input_tensor.to(device)
        for loaded in models:
            outputs = loaded.model(loaded.template_tensor, user_tensor)
            score = float(outputs["similarity_score"].item())
            pred = score >= loaded.threshold
            margin = score - loaded.threshold
            joint_importance = outputs["joint_importance"].detach().cpu().numpy()[0]

            item = {
                "exercise_id": loaded.exercise_id,
                "exercise_name": f"exercise_{loaded.exercise_id + 1:02d}",
                "score": score,
                "threshold": loaded.threshold,
                "raw_threshold": loaded.raw_threshold,
                "margin": margin,
                "predicted_label": "correct" if pred else "incorrect",
                "checkpoint": str(loaded.checkpoint_path),
                "has_phase_decoder": loaded.use_phase_decoder,
            }
            results.append({"item": item, "joint_importance": joint_importance, "loaded": loaded, "outputs": outputs})

    best_result = max(results, key=lambda x: x["item"]["margin"])
    best_item = best_result["item"]
    best_imp = best_result["joint_importance"]
    best_loaded = best_result["loaded"]
    best_outputs = best_result["outputs"]

    phase_outputs: dict | None = None
    if best_loaded.template_xyz is not None:
        warp_np = best_outputs["warp_weights"].detach().cpu()[0].numpy() if "warp_weights" in best_outputs else None
        tmpl_xyz_np = best_loaded.template_xyz[0].cpu().numpy()
        ghost_screen, user_synced, video_to_model, ghost_debug = compute_perfect_ghost(warp_np, tmpl_xyz_np, raw_sequence, ghost_anchor=ghost_anchor)
        ghost_xyz = np.zeros((ghost_screen.shape[0], ghost_screen.shape[1], 3), dtype=np.float32)
        ghost_xyz[..., :2] = ghost_screen
        user_synced_3 = np.zeros_like(ghost_xyz)
        user_synced_3[..., :2] = user_synced
        err_mag = best_outputs["joint_error_magnitude"].detach().cpu().numpy()[0] if "joint_error_magnitude" in best_outputs else np.linalg.norm(ghost_screen - user_synced, axis=-1).mean(axis=0)
        joint_conf = best_outputs["joint_confidence"].detach().cpu().numpy()[0] if "joint_confidence" in best_outputs else np.ones((ghost_screen.shape[0], 17), dtype=np.float32)
        phase_outputs = {
            "ghost_xyz": ghost_xyz,
            "user_synced": user_synced_3,
            "delta_xy": ghost_xyz - user_synced_3,
            "joint_error_mag": err_mag,
            "joint_confidence": joint_conf,
            "video_to_model": video_to_model,
            "ghost_debug": ghost_debug,
        }

    if phase_outputs is not None:
        err_mag = phase_outputs["joint_error_mag"]
        ranked = np.argsort(-err_mag)
        worst_joints = [
            {
                "joint": COCO17_NAMES[int(idx)],
                "joint_index": int(idx),
                "deviation": float(err_mag[int(idx)]),
                "importance": float(best_imp[int(idx)]),
                "problem_score": float(0.6 * err_mag[int(idx)] / max(1e-6, err_mag.max()) + 0.4 * best_imp[int(idx)] / max(1e-6, best_imp.max())),
            }
            for idx in ranked
        ]
    else:
        worst_joints = compute_joint_deviation(input_tensor.to(device), best_loaded.template_tensor, best_imp)

    all_results = [r["item"] for r in sorted(results, key=lambda x: x["item"]["margin"], reverse=True)]
    return {"best": best_item, "all": all_results}, best_imp, worst_joints, phase_outputs


def _pixel(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return int(x * w), int(y * h)


def _joint_color(idx: int, bad_idxs: set[int], imp_idxs: set[int]) -> tuple[int, int, int]:
    if idx in bad_idxs:
        return COLOR_PROBLEM
    if idx in imp_idxs:
        return COLOR_IMPORTANT
    return COLOR_CORRECT


def draw_skeleton(frame: np.ndarray, points: list[tuple[int, int] | None], joint_colors: list[tuple[int, int, int]], norm: np.ndarray, width: int, bad_idxs: set[int], imp_idxs: set[int], line_thickness: int = 2) -> None:
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


def draw_ghost_skeleton(frame: np.ndarray, ghost_xyz: np.ndarray, width: int, height: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    pts: list[tuple[int, int] | None] = []
    for j in range(17):
        x, y = float(ghost_xyz[j, 0]), float(ghost_xyz[j, 1])
        if -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05:
            pts.append(_pixel(max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), width, height))
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


def draw_correction_arrows(frame: np.ndarray, user_pts: list[tuple[int, int] | None], ghost_xyz: np.ndarray, joint_conf: np.ndarray, width: int, height: int, bad_idxs: set[int], min_px: int = 6) -> None:
    for j in bad_idxs:
        if user_pts[j] is None:
            continue
        gx, gy = float(ghost_xyz[j, 0]), float(ghost_xyz[j, 1])
        if not (-0.05 <= gx <= 1.05 and -0.05 <= gy <= 1.05):
            continue
        tip = (int(max(0, min(width - 1, gx * width))), int(max(0, min(height - 1, gy * height))))
        dx = tip[0] - user_pts[j][0]
        dy = tip[1] - user_pts[j][1]
        if abs(dx) < min_px and abs(dy) < min_px:
            continue
        thickness = max(1, int(1 + float(joint_conf[j]) * 3))
        cv2.arrowedLine(frame, user_pts[j], tip, COLOR_ARROW, thickness, cv2.LINE_AA, tipLength=0.25)


def draw_hud(frame: np.ndarray, best: dict, worst_joints: list[dict], top3_names: list[str], has_phase: bool, joint_corr: np.ndarray | None = None) -> None:
    lines = [
        f"Pred: {best['predicted_label']}  ({best['exercise_name']})",
        f"Score: {best['score']:.3f}  |  Thr: {best['threshold']:.3f}",
        f"Top-attn: {', '.join(top3_names)}",
    ]
    if has_phase and joint_corr is not None:
        sync_parts = []
        for w in worst_joints[:3]:
            ji = int(w["joint_index"])
            corr_val = float(joint_corr[ji]) if ji < len(joint_corr) else 0.0
            sync_parts.append(f"{w['joint']}[sync={corr_val:+.2f}]")
        lines.append("Fix: " + "  ".join(sync_parts))
    elif has_phase:
        lines.append("Fix: " + "  ".join(f"{w['joint']} ({w['deviation']:.3f})" for w in worst_joints[:3]))
    else:
        lines.append("Fix: " + ", ".join(w["joint"] for w in worst_joints[:3]))

    font_scale, thickness, pad, line_h = 0.60, 2, 8, 26
    panel_w = 640
    panel_h = len(lines) * line_h + pad * 2
    sub = frame[0:panel_h, 0:panel_w].copy()
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(frame[0:panel_h, 0:panel_w], 0.65, sub, 0.35, 0, frame[0:panel_h, 0:panel_w])
    fix_color = (80, 140, 255)
    if joint_corr is not None and worst_joints:
        mean_worst_corr = np.mean([float(joint_corr[int(w["joint_index"])]) for w in worst_joints[:3] if int(w["joint_index"]) < len(joint_corr)])
        if mean_worst_corr > 0.6:
            fix_color = (10, 220, 10)
        elif mean_worst_corr > 0.2:
            fix_color = (0, 200, 255)
        else:
            fix_color = (60, 60, 255)
    colors = [(10, 220, 10) if best["predicted_label"] == "correct" else (60, 60, 255), (240, 240, 240), (120, 220, 255), fix_color]
    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(frame, line, (pad, pad + (i + 1) * line_h - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def annotate_video(video_path: Path, output_path: Path, summary: dict, joint_importance: np.ndarray, worst_joints: list[dict], phase_outputs: dict | None, pose_model_path: Path) -> str:
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    used_codec = ""
    for codec in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if candidate.isOpened():
            writer = candidate
            used_codec = codec
            break
        candidate.release()
    if writer is None:
        cap.release()
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    norm_imp = joint_importance.astype(np.float32)
    norm_imp = (norm_imp - norm_imp.min()) / max(1e-6, float(norm_imp.max() - norm_imp.min()))
    top3_idxs = set(int(i) for i in np.argsort(-joint_importance)[:3])
    top3_names = [COCO17_NAMES[i] for i in np.argsort(-joint_importance)[:3]]
    bad_idxs = {int(w["joint_index"]) for w in worst_joints[:3]}
    has_phase = phase_outputs is not None

    ghost_seq = None
    user_synced_seq = None
    joint_conf_seq = None
    T_model = 0
    if has_phase:
        ghost_seq = phase_outputs["ghost_xyz"]
        user_synced_seq = phase_outputs["user_synced"]
        joint_conf_seq = phase_outputs["joint_confidence"]
        T_model = ghost_seq.shape[0]

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            model_t = min(frame_idx, T_model - 1) if T_model > 0 else 0
            joint_corr: np.ndarray | None = None
            if has_phase and ghost_seq is not None and user_synced_seq is not None:
                joint_corr = compute_temporal_correlation(ghost_seq[..., :2], user_synced_seq[..., :2], model_t)

            if has_phase and ghost_seq is not None:
                draw_ghost_skeleton(frame, ghost_seq[model_t], width, height)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                user_pts: list[tuple[int, int] | None] = []
                joint_colors: list[tuple[int, int, int]] = []
                for local_idx, mp_idx in enumerate(MP_COCO17_IDXS):
                    lm = lms[mp_idx]
                    px, py = int(lm.x * width), int(lm.y * height)
                    user_pts.append((px, py) if 0 <= px < width and 0 <= py < height else None)
                    base_color = _joint_color(local_idx, bad_idxs, top3_idxs)
                    if joint_corr is not None and local_idx in bad_idxs and float(joint_corr[local_idx]) < 0.2:
                        base_color = (0, 0, 255)
                    joint_colors.append(base_color)

                draw_skeleton(frame, user_pts, joint_colors, norm_imp, width, bad_idxs, top3_idxs)

                if has_phase and ghost_seq is not None and joint_conf_seq is not None:
                    T_conf = joint_conf_seq.shape[0]
                    conf_t = min(int(frame_idx * T_conf / max(1, T_model)), T_conf - 1)
                    draw_correction_arrows(frame, user_pts, ghost_seq[model_t], joint_conf_seq[conf_t], width, height, bad_idxs)

            draw_hud(frame, summary["best"], worst_joints, top3_names, has_phase, joint_corr)
            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    return used_codec


def process_video(
    video_path: Path, 
    models: list[LoadedExerciseModel], 
    preprocessor: UIPRMDPreprocessor, 
    output_dir: Path, 
    device: torch.device, 
    pose_model_path: Path, 
    ghost_anchor: str = "hips"
) -> dict:
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. SAFE EXTRACTION WITH ERROR HANDLING ---
    try:
        raw_sequence = extract_mediapipe_sequence(video_path, pose_model_path)
    except ValueError as e:
        error_msg = str(e)
        if error_msg == "NO_HUMAN_DETECTED":
            return {
                "error": True,
                "message": "No human body detected. Please ensure the patient is clearly visible in the camera frame."
            }
        elif error_msg == "VIDEO_TOO_SHORT":
            return {
                "error": True,
                "message": "The video is too short to analyze. Please upload a video containing a complete exercise repetition."
            }
        raise  # Re-raise if it's an unexpected ValueError
    # ----------------------------------------------

    # 2. PROCEED WITH NORMAL INFERENCE
    in_ch = models[0].in_channels if models else 12
    input_tensor = build_model_input(raw_sequence, preprocessor, in_channels=in_ch)

    summary, joint_importance, worst_joints, phase_outputs = run_prediction(
        input_tensor, raw_sequence, models, device, ghost_anchor=ghost_anchor
    )

    annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
    output_codec = annotate_video(
        video_path, annotated_path, summary, joint_importance, worst_joints, phase_outputs, pose_model_path
    )

    # 3. BUILD SUCCESS REPORT
    report = {
        "video": str(video_path),
        "annotated_video": str(annotated_path),
        "annotated_video_codec": output_codec,
        "num_frames": int(raw_sequence.shape[0]),
        "has_phase_decoder": phase_outputs is not None,
        "ghost_anchor": ghost_anchor,
        "worst_joints": worst_joints[:5],
        **summary,
    }
    
    (output_dir / f"{video_path.stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference on Video-kineto with phase-aware corrective feedback.")
    parser.add_argument("--video", type=Path, default=None, help="Optional single video path")
    parser.add_argument("--exercise-id", type=int, default=1, help="1-based exercise id for single-video runs")
    parser.add_argument("--input-dir", type=Path, default=Path("Video-kineto") / "UIPRMD-videos")
    parser.add_argument("--output-dir", type=Path, default=Path("Video-kineto-annotated"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints") / "uiprmd")
    parser.add_argument("--pose-model", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ghost-anchor", type=str, default="hips", choices=["hips", "ankles", "heels", "foot_index"], help="Anchor ghost overlay to this body reference (default: hips).")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_model_path = ensure_pose_task_model(args.pose_model)

    device = resolve_device(args.device)
    models = load_models(args.checkpoints_root, device)
    preprocessor = UIPRMDPreprocessor()

    if args.video is not None:
        selected_models = [model for model in models if model.exercise_id == args.exercise_id - 1]
        if not selected_models:
            selected_models = models
        report = process_video(args.video, selected_models, preprocessor, args.output_dir, device, pose_model_path, ghost_anchor=args.ghost_anchor)
        print(json.dumps(report["best"], indent=2))
        return

    specific_models: list[list[LoadedExerciseModel] | None] = [None] * len(models)
    for idx, m in enumerate(models):
        print(f"Loaded exercise {m.exercise_id} from {m.checkpoint_path} (thr={m.threshold:.3f} raw={m.raw_threshold:.3f}, phase_decoder={m.use_phase_decoder})")
        specific_models[idx] = [m]

    video_files = sorted(args.input_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {args.input_dir}")

    all_reports = []
    evaluation_rows = []

    for video_path in video_files:
        exercise_id = int(video_path.stem.split("-")[0].replace("ex", ""))
        model_idx = exercise_id - 1
        if exercise_id < 1 or exercise_id > len(models):
            print(f"Skipping {video_path.name}: exercise ID {exercise_id} out of range.")
            continue

        report = process_video(video_path, specific_models[model_idx], preprocessor, args.output_dir, device, pose_model_path, ghost_anchor=args.ghost_anchor)
        all_reports.append(report)
        print(json.dumps(report["best"], indent=2))

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
            "Match": "TRUE" if expected_label == actual_label else "FALSE",
        })

    summary_path = args.output_dir / "all_videos_summary.json"
    summary_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    csv_path = args.output_dir / "evaluation_overview.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["Video Name", "Exercise", "Score", "Threshold", "Expected", "Actual", "Match"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    print(f"Saved evaluation overview: {csv_path}")


if __name__ == "__main__":
    main()
