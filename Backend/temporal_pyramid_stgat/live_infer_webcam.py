#!/usr/bin/env python3
"""
Live webcam inference for Temporal Pyramid STGAT.

Features:
- Real-time MediaPipe 33-joint extraction from webcam
- Live skeleton rendering
- Rolling-window inference with score + Correct/Incorrect label
- Optional reference (expected) skeleton overlay in a different color

Usage:
  python -m temporal_pyramid_stgat.live_infer_webcam \
      --checkpoint ./temporal_pyramid_stgat/weights/pyramid_stgat_mediapipe33_exercise_0_best_best_acc.pt \
      --exercise 0 \
      --camera-index 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# Add Backend root so absolute imports resolve when running this file directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from temporal_pyramid_stgat.inference import PyramidSTGATInference
from temporal_pyramid_stgat.preprocessing.mediapipe_uiprmd_loader import MediaPipeUIsprmdLoader
from temporal_pyramid_stgat.utils.inference_utils import QualityScoreInterpreter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SKELETON_EDGES_33 = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
]


def _resolve_pose_model() -> str:
    model_dir = Path("temporal_pyramid_stgat") / "weights"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pose_landmarker_lite.task"
    if model_path.exists():
        return str(model_path)

    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    logger.info(f"Downloading MediaPipe pose model: {url}")
    import urllib.request

    urllib.request.urlretrieve(url, str(model_path))
    return str(model_path)


def _resize_time(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linear temporal interpolation to (target_len, J, C)."""
    t, j, c = seq.shape
    if t == target_len:
        return seq.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, t)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, j, c), dtype=np.float32)
    for jj in range(j):
        for cc in range(c):
            out[:, jj, cc] = np.interp(x_new, x_old, seq[:, jj, cc])
    return out


def _adapt_joint_count(seq: np.ndarray, target_joints: int) -> np.ndarray:
    t, j, c = seq.shape
    if j == target_joints:
        return seq
    if j > target_joints:
        return seq[:, :target_joints, :]
    pad = np.zeros((t, target_joints - j, c), dtype=np.float32)
    return np.concatenate([seq, pad], axis=1)


def _resolve_uiprmd_root() -> Optional[Path]:
    rel = Path("Datasets") / "UIPRMD"
    candidates = [
        rel,
        Path.cwd().parent / rel,
        Path(__file__).resolve().parents[2] / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _build_reference_correct_sequence(exercise_id: int, target_len: int) -> Optional[np.ndarray]:
    dataset_root = _resolve_uiprmd_root()
    if dataset_root is None:
        logger.warning("Could not locate Datasets/UIPRMD. Reference overlay disabled.")
        return None

    try:
        loader = MediaPipeUIsprmdLoader(str(dataset_root))
        coords, labels, _ = loader.load_all(exercise_id=exercise_id)
    except Exception as e:
        logger.warning(f"Failed loading reference data: {e}")
        return None

    if len(coords) == 0:
        return None

    mask = labels == 0
    if not np.any(mask):
        logger.warning("No correct samples found for requested exercise.")
        return None

    reference = coords[mask].mean(axis=0).astype(np.float32)
    reference = _resize_time(reference, target_len)
    return reference


def _align_reference_to_user(reference_frame: np.ndarray, user_frame: np.ndarray) -> Optional[np.ndarray]:
    """Align reference skeleton to user via similarity transform on torso anchors."""
    anchor_ids = [11, 12, 25, 26]
    if reference_frame.shape[0] < 27 or user_frame.shape[0] < 27:
        return None

    ref_xy = reference_frame[:, :2].astype(np.float32)
    user_xy = user_frame[:, :2].astype(np.float32)

    valid = []
    for idx in anchor_ids:
        x, y = user_xy[idx]
        if x > 0 and y > 0:
            valid.append(idx)

    if len(valid) < 2:
        return None

    ref_sel = ref_xy[valid]
    user_sel = user_xy[valid]

    ref_ctr = ref_sel.mean(axis=0, keepdims=True)
    user_ctr = user_sel.mean(axis=0, keepdims=True)
    x = ref_sel - ref_ctr
    y = user_sel - user_ctr

    ref_norm = float(np.linalg.norm(x))
    user_norm = float(np.linalg.norm(y))
    if ref_norm < 1e-6 or user_norm < 1e-6:
        return None

    h = x.T @ y
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    scale = user_norm / ref_norm

    aligned_xy = scale * ((ref_xy - ref_ctr) @ r) + user_ctr
    aligned = np.zeros_like(user_frame, dtype=np.float32)
    aligned[:, :2] = aligned_xy
    return aligned


def run_live(args: argparse.Namespace) -> None:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as e:
        raise ImportError("Live mode requires opencv-python and mediapipe in the active environment.") from e

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    inference = PyramidSTGATInference(args.checkpoint, device=args.device)
    target_len = int(inference.config.seq_length)
    target_joints = int(inference.config.num_joints)

    if target_joints != 33:
        logger.warning(
            "Checkpoint expects %d joints. Live MediaPipe provides 33 joints; will adapt automatically.",
            target_joints,
        )

    pose_model_path = _resolve_pose_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=args.min_detection_conf,
        min_pose_presence_confidence=args.min_presence_conf,
        min_tracking_confidence=args.min_tracking_conf,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    reference_seq = None
    if args.show_reference and target_joints == 33:
        reference_seq = _build_reference_correct_sequence(args.exercise, target_len)
        if reference_seq is not None:
            logger.info("Reference overlay enabled (cyan): mean correct trajectory")

    frame_buffer = deque(maxlen=max(args.window_len, 2))
    pred_label = "Warming up"
    quality_score = 0.0
    confidence = 0.0
    quality_category = "unknown"
    feedback = "Collecting frames..."

    frame_idx = 0
    t0 = time.time()
    fps_smoothed = 0.0

    def in_bounds(x: int, y: int, w: int, h: int) -> bool:
        return 0 <= x < w and 0 <= y < h

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Webcam frame read failed; stopping live session.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            pose_arr = np.zeros((33, 3), dtype=np.float32)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                for idx, lm in enumerate(landmarks[:33]):
                    pose_arr[idx, 0] = float(lm.x) * w
                    pose_arr[idx, 1] = float(lm.y) * h
                    pose_arr[idx, 2] = float(lm.z) * w

            frame_buffer.append(pose_arr)

            # Draw user skeleton (yellow)
            for a, b in SKELETON_EDGES_33:
                xa, ya = int(pose_arr[a, 0]), int(pose_arr[a, 1])
                xb, yb = int(pose_arr[b, 0]), int(pose_arr[b, 1])
                if in_bounds(xa, ya, w, h) and in_bounds(xb, yb, w, h):
                    cv2.line(frame, (xa, ya), (xb, yb), (255, 200, 80), 2, cv2.LINE_AA)
            for j in range(33):
                xj, yj = int(pose_arr[j, 0]), int(pose_arr[j, 1])
                if in_bounds(xj, yj, w, h):
                    cv2.circle(frame, (xj, yj), 3, (80, 240, 255), -1, cv2.LINE_AA)

            # Run model every stride frames once enough buffer is available.
            if len(frame_buffer) >= max(args.min_frames_for_prediction, 2) and (frame_idx % max(args.infer_stride, 1) == 0):
                seq = np.array(frame_buffer, dtype=np.float32)
                seq = _adapt_joint_count(seq, target_joints)
                seq = _resize_time(seq, target_len)

                res = inference.predict_from_sequence(seq)
                pred_label = str(res["prediction_label"])
                quality_score = float(res["quality_score"])
                confidence = float(res["confidence"])
                quality_category, feedback = QualityScoreInterpreter.interpret(quality_score)

            # Draw reference overlay (cyan), temporally matched to buffer progress.
            if reference_seq is not None and len(frame_buffer) > 0:
                ref_idx = int((len(frame_buffer) - 1) / max(1, frame_buffer.maxlen - 1) * (target_len - 1))
                ref_frame = reference_seq[ref_idx]
                aligned = _align_reference_to_user(ref_frame, pose_arr)
                if aligned is not None:
                    for a, b in SKELETON_EDGES_33:
                        xa, ya = int(aligned[a, 0]), int(aligned[a, 1])
                        xb, yb = int(aligned[b, 0]), int(aligned[b, 1])
                        if in_bounds(xa, ya, w, h) and in_bounds(xb, yb, w, h):
                            cv2.line(frame, (xa, ya), (xb, yb), (70, 210, 255), 2, cv2.LINE_AA)
                    for j in range(33):
                        xr, yr = int(aligned[j, 0]), int(aligned[j, 1])
                        if in_bounds(xr, yr, w, h):
                            cv2.circle(frame, (xr, yr), 2, (70, 210, 255), -1, cv2.LINE_AA)

            # Heads-up display
            now = time.time()
            dt = max(now - t0, 1e-6)
            inst_fps = 1.0 / dt
            fps_smoothed = inst_fps if fps_smoothed == 0.0 else (0.9 * fps_smoothed + 0.1 * inst_fps)
            t0 = now

            is_correct = pred_label.lower().startswith("correct")
            status_color = (60, 170, 70) if is_correct else (40, 40, 220)

            overlay = frame.copy()
            cv2.rectangle(overlay, (12, 12), (w - 12, 150), status_color, -1)
            frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)

            cv2.putText(frame, f"Live Assessment: {pred_label}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78, status_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Score: {quality_score:.3f}  Conf: {confidence:.2%}  FPS: {fps_smoothed:.1f}", (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Quality: {quality_category}", (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Feedback: {feedback[:72]}", (24, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            if reference_seq is not None:
                cv2.putText(frame, "Legend: Your pose=yellow | Reference=cyan", (24, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Press q to quit, r to reset window buffer", (24, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)

            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                frame_buffer.clear()
                pred_label = "Warming up"
                quality_score = 0.0
                confidence = 0.0
                quality_category = "unknown"
                feedback = "Buffer reset. Collecting frames..."

            frame_idx += 1
    finally:
        detector.close()
        cap.release()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live webcam prediction with Temporal Pyramid STGAT")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./weights/pyramid_stgat_exercise_0_best_best_acc.pt",
        help="Path to checkpoint",
    )
    parser.add_argument("--exercise", type=int, default=0, help="Exercise ID for reference overlay")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index for OpenCV")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--window-len", type=int, default=240, help="Rolling frame buffer size")
    parser.add_argument("--infer-stride", type=int, default=8, help="Run model every N frames")
    parser.add_argument("--min-frames-for-prediction", type=int, default=45, help="Minimum buffer frames before first prediction")
    parser.add_argument("--show-reference", action="store_true", help="Overlay expected reference skeleton (cyan)")
    parser.add_argument("--window-name", type=str, default="KinetoCheck Live")
    parser.add_argument("--min-detection-conf", type=float, default=0.5)
    parser.add_argument("--min-presence-conf", type=float, default=0.5)
    parser.add_argument("--min-tracking-conf", type=float, default=0.5)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_live(args)


if __name__ == "__main__":
    main()
