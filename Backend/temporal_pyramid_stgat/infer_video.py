"""
Video assessment with Temporal Pyramid STGAT deployment checkpoint.

Pipeline:
1) Extract pose keypoints from video (YOLO or MediaPipe via existing backend factory)
2) Adapt keypoints to model input shape (joint count and coordinate dims)
3) Evaluate sliding windows and save per-window CSV
4) Aggregate to final video-level result and save summary CSV

Usage:
    d:/Programming/KinetoCheck/.venv312/Scripts/python.exe -m temporal_pyramid_stgat.infer_video \
        --video D:/Programming/KinetoCheck/Video-kineto/my_video.mp4 \
        --exercise 0
"""

from __future__ import annotations

import os
import csv
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add Backend root so absolute imports resolve when running this file directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from temporal_pyramid_stgat.inference import PyramidSTGATInference
from temporal_pyramid_stgat.preprocessing.mediapipe_angle_calculator import MediaPipeAngleCalculator
from temporal_pyramid_stgat.preprocessing.mediapipe_uiprmd_loader import MediaPipeUIsprmdLoader
from temporal_pyramid_stgat.utils.inference_utils import QualityScoreInterpreter


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Prediction-time compatibility projection for MediaPipe33 checkpoints trained
# through the Vicon->MediaPipe heuristic mapper. target_idx <- source_idx.
_MEDIAPIPE33_TRAINING_LAYOUT_FROM_RAW = {
    0: 0,    # Nose
    11: 12,  # Right shoulder
    12: 11,  # Left shoulder
    13: 14,  # Right elbow
    14: 13,  # Left elbow
    15: 16,  # Right wrist
    16: 15,  # Left wrist
    25: 24,  # Right hip
    26: 23,  # Left hip
    27: 26,  # Right knee
    28: 25,  # Left knee
    29: 28,  # Right ankle
    30: 27,  # Left ankle
}


def _default_checkpoint(exercise_id: int) -> str:
    return str(Path("temporal_pyramid_stgat") / "weights" / f"pyramid_stgat_exercise_{exercise_id}_deployment.pt")


def _read_video_meta(video_path: str) -> Dict[str, float]:
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV (cv2) is not installed. FPS/frame metadata will be unavailable.")
        return {'fps': 0.0, 'frames': 0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'fps': 0.0, 'frames': 0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {'fps': fps, 'frames': frames}


def _extract_keypoints_yolo(video_path: str) -> np.ndarray:
    """Extract (T, 17, 2) keypoints using YOLOv8 pose without app-layer imports."""
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "YOLO extraction requires opencv-python and ultralytics in the active environment."
        ) from e

    model = YOLO("yolov8n-pose.pt", task="pose")
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(frame, task="pose", verbose=False, half=False)
            if results and results[0].keypoints is not None:
                kps = results[0].keypoints.xy.cpu().numpy()
                if len(kps) > 0:
                    frames.append(kps[0][:17, :2].astype(np.float32))
                    continue
            frames.append(np.zeros((17, 2), dtype=np.float32))
    finally:
        cap.release()

    return np.array(frames, dtype=np.float32)


def _resolve_mediapipe_pose_model() -> str:
    """Return local path to MediaPipe Pose Landmarker model, downloading if needed."""
    model_dir = Path("temporal_pyramid_stgat") / "weights"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pose_landmarker_lite.task"
    if model_path.exists():
        return str(model_path)

    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    logger.info(f"Downloading MediaPipe Pose model: {url}")
    import urllib.request

    urllib.request.urlretrieve(url, str(model_path))
    return str(model_path)


def _extract_keypoints_mediapipe(video_path: str) -> np.ndarray:
    """Extract (T, 33, 3) keypoints using MediaPipe Tasks Pose Landmarker."""
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as e:
        raise ImportError(
            "MediaPipe extraction requires mediapipe and opencv-python in the active environment."
        ) from e

    model_path = _resolve_mediapipe_pose_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            h, w = frame.shape[:2]
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                arr = np.zeros((33, 3), dtype=np.float32)
                for idx, lm in enumerate(landmarks[:33]):
                    arr[idx, 0] = float(lm.x) * w
                    arr[idx, 1] = float(lm.y) * h
                    arr[idx, 2] = float(lm.z) * w
                frames.append(arr)
            else:
                frames.append(np.zeros((33, 3), dtype=np.float32))
    finally:
        cap.release()
        detector.close()

    return np.array(frames, dtype=np.float32)


def _extract_keypoints(video_path: str, pose_backend: str) -> np.ndarray:
    backend = (pose_backend or 'yolo').lower()
    if backend == 'yolo':
        return _extract_keypoints_yolo(video_path)
    if backend == 'mediapipe':
        return _extract_keypoints_mediapipe(video_path)
    raise ValueError(f"Unsupported pose backend: {pose_backend}")


def _to_3d(keypoints: np.ndarray) -> np.ndarray:
    """Ensure (T, J, 3)."""
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected keypoints shape (T, J, C), got {arr.shape}")
    if arr.shape[2] == 3:
        return arr
    if arr.shape[2] == 2:
        z = np.zeros((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
        return np.concatenate([arr, z], axis=2)
    raise ValueError(f"Expected keypoint dim 2 or 3, got {arr.shape[2]}")


def _project_mediapipe33_to_training_layout(seq: np.ndarray) -> np.ndarray:
    """
    Project raw MediaPipe landmarks into the sparse 33-slot layout expected by
    mapper-trained checkpoints.

    Raw MediaPipe extraction provides dense canonical indices. The 33-joint
    training pipeline was built from a sparse heuristic Vicon->MediaPipe map,
    so we mirror that slot layout at inference to reduce domain mismatch.
    """
    if seq.ndim != 3 or seq.shape[1] < 33:
        return seq

    out = np.zeros((seq.shape[0], 33, seq.shape[2]), dtype=np.float32)
    for target_idx, source_idx in _MEDIAPIPE33_TRAINING_LAYOUT_FROM_RAW.items():
        out[:, target_idx, :] = seq[:, source_idx, :]
    return out


def _resize_time(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linear temporal interpolation to (target_len, J, C)."""
    T, J, C = seq.shape
    if T == target_len:
        return seq.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, J, C), dtype=np.float32)
    for j in range(J):
        for c in range(C):
            out[:, j, c] = np.interp(x_new, x_old, seq[:, j, c])
    return out


def _adapt_joint_count(seq: np.ndarray, target_joints: int) -> np.ndarray:
    """Pad/truncate joints to model expected joint count."""
    T, J, C = seq.shape
    if J == target_joints:
        return seq
    if J > target_joints:
        return seq[:, :target_joints, :]
    pad = np.zeros((T, target_joints - J, C), dtype=np.float32)
    return np.concatenate([seq, pad], axis=1)


def _sliding_windows(total_frames: int, window: int, stride: int) -> List[Tuple[int, int]]:
    if total_frames <= window:
        return [(0, total_frames)]
    spans = []
    start = 0
    while start + window <= total_frames:
        spans.append((start, start + window))
        start += stride
    if spans and spans[-1][1] < total_frames:
        spans.append((total_frames - window, total_frames))
    return spans


def _write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_uiprmd_root() -> Optional[Path]:
    """Resolve UI-PRMD dataset root from common workspace layouts."""
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


def _build_reference_correct_sequence(
    exercise_id: int,
    target_frames: int,
    target_joints: int,
) -> Optional[np.ndarray]:
    """Build mean reference trajectory from correct training samples."""
    dataset_root = _resolve_uiprmd_root()
    if dataset_root is None:
        logger.warning("Could not locate Datasets/UIPRMD. Reference overlay disabled.")
        return None

    try:
        loader = MediaPipeUIsprmdLoader(str(dataset_root))
        coords, labels, _ = loader.load_all(exercise_id=exercise_id)
    except Exception as e:
        logger.warning(f"Failed loading reference data for overlay: {e}")
        return None

    if len(coords) == 0:
        return None

    correct_mask = labels == 0
    if not np.any(correct_mask):
        logger.warning("No correct samples found for requested exercise. Reference overlay disabled.")
        return None

    ref_seq = coords[correct_mask].mean(axis=0).astype(np.float32)
    ref_seq = _adapt_joint_count(ref_seq, target_joints)
    ref_seq = _resize_time(ref_seq, target_frames)
    return ref_seq


def _align_reference_to_user(reference_frame: np.ndarray, user_frame: np.ndarray) -> Optional[np.ndarray]:
    """Align reference skeleton to user frame via similarity transform on torso anchors."""
    anchor_ids = [11, 12, 25, 26]  # shoulders + hips in MediaPipe indexing

    if reference_frame.shape[0] < max(anchor_ids) + 1 or user_frame.shape[0] < max(anchor_ids) + 1:
        return None

    ref_xy = reference_frame[:, :2].astype(np.float32)
    user_xy = user_frame[:, :2].astype(np.float32)

    valid = []
    for idx in anchor_ids:
        ux, uy = user_xy[idx]
        if ux > 0 and uy > 0:
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


def _phase_match_reference_indices(
    user_angles: np.ndarray,
    reference_angles: np.ndarray,
    search_radius: int = 18,
) -> np.ndarray:
    """Monotonic angle-space matching from user frames to reference frames."""
    u_len = user_angles.shape[0]
    r_len = reference_angles.shape[0]
    if u_len == 0 or r_len == 0:
        return np.zeros((u_len,), dtype=np.int64)

    indices = np.zeros((u_len,), dtype=np.int64)
    prev = 0
    for i in range(u_len):
        expected = int(i * (r_len - 1) / max(u_len - 1, 1))
        lo = max(prev, expected - search_radius)
        hi = min(r_len - 1, expected + search_radius)

        if lo > hi:
            lo = prev
            hi = min(r_len - 1, max(prev, expected))

        candidates = reference_angles[lo:hi + 1]  # (K, A)
        diffs = np.mean(np.abs(candidates - user_angles[i]), axis=1)
        best = lo + int(np.argmin(diffs))
        indices[i] = best
        prev = best

    return indices


def _top_angle_deviation_text(
    user_angles: np.ndarray,
    reference_angles: np.ndarray,
    frame_to_ref_idx: np.ndarray,
    start_frame: int,
    end_frame: int,
    top_k: int = 3,
) -> List[str]:
    """Return top-k mean absolute angle deviations for a frame range."""
    if user_angles.size == 0 or reference_angles.size == 0 or len(frame_to_ref_idx) == 0:
        return ["", "", ""]

    s = max(0, int(start_frame))
    e = min(int(end_frame), user_angles.shape[0])
    if s >= e:
        return ["", "", ""]

    user_seg = user_angles[s:e]
    ref_idx_seg = frame_to_ref_idx[s:e]
    ref_seg = reference_angles[ref_idx_seg]

    mad = np.mean(np.abs(user_seg - ref_seg), axis=0)  # radians
    top = np.argsort(mad)[::-1][:top_k]

    lines = []
    for idx in top:
        name = MediaPipeAngleCalculator.ANGLE_NAMES[int(idx)]
        deg = float(np.degrees(mad[int(idx)]))
        lines.append(f"{name}: {deg:.1f}deg")

    while len(lines) < top_k:
        lines.append("")

    return lines


def _annotate_video(
    video_path: str,
    output_path: str,
    window_rows: List[Dict],
    summary: Dict,
    raw_keypoints: np.ndarray,
    reference_keypoints: Optional[np.ndarray] = None,
    reference_frame_indices: Optional[np.ndarray] = None,
) -> str:
    """Render a video overlay showing per-frame assessment status from window scores."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV is required for annotated video output.") from e

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for annotation: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(str(Path(output_path).parent), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    parsed = []
    for w in window_rows:
        parsed.append({
            'start': int(w['start_frame']),
            'end': int(w['end_frame']),
            'prediction': int(w['prediction']),
            'quality_score': float(w['quality_score']),
            'confidence': float(w['confidence']),
            'feedback': str(w['feedback']),
            'deviation': str(w.get('key_deviation_summary', '')),
        })

    def _in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    # Use COCO-17 edges when 17 joints are present, otherwise use a compact
    # MediaPipe-33 body chain subset for visualization.
    if raw_keypoints.shape[1] >= 33:
        skeleton_edges = [
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (11, 12),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27),
            (24, 26), (26, 28),
            (0, 1), (1, 2), (2, 3),
            (0, 4), (4, 5), (5, 6),
        ]
    else:
        skeleton_edges = [
            (5, 7), (7, 9),      # left arm
            (6, 8), (8, 10),     # right arm
            (5, 6),              # shoulders
            (5, 11), (6, 12),    # torso sides
            (11, 12),            # hips
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
            (0, 1), (0, 2),      # face
            (1, 3), (2, 4),      # face
        ]

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            active = [w for w in parsed if w['start'] <= frame_idx < w['end']]
            if active:
                mean_score = float(np.mean([w['quality_score'] for w in active]))
                mean_conf = float(np.mean([w['confidence'] for w in active]))
                pred = int(round(float(np.mean([w['prediction'] for w in active]))))
                feedback = active[int(np.argmin([w['quality_score'] for w in active]))]['feedback']
                deviation = active[int(np.argmin([w['quality_score'] for w in active]))]['deviation']
            else:
                mean_score = float(summary['quality_score_mean'])
                mean_conf = float(summary['confidence_mean'])
                pred = int(summary['prediction'])
                feedback = str(summary['feedback'])
                deviation = str(summary.get('key_deviation_summary', ''))

            label = "Correct" if pred == 0 else "Incorrect"
            color = (60, 170, 70) if pred == 0 else (40, 40, 220)

            # Draw pose if keypoints are available for this frame.
            if frame_idx < raw_keypoints.shape[0]:
                kps = raw_keypoints[frame_idx]
                for a, b in skeleton_edges:
                    if a < kps.shape[0] and b < kps.shape[0]:
                        xa, ya = int(kps[a, 0]), int(kps[a, 1])
                        xb, yb = int(kps[b, 0]), int(kps[b, 1])
                        if _in_bounds(xa, ya) and _in_bounds(xb, yb):
                            cv2.line(frame, (xa, ya), (xb, yb), (255, 200, 80), 2, cv2.LINE_AA)
                for j in range(kps.shape[0]):
                    x, y = int(kps[j, 0]), int(kps[j, 1])
                    if _in_bounds(x, y):
                        cv2.circle(frame, (x, y), 3, (80, 240, 255), -1, cv2.LINE_AA)

            # Draw aligned reference (expected) skeleton in a different color.
            if reference_keypoints is not None and frame_idx < reference_keypoints.shape[0]:
                ref_idx = frame_idx
                if reference_frame_indices is not None and frame_idx < len(reference_frame_indices):
                    ref_idx = int(reference_frame_indices[frame_idx])
                ref_idx = max(0, min(ref_idx, reference_keypoints.shape[0] - 1))

                ref_frame = _align_reference_to_user(reference_keypoints[ref_idx], raw_keypoints[frame_idx])
                if ref_frame is not None:
                    for a, b in skeleton_edges:
                        if a < ref_frame.shape[0] and b < ref_frame.shape[0]:
                            xa, ya = int(ref_frame[a, 0]), int(ref_frame[a, 1])
                            xb, yb = int(ref_frame[b, 0]), int(ref_frame[b, 1])
                            if _in_bounds(xa, ya) and _in_bounds(xb, yb):
                                cv2.line(frame, (xa, ya), (xb, yb), (70, 210, 255), 2, cv2.LINE_AA)
                    for j in range(ref_frame.shape[0]):
                        x, y = int(ref_frame[j, 0]), int(ref_frame[j, 1])
                        if _in_bounds(x, y):
                            cv2.circle(frame, (x, y), 2, (70, 210, 255), -1, cv2.LINE_AA)

            overlay = frame.copy()
            cv2.rectangle(overlay, (12, 12), (width - 12, 128), color, -1)
            frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)

            cv2.putText(frame, f"Assessment: {label}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Quality: {mean_score:.3f}  Confidence: {mean_conf:.2%}", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Feedback: {feedback[:80]}", (24, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            if deviation:
                cv2.putText(frame, f"Main deviation: {deviation[:92]}", (24, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 245, 180), 1, cv2.LINE_AA)
            if reference_keypoints is not None:
                cv2.putText(frame, "Legend: Your pose=yellow | Reference=cyan", (24, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return output_path


def assess_video(
    video_path: str,
    checkpoint_path: str,
    output_dir: str,
    pose_backend: str,
    window_len: int,
    stride: int,
    save_annotated_video: bool,
    exercise_id: int,
    mediapipe_layout: str,
) -> Dict[str, str]:
    """Run full video assessment and save CSV outputs."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    meta = _read_video_meta(video_path)
    fps = float(meta['fps']) if meta['fps'] else 0.0

    logger.info(f"Loading inference model from {checkpoint_path}")
    inference = PyramidSTGATInference(checkpoint_path, device="cuda")

    target_joints = int(inference.config.num_joints)
    target_len = int(inference.config.seq_length)

    logger.info(f"Extracting keypoints with backend: {pose_backend}")
    raw_kps = _extract_keypoints(video_path, pose_backend)
    raw_kps = _to_3d(raw_kps)
    logger.info(f"Raw keypoints shape: {raw_kps.shape}")

    model_kps = raw_kps
    if pose_backend.lower() == 'mediapipe' and target_joints == 33:
        layout = (mediapipe_layout or 'projected').lower()
        if layout == 'projected':
            model_kps = _project_mediapipe33_to_training_layout(raw_kps)
            logger.info(
                "Applied MediaPipe33 compatibility projection to mirror mapper-trained layout "
                "(sparse slot remap)."
            )
        elif layout == 'raw':
            logger.info("Using raw MediaPipe33 landmark ordering (no compatibility projection).")
        else:
            raise ValueError(f"Unsupported mediapipe layout mode: {mediapipe_layout}")

    reference_keypoints = None
    reference_angles = None
    user_angles = None
    reference_frame_indices = None
    if pose_backend.lower() == 'mediapipe' and target_joints == 33:
        reference_keypoints = _build_reference_correct_sequence(
            exercise_id=exercise_id,
            target_frames=model_kps.shape[0],
            target_joints=target_joints,
        )
        if reference_keypoints is not None:
            try:
                user_angles = MediaPipeAngleCalculator.extract_angles(model_kps)
                reference_angles = MediaPipeAngleCalculator.extract_angles(reference_keypoints)
                reference_frame_indices = _phase_match_reference_indices(user_angles, reference_angles)
                logger.info("Reference overlay enabled (phase-matched mean correct trajectory)")
            except Exception as e:
                logger.warning(f"Phase-matching failed; using direct frame mapping. Reason: {e}")
                reference_frame_indices = None

    extracted_joints = int(model_kps.shape[1])
    if extracted_joints != target_joints:
        logger.warning(
            "Joint-count mismatch: model expects %d joints but extractor produced %d. "
            "Input will be padded/truncated, which can strongly degrade quality score.",
            target_joints,
            extracted_joints,
        )

    spans = _sliding_windows(model_kps.shape[0], max(window_len, 2), max(stride, 1))
    logger.info(f"Scoring {len(spans)} windows")

    window_rows = []
    labels = []
    scores = []
    confs = []

    for idx, (s, e) in enumerate(spans, start=1):
        seq = model_kps[s:e]
        seq = _adapt_joint_count(seq, target_joints)
        seq = _resize_time(seq, target_len)

        result = inference.predict_from_sequence(seq)
        category, feedback = QualityScoreInterpreter.interpret(float(result['quality_score']))

        labels.append(int(result['prediction']))
        scores.append(float(result['quality_score']))
        confs.append(float(result['confidence']))

        window_rows.append({
            'window_id': idx,
            'start_frame': s,
            'end_frame': e,
            'start_sec': (s / fps) if fps > 0 else '',
            'end_sec': (e / fps) if fps > 0 else '',
            'prediction': int(result['prediction']),
            'prediction_label': result['prediction_label'],
            'quality_score': float(result['quality_score']),
            'confidence': float(result['confidence']),
            'quality_category': category,
            'feedback': feedback,
            'deviation_1': '',
            'deviation_2': '',
            'deviation_3': '',
            'key_deviation_summary': '',
        })

        if user_angles is not None and reference_angles is not None and reference_frame_indices is not None:
            dev = _top_angle_deviation_text(
                user_angles=user_angles,
                reference_angles=reference_angles,
                frame_to_ref_idx=reference_frame_indices,
                start_frame=s,
                end_frame=e,
                top_k=3,
            )
            window_rows[-1]['deviation_1'] = dev[0]
            window_rows[-1]['deviation_2'] = dev[1]
            window_rows[-1]['deviation_3'] = dev[2]
            window_rows[-1]['key_deviation_summary'] = dev[0]

    # Aggregate video-level result
    mean_score = float(np.mean(scores)) if scores else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0

    # Majority vote on labels (0=Correct, 1=Incorrect)
    if labels:
        pred_label = int(round(float(np.mean(labels))))
    else:
        pred_label = 1
    pred_label_str = 'Correct' if pred_label == 0 else 'Incorrect'
    category, feedback = QualityScoreInterpreter.interpret(mean_score)

    summary_row = [{
        'video_path': str(Path(video_path).resolve()),
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        'pose_backend': pose_backend,
        'mediapipe_layout': mediapipe_layout if pose_backend.lower() == 'mediapipe' else '',
        'frames_total': int(model_kps.shape[0]),
        'fps': fps,
        'num_windows': len(window_rows),
        'window_len': window_len,
        'stride': stride,
        'prediction': pred_label,
        'prediction_label': pred_label_str,
        'quality_score_mean': mean_score,
        'confidence_mean': mean_conf,
        'quality_category': category,
        'feedback': feedback,
        'key_deviation_summary': (window_rows[int(np.argmin(scores))].get('key_deviation_summary', '') if scores else ''),
    }]

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(video_path).stem
    window_csv = str(Path(output_dir) / f"{stem}_window_metrics.csv")
    summary_csv = str(Path(output_dir) / f"{stem}_summary.csv")

    _write_csv(window_csv, window_rows)
    _write_csv(summary_csv, summary_row)

    annotated_video_path = ''
    if save_annotated_video:
        annotated_video_path = str(Path(output_dir) / f"{stem}_annotated.mp4")
        _annotate_video(
            video_path,
            annotated_video_path,
            window_rows,
            summary_row[0],
            raw_kps,
            reference_keypoints=reference_keypoints,
            reference_frame_indices=reference_frame_indices,
        )
        logger.info(f"Saved annotated video: {annotated_video_path}")

    logger.info(f"Saved window metrics CSV: {window_csv}")
    logger.info(f"Saved summary CSV: {summary_csv}")
    logger.info(
        f"Final video assessment -> {pred_label_str} | score={mean_score:.3f} | conf={mean_conf:.2%}"
    )

    return {
        'window_csv': window_csv,
        'summary_csv': summary_csv,
        'annotated_video': annotated_video_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Assess a video with deployment checkpoint and export CSV metrics")
    parser.add_argument('--video', type=str, default="../Video-kineto/deep_squat_one_rep.mp4", help='Path to input video')
    parser.add_argument('--exercise', type=int, default=0, help='Exercise ID used to locate default deployment checkpoint')
    parser.add_argument('--checkpoint', type=str, default="./temporal_pyramid_stgat/weights/pyramid_stgat_mediapipe33_exercise_0_best_best_acc.pt", help='Optional checkpoint override')
    parser.add_argument('--output-dir', type=str, default='temporal_pyramid_stgat/outputs/video_assessment')
    parser.add_argument('--pose-backend', type=str, default='mediapipe', choices=['yolo', 'mediapipe'])
    parser.add_argument(
        '--mediapipe-layout',
        type=str,
        default='projected',
        choices=['projected', 'raw'],
        help='For 33-joint MediaPipe checkpoints: projected remaps landmarks into mapper-trained sparse layout; raw keeps native MediaPipe ordering.',
    )
    parser.add_argument('--window-len', type=int, default=240, help='Sliding window length in frames')
    parser.add_argument('--stride', type=int, default=60, help='Sliding stride in frames')
    parser.add_argument('--save-annotated-video', action='store_true', help='Render an annotated output video with feedback overlay')
    args = parser.parse_args()

    ckpt = args.checkpoint or _default_checkpoint(args.exercise)
    assess_video(
        video_path=args.video,
        checkpoint_path=ckpt,
        output_dir=args.output_dir,
        pose_backend=args.pose_backend,
        window_len=args.window_len,
        stride=args.stride,
        save_annotated_video=args.save_annotated_video,
        exercise_id=args.exercise,
        mediapipe_layout=args.mediapipe_layout,
    )


if __name__ == '__main__':
    main()
