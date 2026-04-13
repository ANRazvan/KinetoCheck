#!/usr/bin/env python3
"""
Generate skeleton-only reference videos from UI-PRMD correct samples.

Outputs are rendered on a clean canvas (no RGB background) so you can inspect
how movement should look according to dataset-derived trajectories.

Usage:
  python -m temporal_pyramid_stgat.generate_reference_videos --exercise 0
  python -m temporal_pyramid_stgat.generate_reference_videos --exercise 0 --num-samples 5
  python -m temporal_pyramid_stgat.generate_reference_videos --all-exercises
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Add Backend root so absolute imports resolve when running this file directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from temporal_pyramid_stgat.preprocessing.mediapipe_uiprmd_loader import MediaPipeUIsprmdLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Stable subset of MediaPipe edges that are reliably populated by our
# Vicon->MediaPipe mapping and suitable for clean reference rendering.
REFERENCE_EDGES = [
    # Upper body
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    # Torso + hips
    (11, 25), (12, 26), (25, 26),
    # Legs
    (25, 27), (27, 29),
    (26, 28), (28, 30),
    # Head anchor
    (0, 11), (0, 12),
]


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


def _normalize_to_canvas(seq: np.ndarray, width: int, height: int, margin: int = 50) -> np.ndarray:
    """Map raw coordinates to canvas coordinates with stable sequence-level scaling."""
    # Unmapped joints are zeros in the Vicon->MediaPipe conversion. Treat them as invalid.
    pts = seq[:, :, :2].reshape(-1, 2)
    valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-6)
    pts = pts[valid]

    if pts.shape[0] == 0:
        return np.full_like(seq[:, :, :2], np.nan, dtype=np.float32)

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    # Fit while preserving aspect ratio.
    sx = (width - 2 * margin) / span[0]
    sy = (height - 2 * margin) / span[1]
    scale = float(min(sx, sy))

    centered = seq[:, :, :2] - min_xy
    canvas = centered * scale

    # Center the drawing in the available canvas area.
    content_w = span[0] * scale
    content_h = span[1] * scale
    off_x = (width - content_w) * 0.5
    off_y = (height - content_h) * 0.5

    canvas[:, :, 0] += off_x
    canvas[:, :, 1] += off_y

    # Flip vertical axis for a natural upright view.
    canvas[:, :, 1] = height - canvas[:, :, 1]

    invalid = np.linalg.norm(seq[:, :, :2], axis=2) <= 1e-6
    canvas[invalid] = np.nan

    return canvas.astype(np.float32)


def _render_sequence_video(
    seq: np.ndarray,
    output_path: Path,
    title: str,
    fps: int = 30,
    width: int = 960,
    height: int = 960,
) -> None:
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python is required to render reference videos.") from e

    points = _normalize_to_canvas(seq, width=width, height=height)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    try:
        for t in range(points.shape[0]):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Subtle background grid for depth/orientation cues.
            for gy in range(80, height, 80):
                cv2.line(frame, (0, gy), (width, gy), (24, 24, 24), 1, cv2.LINE_AA)
            for gx in range(80, width, 80):
                cv2.line(frame, (gx, 0), (gx, height), (24, 24, 24), 1, cv2.LINE_AA)

            kps = points[t]
            for a, b in REFERENCE_EDGES:
                if a < kps.shape[0] and b < kps.shape[0]:
                    if not np.isfinite(kps[a]).all() or not np.isfinite(kps[b]).all():
                        continue
                    xa, ya = int(kps[a, 0]), int(kps[a, 1])
                    xb, yb = int(kps[b, 0]), int(kps[b, 1])
                    if 0 <= xa < width and 0 <= ya < height and 0 <= xb < width and 0 <= yb < height:
                        cv2.line(frame, (xa, ya), (xb, yb), (80, 210, 255), 2, cv2.LINE_AA)

            for j in range(kps.shape[0]):
                if not np.isfinite(kps[j]).all():
                    continue
                xj, yj = int(kps[j, 0]), int(kps[j, 1])
                if 0 <= xj < width and 0 <= yj < height:
                    cv2.circle(frame, (xj, yj), 3, (0, 255, 255), -1, cv2.LINE_AA)

            cv2.putText(frame, title[:90], (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame {t + 1}/{points.shape[0]}", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)

            writer.write(frame)
    finally:
        writer.release()


def _select_representative_indices(correct_seqs: np.ndarray, num_samples: int) -> List[int]:
    """Pick representative samples closest to mean trajectory (stable references)."""
    n = correct_seqs.shape[0]
    if n == 0 or num_samples <= 0:
        return []

    mean_seq = correct_seqs.mean(axis=0, keepdims=True)
    dists = np.mean(np.abs(correct_seqs - mean_seq), axis=(1, 2, 3))
    order = np.argsort(dists)

    k = min(num_samples, n)
    return [int(i) for i in order[:k]]


def _load_exercise_data(loader: MediaPipeUIsprmdLoader, exercise_id: Optional[int]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    coords, labels, metadata = loader.load_all(exercise_id=exercise_id)
    return coords.astype(np.float32), labels.astype(np.int64), metadata


def generate_for_exercise(
    loader: MediaPipeUIsprmdLoader,
    output_dir: Path,
    exercise_id: int,
    num_samples: int,
    fps: int,
) -> List[Path]:
    coords, labels, _ = _load_exercise_data(loader, exercise_id)
    correct = coords[labels == 0]

    if correct.shape[0] == 0:
        logger.warning(f"Exercise {exercise_id}: no correct samples found, skipping.")
        return []

    saved: List[Path] = []

    mean_seq = correct.mean(axis=0)
    mean_path = output_dir / f"exercise_{exercise_id}_reference_mean.mp4"
    _render_sequence_video(mean_seq, mean_path, title=f"Exercise {exercise_id} - Mean Correct Reference", fps=fps)
    saved.append(mean_path)

    reps = _select_representative_indices(correct, num_samples=num_samples)
    for rank, idx in enumerate(reps, start=1):
        seq = correct[idx]
        out = output_dir / f"exercise_{exercise_id}_reference_sample_{rank}.mp4"
        _render_sequence_video(seq, out, title=f"Exercise {exercise_id} - Correct Sample {rank}", fps=fps)
        saved.append(out)

    logger.info(
        f"Exercise {exercise_id}: generated {len(saved)} reference videos "
        f"from {correct.shape[0]} correct samples"
    )
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate skeleton-only reference videos from UI-PRMD")
    parser.add_argument("--exercise", type=int, default=0, help="Exercise ID (0-based)")
    parser.add_argument("--all-exercises", action="store_true", help="Generate for all exercise IDs found in dataset")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of representative correct sample videos")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="temporal_pyramid_stgat/outputs/reference_videos",
        help="Directory for generated reference videos",
    )
    args = parser.parse_args()

    dataset_root = _resolve_uiprmd_root()
    if dataset_root is None:
        raise FileNotFoundError("Could not locate Datasets/UIPRMD from current workspace.")

    loader = MediaPipeUIsprmdLoader(str(dataset_root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exercises: Sequence[int]
    if args.all_exercises:
        coords, labels, metadata = _load_exercise_data(loader, exercise_id=None)
        _ = coords, labels
        found = sorted({int(m.get("exercise_id", -1)) for m in metadata if m.get("exercise_id") is not None})
        exercises = [e for e in found if e >= 0]
        if not exercises:
            raise RuntimeError("No exercise IDs discovered in metadata.")
    else:
        exercises = [int(args.exercise)]

    all_saved: List[Path] = []
    for ex_id in exercises:
        all_saved.extend(
            generate_for_exercise(
                loader=loader,
                output_dir=output_dir,
                exercise_id=ex_id,
                num_samples=max(0, int(args.num_samples)),
                fps=max(1, int(args.fps)),
            )
        )

    if all_saved:
        logger.info("Generated reference videos:")
        for path in all_saved:
            logger.info(f"  {path}")
    else:
        logger.warning("No videos were generated.")


if __name__ == "__main__":
    main()
