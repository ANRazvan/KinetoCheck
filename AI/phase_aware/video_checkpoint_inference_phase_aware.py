from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor
from Testing.video_checkpoint_inference_phase_aware import (
    LoadedExerciseModel,
    build_model_input,
    draw_ghost_skeleton,
    draw_skeleton,
    ensure_pose_task_model,
    extract_mediapipe_sequence,
    get_cached_models,
    load_models,
    resolve_device,
    run_prediction,
)

__all__ = [
    "LoadedExerciseModel",
    "build_model_input",
    "draw_ghost_skeleton",
    "draw_skeleton",
    "ensure_pose_task_model",
    "extract_mediapipe_sequence",
    "get_cached_models",
    "load_models",
    "process_video",
    "resolve_device",
    "run_prediction",
]


def _select_fourcc() -> tuple[int, str]:
    for codec in ("avc1", "mp4v"):
        return cv2.VideoWriter_fourcc(*codec), codec
    return cv2.VideoWriter_fourcc(*"mp4v"), "mp4v"


def _point_colors(joint_count: int, bad_idxs: set[int], imp_idxs: set[int]) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    for idx in range(joint_count):
        if idx in bad_idxs:
            colors.append((0, 60, 255))
        elif idx in imp_idxs:
            colors.append((0, 200, 255))
        else:
            colors.append((200, 200, 200))
    return colors


def _build_joint_points(frame: np.ndarray, width: int, height: int) -> list[tuple[int, int] | None]:
    points: list[tuple[int, int] | None] = []
    for joint in frame:
        x, y = float(joint[0]), float(joint[1])
        if np.isnan(x) or np.isnan(y):
            points.append(None)
            continue
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        points.append((int(x * width), int(y * height)))
    return points


def process_video(
    video_path: Path,
    models: list[LoadedExerciseModel],
    preprocessor: UIPRMDPreprocessor,
    output_dir: Path,
    device: torch.device,
    pose_model_path: Path,
    ghost_anchor: str = "hips",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_sequence = extract_mediapipe_sequence(video_path, pose_model_path)
    if raw_sequence.size == 0:
        raise RuntimeError(f"No pose frames could be extracted from {video_path}")

    in_channels = models[0].in_channels if models else 12
    model_input = build_model_input(raw_sequence, preprocessor, in_channels=in_channels)
    report, best_imp, worst_joints, phase_outputs = run_prediction(
        input_tensor=model_input,
        raw_sequence=raw_sequence,
        models=models,
        device=device,
        ghost_anchor=ghost_anchor,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_width <= 0 or frame_height <= 0:
        first_ok, first_frame = cap.read()
        if not first_ok:
            cap.release()
            raise RuntimeError(f"Cannot read video frames from {video_path}")
        frame_height, frame_width = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
    fourcc, codec = _select_fourcc()
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        codec = "mp4v"
        writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*codec), fps, (frame_width, frame_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create annotated video writer: {annotated_path}")

    bad_idxs = {int(item["joint_index"]) for item in worst_joints[:3]} if worst_joints else set()
    imp_idxs = set(int(idx) for idx in np.argsort(-best_imp)[:5].tolist())
    joint_colors = _point_colors(raw_sequence.shape[1], bad_idxs, imp_idxs)
    ghost_xyz = phase_outputs["ghost_xyz"] if phase_outputs else None

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_count >= raw_sequence.shape[0]:
            break

        user_points = _build_joint_points(raw_sequence[frame_count], frame_width, frame_height)
        draw_skeleton(
            frame,
            user_points,
            joint_colors,
            np.ones(raw_sequence.shape[1], dtype=np.float32),
            frame_width,
            bad_idxs,
            imp_idxs,
        )

        if ghost_xyz is not None and frame_count < ghost_xyz.shape[0]:
            draw_ghost_skeleton(frame, ghost_xyz[frame_count], frame_width, frame_height)

        cv2.putText(
            frame,
            f"{report['best']['exercise_name']}  score {report['best']['score']:.3f}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    feedback_summary = ", ".join(
        f"{item['joint']} ({item['deviation']:.3f})" for item in worst_joints[:3]
    ) if worst_joints else None

    result = {
        "video": str(video_path),
        "annotated_video": str(annotated_path),
        "annotated_video_codec": codec,
        "num_frames": int(raw_sequence.shape[0]),
        "has_phase_decoder": bool(report["best"].get("has_phase_decoder", False)),
        "ghost_anchor": ghost_anchor,
        "worst_joints": worst_joints,
        "best": report["best"],
        "all": report["all"],
        "annotated_video_rel": annotated_path.name,
    }
    if feedback_summary:
        result["feedback_summary"] = feedback_summary
    return result