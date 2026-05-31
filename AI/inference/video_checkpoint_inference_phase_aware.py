from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from Models.factory import ModelFactory
from Models.stgat_temporal_pyramid_phase_aware import ExerciseEvaluator
from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor, build_features_from_aligned
from visualize import extract_mediapipe_sequence

__all__ = [
    "LoadedExerciseModel",
    "build_model_input",
    "draw_ghost_skeleton",
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
COCO17_JOINT_NAMES = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10),
    (10, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass(slots=True)
class LoadedExerciseModel:
    exercise_id: int
    model: ExerciseEvaluator
    template_tensor: torch.Tensor
    template_xyz_tensor: torch.Tensor | None
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


def ensure_pose_task_model(model_path: Path | None) -> Path:
    candidate = model_path or (BASE_DIR / "pose_landmarker_full.task")
    candidate = candidate.expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Pose task model not found: {candidate}")
    return candidate


def build_model_input(raw_sequence: np.ndarray, preprocessor: UIPRMDPreprocessor, *, in_channels: int = 9) -> torch.Tensor:
    processed = preprocessor.process(raw_sequence)
    velocity = np.diff(processed, axis=0, prepend=processed[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])

    if in_channels == 12:
        features = build_features_from_aligned(processed)
    elif in_channels == 9:
        features = np.concatenate([processed, velocity, acceleration], axis=-1)
        features = np.transpose(features, (2, 0, 1)).copy().astype(np.float32)
    else:
        features = np.concatenate([processed, velocity, acceleration], axis=-1)
        if features.shape[-1] >= in_channels:
            features = features[:, :, :in_channels]
        else:
            pad_width = in_channels - features.shape[-1]
            pad = np.zeros((features.shape[0], features.shape[1], pad_width), dtype=np.float32)
            features = np.concatenate([features, pad], axis=-1)
        features = np.transpose(features, (2, 0, 1)).copy().astype(np.float32)

    return torch.from_numpy(features)


def draw_skeleton(
    frame: np.ndarray,
    points: list[tuple[int, int] | None],
    colors: list[tuple[int, int, int]],
    confidences: np.ndarray,
    width: int,
    height: int,
    bad_idxs: set[int],
    imp_idxs: set[int],
) -> None:
    del confidences, width, height, bad_idxs, imp_idxs
    for a, b in COCO17_EDGES:
        if points[a] is None or points[b] is None:
            continue
        cv2.line(frame, points[a], points[b], (80, 220, 120), 2, cv2.LINE_AA)

    for idx, point in enumerate(points):
        if point is None:
            continue
        cv2.circle(frame, point, 4, colors[idx], -1, cv2.LINE_AA)


def draw_ghost_skeleton(frame: np.ndarray, ghost_xyz: np.ndarray, width: int, height: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    for a, b in COCO17_EDGES:
        pa = ghost_xyz[a]
        pb = ghost_xyz[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        ax, ay = int(np.clip(pa[0], 0.0, 1.0) * width), int(np.clip(pa[1], 0.0, 1.0) * height)
        bx, by = int(np.clip(pb[0], 0.0, 1.0) * width), int(np.clip(pb[1], 0.0, 1.0) * height)
        cv2.line(overlay, (ax, ay), (bx, by), (255, 180, 0), 2, cv2.LINE_AA)

    for joint in ghost_xyz:
        if np.any(np.isnan(joint)):
            continue
        x = int(np.clip(joint[0], 0.0, 1.0) * width)
        y = int(np.clip(joint[1], 0.0, 1.0) * height)
        cv2.circle(overlay, (x, y), 3, (255, 220, 80), -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _build_joint_summary(joint_scores: np.ndarray) -> list[dict[str, float | int | str]]:
    ranked = np.argsort(-joint_scores)
    summary: list[dict[str, float | int | str]] = []
    for rank, joint_index in enumerate(ranked[:3], start=1):
        summary.append(
            {
                "rank": rank,
                "joint_index": int(joint_index),
                "joint": COCO17_JOINT_NAMES[int(joint_index)],
                "deviation": float(joint_scores[int(joint_index)]),
                "importance": float(joint_scores[int(joint_index)]),
            }
        )
    return summary


def load_models(checkpoints_root: Path, device: torch.device) -> list[LoadedExerciseModel]:
    loaded: list[LoadedExerciseModel] = []

    for exercise_dir in sorted(checkpoints_root.glob("exercise_*")):
        ckpt_path = exercise_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model_state_dict = ckpt["model_state_dict"]
        in_channels = int(model_state_dict["encoder.0.spatial_attn.proj.weight"].shape[1])
        use_phase_decoder = bool(ckpt.get("use_phase_decoder", True))

        model = ModelFactory().create_evaluator(
            in_channels=in_channels,
            hidden_channels=tuple(cfg.get("hidden_channels", (64, 128))),
            embedding_dim=int(cfg.get("embedding_dim", 128)),
            use_phase_decoder=use_phase_decoder,
            device=device,
        )
        model.load_state_dict(model_state_dict, strict=False)
        model.eval()

        template_tensor = torch.as_tensor(ckpt.get("template_tensor"), dtype=torch.float32, device=device)
        template_xyz_tensor = ckpt.get("template_xyz_tensor")
        if template_xyz_tensor is not None:
            template_xyz_tensor = torch.as_tensor(template_xyz_tensor, dtype=torch.float32, device=device)

        raw_threshold = float(ckpt.get("raw_threshold", ckpt.get("val_threshold", 0.5)))
        threshold = float(ckpt.get("val_threshold", raw_threshold))
        exercise_id = int(ckpt.get("exercise_id", int(exercise_dir.name.split("_")[-1]) - 1))

        loaded.append(
            LoadedExerciseModel(
                exercise_id=exercise_id,
                model=model,
                template_tensor=template_tensor,
                template_xyz_tensor=template_xyz_tensor,
                threshold=threshold,
                raw_threshold=raw_threshold,
                checkpoint_path=ckpt_path,
                use_phase_decoder=use_phase_decoder,
                in_channels=in_channels,
            )
        )

    return loaded


@lru_cache(maxsize=8)
def _cached_models(checkpoints_root: str, device_key: str) -> tuple[LoadedExerciseModel, ...]:
    device = torch.device(device_key)
    return tuple(load_models(Path(checkpoints_root), device))


def get_cached_models(checkpoints_root: Path, device: torch.device) -> list[LoadedExerciseModel]:
    return list(_cached_models(str(checkpoints_root.resolve()), str(device)))


def run_prediction(
    input_tensor: torch.Tensor,
    raw_sequence: np.ndarray,
    models: list[LoadedExerciseModel],
    device: torch.device,
    ghost_anchor: str = "hips",
) -> tuple[dict, np.ndarray, list[dict[str, float | int | str]], dict[str, np.ndarray | None]]:
    del raw_sequence, ghost_anchor
    if not models:
        raise ValueError("No models available for inference")

    user_tensor = input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    all_results: list[dict] = []
    best_result: dict | None = None
    best_importance = np.zeros(len(COCO17_JOINT_NAMES), dtype=np.float32)
    worst_joints: list[dict[str, float | int | str]] = []
    ghost_xyz: np.ndarray | None = None

    for loaded in models:
        template_tensor = loaded.template_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
        kwargs = {}
        if loaded.template_xyz_tensor is not None:
            kwargs["template_xyz_raw"] = loaded.template_xyz_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

        output = loaded.model(template_tensor, user_tensor, **kwargs)
        score = float(output["similarity_score"].detach().cpu().item())
        predicted_label = "correct" if score >= loaded.threshold else "incorrect"
        joint_importance = output.get("joint_importance")
        if torch.is_tensor(joint_importance):
            joint_importance_np = joint_importance.detach().cpu().numpy().astype(np.float32)
        else:
            joint_importance_np = np.zeros(len(COCO17_JOINT_NAMES), dtype=np.float32)

        result = {
            "exercise_id": loaded.exercise_id,
            "exercise_name": f"Exercise {loaded.exercise_id + 1:02d}",
            "predicted_label": predicted_label,
            "score": score,
            "threshold": loaded.threshold,
            "raw_threshold": loaded.raw_threshold,
            "margin": max(0.0, loaded.threshold - loaded.raw_threshold),
            "checkpoint": str(loaded.checkpoint_path),
            "has_phase_decoder": loaded.use_phase_decoder,
        }
        all_results.append(result)

        if best_result is None or score > float(best_result["score"]):
            best_result = result
            best_importance = joint_importance_np
            if loaded.use_phase_decoder:
                warped = output.get("warped_template_xyz")
                if torch.is_tensor(warped):
                    ghost_xyz = warped.detach().cpu().numpy()[0]
                else:
                    ghost_xyz = None

            if "joint_error_magnitude" in output and torch.is_tensor(output["joint_error_magnitude"]):
                joint_scores = output["joint_error_magnitude"].detach().cpu().numpy()[0]
            else:
                joint_scores = 1.0 - joint_importance_np
            worst_joints = _build_joint_summary(np.asarray(joint_scores, dtype=np.float32))

    report = {
        "best": best_result,
        "all": all_results,
        "worst_joints": worst_joints,
    }
    if worst_joints:
        report["feedback_summary"] = ", ".join(item["joint"] for item in worst_joints[:3])

    phase_outputs = {"ghost_xyz": ghost_xyz}
    return report, best_importance, worst_joints, phase_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a rehabilitation video with KinetoCheck feedback overlays."
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video")
    parser.add_argument("--exercise-id", type=int, default=1, help="Exercise number to load (1-based)")
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("checkpoints") / "uiprmd_phase_aware_rom",
        help="Directory containing exercise checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Video-kineto-annotated") / "single_video_runs",
        help="Directory where outputs will be written",
    )
    parser.add_argument("--pose-model", type=Path, default=None, help="Optional path to the pose model")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cpu, cuda, auto)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    models = load_models(args.checkpoints_root, device)
    selected_models = [model for model in models if model.exercise_id == args.exercise_id - 1]
    if not selected_models:
        selected_models = models

    preprocessor = UIPRMDPreprocessor()
    pose_model_path = ensure_pose_task_model(args.pose_model)

    report = process_video(
        video_path=args.video,
        models=selected_models,
        preprocessor=preprocessor,
        output_dir=args.output_dir,
        device=device,
        pose_model_path=pose_model_path,
    )

    print(report["best"]["predicted_label"])


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
            frame_height,
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