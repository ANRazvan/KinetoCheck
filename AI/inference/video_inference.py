from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor
from inference.inference_engine import (
    LoadedExerciseModel,
    ensure_pose_task_model,
    load_models,
    process_video,
    resolve_device,
)


def _select_models_for_exercise(
    models: list[LoadedExerciseModel],
    exercise_number: int,
) -> list[LoadedExerciseModel]:
    target_exercise_id = exercise_number - 1
    specific_models = [model for model in models if model.exercise_id == target_exercise_id]
    if specific_models:
        return specific_models

    available = ", ".join(
        f"exercise_{model.exercise_id + 1:02d}" for model in models
    ) or "none"
    raise ValueError(
        f"No checkpoint found for exercise {exercise_number}. Available: {available}"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a rehabilitation video with KinetoCheck feedback overlays."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video (.mp4, .mov, .mkv, .avi, .webm)",
    )
    parser.add_argument(
        "--exercise-id",
        type=int,
        default=1,
        help="Exercise number to load (1-based, e.g. 1 for exercise_01)",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("checkpoints") / "uiprmd",
        help="Directory containing exercise_XX checkpoint folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Video-kineto-annotated") / "single_video_runs",
        help="Directory where the annotated video and report will be written",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=None,
        help="Optional path to the MediaPipe pose task model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (cpu, cuda, auto)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    models = load_models(args.checkpoints_root, device)
    selected_models = _select_models_for_exercise(models, args.exercise_id)
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

    worst_joints = report.get("worst_joints", [])
    feedback_summary = report.get("feedback_summary", "")
    if not feedback_summary and worst_joints:
        feedback_summary = ", ".join(item["joint"] for item in worst_joints[:3])

    print("\n" + "=" * 48)
    print("KinetoCheck video annotation complete")
    print("=" * 48)
    print(f"Video file        : {args.video.name}")
    print(f"Exercise          : exercise_{args.exercise_id:02d}")
    print(f"Prediction        : {report['best']['predicted_label']}")
    print(f"Similarity score  : {report['best']['score']:.4f}")
    print(f"Threshold         : {report['best']['threshold']:.4f}")
    print(f"Annotated video   : {report['annotated_video']}")
    if feedback_summary:
        print(f"Feedback          : {feedback_summary}")
    if worst_joints:
        print(
            "Top joints        : "
            + ", ".join(
                f"{item['joint']} ({item['deviation']:.3f})" for item in worst_joints[:3]
            )
        )

    report_path = args.output_dir / f"{args.video.stem}_report.json"
    if report_path.exists():
        print(f"Report            : {report_path}")
    print("=" * 48 + "\n")


if __name__ == "__main__":
    main()