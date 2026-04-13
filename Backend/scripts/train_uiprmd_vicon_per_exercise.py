"""
Train one UI-PRMD Vicon model per exercise using MediaPipe-aligned landmarks.

This script uses:
- training.uiprmd_dataset.UIPRMDDataset
- app.preprocessing.uiprmd_preprocessor.UIPRMDPreprocessor.align_vicon_to_mediapipe

Usage examples:
    python scripts/train_uiprmd_vicon_per_exercise.py
    python scripts/train_uiprmd_vicon_per_exercise.py --model stgat
    python scripts/train_uiprmd_vicon_per_exercise.py --exercise 0 --exercise 3
    python scripts/train_uiprmd_vicon_per_exercise.py --data-dir Datasets/UIPRMD
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure Backend root is importable when running this file directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
backend_root_str = str(BACKEND_ROOT)
if backend_root_str not in sys.path:
    sys.path.insert(0, backend_root_str)

from config import settings
from training.train import train_exercise
from training.training_factory import get_training_factory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one binary correctness model per UI-PRMD exercise (Vicon positions)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model architecture key (default: {settings.ACTIVE_MODEL})",
    )
    parser.add_argument(
        "--exercise",
        type=int,
        action="append",
        default=None,
        help="Exercise ID(s) to train. Omit to train all available UI-PRMD exercises.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional override for UI-PRMD dataset root directory.",
    )
    args = parser.parse_args()

    dataset_key = "uiprmd"
    model_name = args.model or settings.ACTIVE_MODEL
    data_dir = args.data_dir or settings.data_dir_for(dataset_key)
    factory = get_training_factory(dataset_key)

    if not os.path.isdir(data_dir):
        raise SystemExit(f"ERROR: Data directory does not exist: {data_dir}")

    if args.exercise:
        exercise_ids = sorted(set(args.exercise))
    else:
        probe = factory.create_dataset(data_dir, exercise_id=None)
        if hasattr(probe, "exercise_distribution"):
            exercise_ids = sorted(probe.exercise_distribution().keys())
        else:
            exercise_ids = sorted(settings.exercises_for(dataset_key).keys())

    if not exercise_ids:
        raise SystemExit(f"ERROR: No exercises found in {data_dir}")

    print(f"Model architecture : {model_name}")
    print(f"Dataset family     : {factory.dataset_name} (Vicon positions)")
    print(f"Data directory     : {data_dir}")
    print(f"Exercises to train : {exercise_ids}")
    print(f"Weights directory  : {settings.weights_dir_for(dataset_key)}")

    results: dict[int, float] = {}
    for exercise_id in exercise_ids:
        if exercise_id not in settings.exercises_for(dataset_key):
            print(f"WARNING: Unknown exercise_id {exercise_id}, skipping.")
            continue

        best_acc = train_exercise(
            exercise_id=exercise_id,
            model_name=model_name,
            factory=factory,
            dataset_key=dataset_key,
            data_dir=data_dir,
        )
        results[exercise_id] = best_acc

    print("\nTraining summary:")
    for exercise_id in sorted(results):
        name = settings.exercise_name_for(dataset_key, exercise_id)
        print(f"  [{exercise_id}] {name}: best_val_acc={results[exercise_id]:.4f}")


if __name__ == "__main__":
    main()
