"""
Train one correctness model **per exercise**.

Features:
    - tqdm progress bars for epochs & batches
    - Callback system (early stopping, model checkpoint, LR scheduler)
    - Automatic Mixed Precision (AMP) on CUDA
    - Gradient clipping, weight decay
    - pin_memory for CUDA DataLoaders

Usage:
    # Train ALL exercises (loops 0-8):
    python -m training.train

    # Train a single exercise:
    python -m training.train --exercise 3

    # Train a subset:
    python -m training.train --exercise 0 --exercise 1 --exercise 3
"""

import argparse
import os
import sys
import time


import torch
from torch.utils.data import DataLoader, random_split

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback
    tqdm = None  # type: ignore

# Ensure the Backend package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.callbacks import (
    CallbackRunner,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateauCallback,
    TrainingMetrics,
)
from training.training_factory import AbstractTrainingFactory, get_training_factory
from config import settings


def _make_bar(iterable, total, desc, unit="batch", disable=False):
    """Wrap *iterable* with tqdm if available, plain iterator otherwise."""
    if tqdm is not None and not disable:
        return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False, ncols=100)
    return iterable


# ── per-exercise training loop ───────────────────────────────────────

def train_exercise(
    exercise_id: int,
    model_name: str,
    factory: AbstractTrainingFactory | None = None,
    dataset_key: str = "intellirehab",
    data_dir: str | None = None,
) -> float:
    """
    Train a single model for *one* exercise.

    Args:
        exercise_id: Which exercise to train (0-8 for IntelliRehab).
        model_name:  Architecture key recognised by ModelFactory.
        factory:     Training-component factory.  Defaults to
                     ``IntelliRehabTrainingFactory`` when omitted so that
                     existing call-sites keep working unchanged.

    Returns:
        Best validation accuracy achieved.
    """
    if factory is None:
        factory = get_training_factory("intellirehab")
    if data_dir is None:
        data_dir = settings.data_dir_for(dataset_key)

    exercise_name = settings.exercise_name_for(dataset_key, exercise_id)
    print(f"\n{'═' * 60}")
    print(f"  Exercise {exercise_id}: {exercise_name}  [{factory.dataset_name}]")
    print(f"{'═' * 60}")

    # ── dataset (created via factory) ─────────────────────────────
    dataset = factory.create_dataset(data_dir, exercise_id=exercise_id)
    if len(dataset) == 0:
        print(f"  ⚠  No samples for exercise {exercise_id} — skipping.")
        return 0.0

    dist = dataset.label_distribution()
    total = len(dataset)
    for label, count in sorted(dist.items()):
        tag = "correct" if label == 0 else "incorrect"
        print(f"  {tag}: {count} ({100.0 * count / total:.1f}%)")

    train_size = int(0.8 * total)
    val_size = total - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    use_cuda = torch.cuda.is_available() and settings.DEVICE != "cpu"
    loader_kwargs = dict(
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.DATALOADER_WORKERS,
        pin_memory=settings.PIN_MEMORY and use_cuda,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    print(f"  Train: {train_size}  Val: {val_size}")

    # ── model (created via factory) ───────────────────────────────
    model = factory.create_model(model_name)
    num_joints = getattr(dataset, "num_joints", factory.num_joints)
    model.build(num_keypoints=num_joints)
    info = model.get_model_info()
    print(f"  Model : {info}")
    print(f"  Joints: {num_joints}")
    print(f"  Device: {info.get('device', '?')}   AMP: {getattr(model, 'use_amp', False)}")

    weights_dir = settings.weights_dir_for(dataset_key)
    os.makedirs(weights_dir, exist_ok=True)
    save_path = settings.weights_path_for(model_name, exercise_id, dataset=dataset_key)
    print(f"  Weights → {save_path}")

    # ── callbacks ────────────────────────────────────────────────
    callbacks = CallbackRunner()
    callbacks.add(
        EarlyStopping(
            patience=settings.EARLY_STOPPING_PATIENCE,
            min_delta=settings.EARLY_STOPPING_MIN_DELTA,
            monitor="val_loss",
            mode="min",
        )
    )
    callbacks.add(ModelCheckpoint(save_path, model, monitor="val_acc", mode="max"))
    if hasattr(model, "optimizer") and model.optimizer is not None:
        callbacks.add(
            ReduceLROnPlateauCallback(
                optimizer=model.optimizer,
                monitor="val_loss",
                factor=settings.LR_SCHEDULER_FACTOR,
                patience=settings.LR_SCHEDULER_PATIENCE,
                min_lr=settings.LR_SCHEDULER_MIN_LR,
            )
        )

    callbacks.on_train_begin()

    # ── training loop ────────────────────────────────────────────
    total_start = time.time()
    n_train = len(train_loader)
    n_val = len(val_loader)

    epoch_iter = range(settings.EPOCHS)
    if tqdm is not None:
        epoch_bar = tqdm(epoch_iter, desc="  Epochs", unit="ep", ncols=100)
    else:
        epoch_bar = epoch_iter

    for epoch in epoch_bar:
        epoch_start = time.time()

        # ── train ────────────────────────────────────────────
        train_loss, train_acc = 0.0, 0.0
        bar = _make_bar(train_loader, n_train, "  train")
        for batch in bar:
            try:
                m = model.train_step(batch)
                train_loss += m["loss"]
                train_acc += m["accuracy"]
                if tqdm is not None and hasattr(bar, "set_postfix"):
                    bar.set_postfix(loss=f"{m['loss']:.4f}", acc=f"{m['accuracy']:.4f}")
            except Exception as e:
                print(f"\n  ERROR train batch: {e}")
                continue
        train_loss /= max(n_train, 1)
        train_acc /= max(n_train, 1)

        # ── validate ─────────────────────────────────────────
        val_loss, val_acc = 0.0, 0.0
        bar = _make_bar(val_loader, n_val, "  val  ")
        for batch in bar:
            try:
                m = model.eval_step(batch)
                val_loss += m["loss"]
                val_acc += m["accuracy"]
            except Exception as e:
                print(f"\n  ERROR val batch: {e}")
                continue
        val_loss /= max(n_val, 1)
        val_acc /= max(n_val, 1)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        current_lr = model.optimizer.param_groups[0]["lr"] if model.optimizer else 0

        # ── metrics → callbacks ──────────────────────────────
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            total_epochs=settings.EPOCHS,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr,
            epoch_time=epoch_time,
            elapsed=elapsed,
        )
        callbacks.on_epoch_end(metrics)

        # Update tqdm postfix / fallback print
        summary = (
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"v_loss={val_loss:.4f} v_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )
        if tqdm is not None and hasattr(epoch_bar, "set_postfix_str"):
            epoch_bar.set_postfix_str(summary)
        else:
            remaining = (elapsed / (epoch + 1)) * (settings.EPOCHS - epoch - 1)
            print(
                f"  Epoch {epoch + 1}/{settings.EPOCHS} ({epoch_time:.1f}s) | "
                f"{summary} | ETA {remaining / 60:.1f}m"
            )

        if callbacks.should_stop:
            break

    callbacks.on_train_end()

    total_time = time.time() - total_start
    # Retrieve best accuracy that was checkpointed
    ckpt = next((c for c in callbacks.callbacks if isinstance(c, ModelCheckpoint)), None)
    best_val_acc = ckpt.best if ckpt and ckpt.best is not None else 0.0

    print(
        f"\n  Exercise {exercise_id} done in {total_time / 60:.1f} min — "
        f"best val_acc={best_val_acc:.4f}"
    )
    return best_val_acc


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train per-exercise models")
    parser.add_argument(
        "--exercise",
        type=int,
        action="append",
        default=None,
        help="Exercise ID(s) to train. Omit to train all (0-8).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model architecture (default: {settings.ACTIVE_MODEL})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="uiprmd",
        help="Dataset family: 'uiprmd' (default) or 'intellirehab'.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional override for dataset root path.",
    )
    args = parser.parse_args()

    dataset_key = args.dataset.lower()
    model_name = args.model or settings.ACTIVE_MODEL
    factory = get_training_factory(dataset_key)
    data_dir = args.data_dir or settings.data_dir_for(dataset_key)

    # If exercises are not explicitly specified, use configured IntelliRehab list,
    # or auto-discover available exercise IDs for other datasets.
    if args.exercise:
        exercise_ids = args.exercise
    elif dataset_key == "intellirehab":
        exercise_ids = sorted(settings.exercises_for(dataset_key).keys())
    else:
        probe = factory.create_dataset(data_dir, exercise_id=None)
        if hasattr(probe, "exercise_distribution"):
            exercise_ids = sorted(probe.exercise_distribution().keys())
        else:
            exercise_ids = []

    # Device banner
    if settings.DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.DEVICE
    cuda_info = ""
    if device == "cuda" and torch.cuda.is_available():
        cuda_info = f" ({torch.cuda.get_device_name(0)})"

    print(f"Model architecture : {model_name}")
    print(f"Dataset family     : {factory.dataset_name}")
    print(f"Data directory     : {data_dir}")
    print(f"Weights directory  : {settings.weights_dir_for(dataset_key)}")
    print(f"Exercises to train : {exercise_ids}")
    print(f"Device             : {device}{cuda_info}")
    print(f"AMP                : {settings.USE_AMP and device == 'cuda'}")
    print(f"Epochs             : {settings.EPOCHS}")
    print(f"Batch size         : {settings.BATCH_SIZE}")
    print(f"Learning rate      : {settings.LEARNING_RATE}")
    print(f"Weight decay       : {settings.WEIGHT_DECAY}")
    print(f"Grad clip norm     : {settings.GRAD_CLIP_NORM}")
    print(f"Early stopping     : patience={settings.EARLY_STOPPING_PATIENCE}")
    print(f"LR scheduler       : ReduceOnPlateau(factor={settings.LR_SCHEDULER_FACTOR}, patience={settings.LR_SCHEDULER_PATIENCE})")

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory does not exist: {data_dir}")
        sys.exit(1)

    results: dict[int, float] = {}

    if not exercise_ids:
        print(f"ERROR: No exercises found for dataset '{dataset_key}' in: {data_dir}")
        sys.exit(1)

    try:
        for eid in exercise_ids:
            if eid not in settings.exercises_for(dataset_key):
                print(f"WARNING: Unknown exercise_id {eid}, skipping.")
                continue
            results[eid] = train_exercise(
                eid,
                model_name,
                factory=factory,
                dataset_key=dataset_key,
                data_dir=data_dir,
            )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)

    # ── summary ──────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Training summary")
    print(f"{'═' * 60}")
    for eid, acc in results.items():
        name = settings.exercise_name_for(dataset_key, eid)
        status = f"val_acc={acc:.4f}" if acc > 0 else "SKIPPED (no data)"
        print(f"  [{eid}] {name:35s} {status}")
    print()


if __name__ == "__main__":
    main()