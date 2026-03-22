"""Three-stage UI-PRMD angles training.

Stage 1: pretrain on Vicon angles (segmented, clean)
Stage 2: fine-tune on Kinect angles (segmented, realistic noise)
Stage 3: train on mixed Vicon+Kinect angles with augmentation
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from training.augmentation import AugmentationPipeline, AugmentedDataset
from training.callbacks import (
    CallbackRunner,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateauCallback,
    TrainingMetrics,
)
from training.uiprmd_angles_dataset import UIPRMDAnglesDataset
from app.models.model_factory import ModelFactory


def _make_bar(iterable, total, desc, unit="batch"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False, ncols=100)


def _build_loaders(dataset, batch_size: int):
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    use_cuda = torch.cuda.is_available() and settings.DEVICE != "cpu"
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=settings.DATALOADER_WORKERS,
        pin_memory=settings.PIN_MEMORY and use_cuda,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _try_transfer_weights(model, src_path: str) -> None:
    if not src_path or not os.path.exists(src_path):
        return
    if getattr(model, "model", None) is None:
        return

    try:
        state = torch.load(src_path, map_location=model.device)
        if isinstance(state, dict):
            model.model.load_state_dict(state, strict=False)
            print(f"  Transfer loaded (strict=False): {src_path}")
    except Exception as exc:
        print(f"  Warning: could not transfer from {src_path}: {exc}")


def _train_one_stage(
    *,
    stage_name: str,
    exercise_id: int,
    exercise_name: str,
    dataset,
    model,
    epochs: int,
    save_path: str,
) -> float:
    if len(dataset) == 0:
        print(f"  No samples for {stage_name}. Skipping.")
        return 0.0

    train_loader, val_loader = _build_loaders(dataset, settings.BATCH_SIZE)
    print(f"  {stage_name}: train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

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
    if getattr(model, "optimizer", None) is not None:
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
    start = time.time()
    epoch_iter = range(epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc=f"{stage_name} epochs", unit="ep", ncols=100)

    for epoch in epoch_iter:
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        for batch in _make_bar(train_loader, len(train_loader), f"  {stage_name} train"):
            m = model.train_step(batch)
            train_loss += m["loss"]
            train_acc += m["accuracy"]

        for batch in _make_bar(val_loader, len(val_loader), f"  {stage_name} val"):
            m = model.eval_step(batch)
            val_loss += m["loss"]
            val_acc += m["accuracy"]

        train_loss /= max(len(train_loader), 1)
        train_acc /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        val_acc /= max(len(val_loader), 1)

        lr = model.optimizer.param_groups[0]["lr"] if model.optimizer else 0.0
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            total_epochs=epochs,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=lr,
            epoch_time=0.0,
            elapsed=time.time() - start,
        )
        callbacks.on_epoch_end(metrics)
        if callbacks.should_stop:
            break

    callbacks.on_train_end()
    ckpt = next((c for c in callbacks.callbacks if isinstance(c, ModelCheckpoint)), None)
    best = ckpt.best if ckpt and ckpt.best is not None else 0.0
    print(
        f"  {stage_name} complete | exercise={exercise_id} ({exercise_name}) | "
        f"best_val_acc={best:.4f} | weights={save_path}"
    )
    return best


def main():
    parser = argparse.ArgumentParser(description="Three-stage UI-PRMD angles training")
    parser.add_argument("--exercise", type=int, action="append", default=None, help="Exercise IDs (0-9)")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS, help="Epochs per stage")
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Stage to start from (1, 2, or 3). Useful for resume.",
    )
    parser.add_argument("--lr-stage1", type=float, default=settings.LR_STAGE1_ANGLES)
    parser.add_argument("--lr-stage2", type=float, default=settings.LR_STAGE2_ANGLES)
    parser.add_argument("--lr-stage3", type=float, default=settings.LR_STAGE3_ANGLES)
    parser.add_argument("--feature-dim", type=int, default=settings.ANGLES_KINECT_DIM)
    args = parser.parse_args()

    settings.BATCH_SIZE = args.batch_size
    data_dir = settings.UIPRMD_DATA_DIR
    if not Path(data_dir).is_dir():
        raise FileNotFoundError(f"UIPRMD data dir not found: {data_dir}")

    exercises = args.exercise if args.exercise else list(range(10))
    common_dim = args.feature_dim

    aug_stage3 = AugmentationPipeline(
        angle_noise_std=settings.AUGMENT_ANGLE_NOISE_STD,
        frame_dropout_p=settings.AUGMENT_FRAME_DROPOUT,
        scale_jitter_range=settings.AUGMENT_SCALE_JITTER,
        enable_angles=True,
    )

    for exercise_id in sorted(exercises):
        exercise_name = settings.UIPRMD_EXERCISES.get(exercise_id, f"Exercise {exercise_id}")
        print(f"\n{'=' * 80}")
        print(f"Exercise {exercise_id}: {exercise_name}")
        print(f"{'=' * 80}")

        weights_root = Path(settings.WEIGHTS_DIR) / "uiprmd_angles"
        weights_root.mkdir(parents=True, exist_ok=True)
        s1_path = str(weights_root / f"stage1_ex{exercise_id}_best.pt")
        s2_path = str(weights_root / f"stage2_ex{exercise_id}_best.pt")
        s3_path = str(weights_root / f"stage3_ex{exercise_id}_best.pt")

        acc1 = float("nan")
        acc2 = float("nan")

        # Stage 1: Vicon segmented
        if args.start_stage <= 1:
            s1_dataset = UIPRMDAnglesDataset(
                data_dir,
                modality="vicon",
                exercise_id=exercise_id,
                use_segmented=True,
                feature_dim=common_dim,
            )
            s1_model = ModelFactory.create(settings.ACTIVE_MODEL)
            s1_model.build(num_keypoints=common_dim, keypoint_dim=1, lr=args.lr_stage1)
            acc1 = _train_one_stage(
                stage_name="Stage1-Vicon",
                exercise_id=exercise_id,
                exercise_name=exercise_name,
                dataset=s1_dataset,
                model=s1_model,
                epochs=args.epochs,
                save_path=s1_path,
            )
        else:
            print(f"Skipping Stage 1 (start-stage={args.start_stage})")

        # Stage 2: Kinect segmented, transfer from Stage 1
        if args.start_stage <= 2:
            s2_dataset = UIPRMDAnglesDataset(
                data_dir,
                modality="kinect",
                exercise_id=exercise_id,
                use_segmented=True,
                feature_dim=common_dim,
            )
            s2_model = ModelFactory.create(settings.ACTIVE_MODEL)
            s2_model.build(num_keypoints=common_dim, keypoint_dim=1, lr=args.lr_stage2)
            _try_transfer_weights(s2_model, s1_path)
            acc2 = _train_one_stage(
                stage_name="Stage2-Kinect",
                exercise_id=exercise_id,
                exercise_name=exercise_name,
                dataset=s2_dataset,
                model=s2_model,
                epochs=args.epochs,
                save_path=s2_path,
            )
        else:
            print(f"Skipping Stage 2 (start-stage={args.start_stage})")

        # Stage 3: mixed Vicon+Kinect (segmented + non-segmented) + augmentation
        v_mix = UIPRMDAnglesDataset(
            data_dir,
            modality="vicon",
            exercise_id=exercise_id,
            use_segmented=False,
            feature_dim=common_dim,
        )
        k_mix = UIPRMDAnglesDataset(
            data_dir,
            modality="kinect",
            exercise_id=exercise_id,
            use_segmented=False,
            feature_dim=common_dim,
        )
        mix_dataset = ConcatDataset([
            AugmentedDataset(v_mix, aug_stage3),
            AugmentedDataset(k_mix, aug_stage3),
        ])

        s3_model = ModelFactory.create(settings.ACTIVE_MODEL)
        s3_model.build(num_keypoints=common_dim, keypoint_dim=1, lr=args.lr_stage3)
        _try_transfer_weights(s3_model, s2_path)
        acc3 = _train_one_stage(
            stage_name="Stage3-MixedAug",
            exercise_id=exercise_id,
            exercise_name=exercise_name,
            dataset=mix_dataset,
            model=s3_model,
            epochs=args.epochs,
            save_path=s3_path,
        )

        if args.start_stage >= 3 and not os.path.exists(s2_path):
            print(
                "Warning: --start-stage 3 used without existing Stage 2 checkpoint; "
                "Stage 3 started from random initialization."
            )

        s1_text = f"{acc1:.4f}" if acc1 == acc1 else "SKIPPED"
        s2_text = f"{acc2:.4f}" if acc2 == acc2 else "SKIPPED"
        print(
            f"Summary ex={exercise_id} | "
            f"stage1={s1_text} stage2={s2_text} stage3={acc3:.4f}"
        )


if __name__ == "__main__":
    main()
