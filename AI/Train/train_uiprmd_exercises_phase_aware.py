from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phase_aware import ContrastiveLoss, DeltaRegressionLoss, ExerciseEvaluator, RangeOfMotionLoss
from Models.factory import ModelFactory, LossFactory
from Train.split_strategies import SplitPlan, build_split_plans
from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor, build_features_from_aligned
from Preprocessing.UIPRMD_loader import UIPRMDLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    output_dir: Path
    exercise_ids: tuple[int, ...]
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_channels: tuple[int, ...] = (64, 128)
    embedding_dim: int = 128
    margin: float = 1.0
    delta_weight: float = 0.0
    rom_weight: float = 2
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    patience: int = 10
    num_workers: int = 0
    device: str = "auto"
    use_phase_decoder: bool = True
    split_mode: str = "subject"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ExercisePairDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        template_tensor: torch.Tensor,
        preprocessor: UIPRMDPreprocessor,
    ):
        self.template = template_tensor.detach().clone().float()
        self.samples: list[dict] = []

        for record in records:
            user_tensor = self._record_to_tensor(record, preprocessor)
            target = 1.0 if int(record["label"]) == 0 else 0.0

            self.samples.append(
                {
                    "template": self.template,
                    "user": user_tensor,
                    "target": torch.tensor(target, dtype=torch.float32),
                    "label": int(record["label"]),
                    "exercise_id": int(record["exercise_id"]),
                    "subject_id": int(record["subject_id"]),
                    "file": str(record["file"]),
                }
            )

    @staticmethod
    def _record_to_tensor(
        record: dict, preprocessor: UIPRMDPreprocessor
    ) -> torch.Tensor:
        aligned   = preprocessor.align_vicon_to_mediapipe(record["sequence"])
        processed = preprocessor.process(aligned)            # (T, 17, 3)
        features  = build_features_from_aligned(processed)  # (12, T, 17)
        return torch.from_numpy(features).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_by_subject(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not records:
        return [], [], []

    subjects = sorted({int(r["subject_id"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(subjects)

    total = len(subjects)
    if total == 1:
        subject_to_split = {subjects[0]: "train"}
    else:
        raw_sum = max(1e-9, train_ratio + val_ratio + test_ratio)
        n_train = max(1, int(round(total * train_ratio / raw_sum)))
        n_val   = max(1, int(round(total * val_ratio  / raw_sum))) if total >= 3 else 0
        n_test  = max(1, total - n_train - n_val)               if total >= 3 else max(0, total - n_train)

        while n_train + n_val + n_test > total:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        while n_train + n_val + n_test < total:
            if n_train <= n_val and n_train <= n_test:
                n_train += 1
            elif n_val <= n_test:
                n_val += 1
            else:
                n_test += 1

        subject_to_split = {}
        for s in subjects[:n_train]:
            subject_to_split[s] = "train"
        for s in subjects[n_train: n_train + n_val]:
            subject_to_split[s] = "val"
        for s in subjects[n_train + n_val: n_train + n_val + n_test]:
            subject_to_split[s] = "test"

    train_records, val_records, test_records = [], [], []
    for r in records:
        split = subject_to_split.get(int(r["subject_id"]), "train")
        if split == "train":
            train_records.append(r)
        elif split == "val":
            val_records.append(r)
        else:
            test_records.append(r)

    return train_records, val_records, test_records


# def build_template_tensor(
#     records: list[dict], preprocessor: UIPRMDPreprocessor
# ) -> torch.Tensor:
#     correct_records = [r for r in records if int(r["label"]) == 0]
#     if not correct_records:
#         raise ValueError("No correct samples to build a template.")
#
#     tensors = []
#     for r in correct_records:
#         aligned   = preprocessor.align_vicon_to_mediapipe(r["sequence"])
#         processed = preprocessor.process(aligned)
#         features  = build_features_from_aligned(processed)  # (12, T, 17)
#         tensors.append(torch.from_numpy(features).float())
#
#     return torch.stack(tensors, dim=0).mean(dim=0)

def build_template_tensor(
    records: list[dict], preprocessor: UIPRMDPreprocessor
) -> torch.Tensor:
    correct_records = [r for r in records if int(r["label"]) == 0]
    if not correct_records:
        raise ValueError("No correct samples to build a template.")

    # 1. Find the record with the median raw frame length
    # This guarantees we pick a squat performed at a "normal" human speed.
    lengths = [len(r["sequence"]) for r in correct_records]
    median_idx = int(np.argsort(lengths)[len(lengths) // 2])
    best_record = correct_records[median_idx]

    # 2. Process ONLY this single, high-quality record
    aligned   = preprocessor.align_vicon_to_mediapipe(best_record["sequence"])
    processed = preprocessor.process(aligned)
    features  = build_features_from_aligned(processed)  # (12, T, 17)

    return torch.from_numpy(features).float()


def _resample_to(arr: np.ndarray, target_T: int) -> np.ndarray:
    """
    Linearly resample a (T, ...) array to (target_T, ...) along axis 0.
    All inner dimensions are preserved; only the time axis is resampled.
    """
    T_i = arr.shape[0]
    if T_i == target_T:
        return arr
    if T_i == 1:
        return np.repeat(arr, target_T, axis=0)
    original_trailing = arr.shape[1:]
    flat = arr.reshape(T_i, -1)                                # (T_i, F)
    src_t = np.linspace(0.0, 1.0, T_i,      dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_T, dtype=np.float32)
    out = np.zeros((target_T, flat.shape[1]), dtype=np.float32)
    for feat in range(flat.shape[1]):
        out[:, feat] = np.interp(dst_t, src_t, flat[:, feat])
    return out.reshape(target_T, *original_trailing)


# def build_raw_xyz_template(records: list[dict], preprocessor: UIPRMDPreprocessor) -> torch.Tensor:
#     """
#     Build a template in RAW un-preprocessed XY coordinates for the overlay.
#
#     WHY NOT align_vicon_to_mediapipe() / preprocessor.process():
#     -------------------------------------------------------------
#     align_vicon_to_mediapipe() subtracts the hip midpoint, putting output
#     in body-centred metric space — NOT [0,1] image fractions.
#     preprocessor.process() z-scores on top of that — even further off.
#     The ghost overlay needs image-fraction XY (matching MediaPipe output at
#     runtime) so joints map to the correct pixel positions.
#
#     Coordinate source:
#     ------------------
#     For MediaPipe datasets, record["sequence"] already contains raw
#     image-fraction XY in shape (T, 17, 3) or flat (T, 51).  We use that.
#     For Vicon datasets (39-joint), image fractions are unavailable; we fall
#     back to body-centred coords — the overlay will be approximate.
#
#     Variable-length fix:
#     --------------------
#     Sequences have different frame counts.  We resample every sequence to
#     the median length via linear interpolation before averaging, which fixes
#     the "stack expects each tensor to be equal size" RuntimeError.
#
#     Returns
#     -------
#     torch.Tensor  shape (3, T, J)
#         Ch 0,1 = image-fraction X, Y  (~[0,1]).
#         Ch 2   = Z / depth (not used for 2-D overlay).
#     """
#     correct_records = [r for r in records if int(r["label"]) == 0]
#     if not correct_records:
#         raise ValueError("No correct samples to build raw XYZ template.")
#
#     raw_list: list[np.ndarray] = []
#     using_raw_image_fractions = True
#
#     for r in correct_records:
#         seq = np.asarray(r["sequence"], dtype=np.float32)
#
#         if seq.ndim == 3 and seq.shape[1] == 17:
#             # (T, 17, 3) — raw MediaPipe image-fraction XYZ
#             raw_list.append(seq)
#         elif seq.ndim == 2 and seq.shape[1] == 51:
#             # (T, 51) flat MediaPipe → reshape to (T, 17, 3)
#             raw_list.append(seq.reshape(seq.shape[0], 17, 3))
#         else:
#             # Vicon data (39-joint or flat): image fractions unavailable.
#             # Use body-centred aligned coords as a best approximation.
#             using_raw_image_fractions = False
#             aligned = preprocessor.align_vicon_to_mediapipe(seq)  # (T, 17, 3)
#             raw_list.append(aligned)
#
#     if not using_raw_image_fractions:
#         print(
#             "  [warn] build_raw_xyz_template: dataset appears to be Vicon data "
#             "(not MediaPipe image fractions).  Ghost overlay will use "
#             "body-centred coordinates — joint positions will be approximate."
#         )
#
#     # Resample all sequences to the median frame count, then average.
#     lengths = [a.shape[0] for a in raw_list]
#     target_T = int(np.median(lengths))
#
#     tensors: list[torch.Tensor] = []
#     for seq_arr in raw_list:
#         resampled = _resample_to(seq_arr, target_T)          # (target_T, 17, 3)
#         xyz = np.transpose(resampled, (2, 0, 1)).copy()      # (3, target_T, 17)
#         tensors.append(torch.from_numpy(xyz).float())
#
#     return torch.stack(tensors, dim=0).mean(dim=0)           # (3, target_T, 17)

def build_raw_xyz_template(
    records: list[dict], preprocessor: UIPRMDPreprocessor
) -> torch.Tensor:
    correct_records = [r for r in records if int(r["label"]) == 0]
    if not correct_records:
        raise ValueError("No correct samples to build raw XYZ template.")

    # 1. Select the exact same median record
    lengths = [len(r["sequence"]) for r in correct_records]
    median_idx = int(np.argsort(lengths)[len(lengths) // 2])
    best_record = correct_records[median_idx]

    seq = np.asarray(best_record["sequence"], dtype=np.float32)

    # 2. Format it directly without temporal resampling/averaging
    if seq.ndim == 3 and seq.shape[1] == 17:
        raw_seq = seq
    elif seq.ndim == 2 and seq.shape[1] == 51:
        raw_seq = seq.reshape(seq.shape[0], 17, 3)
    else:
        raw_seq = preprocessor.align_vicon_to_mediapipe(seq)

    # Transpose to (3, T, J)
    xyz = np.transpose(raw_seq, (2, 0, 1)).copy()
    return torch.from_numpy(xyz).float()

def collate_batch(batch: list[dict]) -> dict:
    templates = torch.stack([item["template"] for item in batch], dim=0)
    users     = torch.stack([item["user"]     for item in batch], dim=0)
    targets   = torch.stack([item["target"]   for item in batch], dim=0)
    return {
        "template":    templates,
        "user":        users,
        "target":      targets,
        "label":       torch.tensor([item["label"]       for item in batch], dtype=torch.long),
        "exercise_id": torch.tensor([item["exercise_id"] for item in batch], dtype=torch.long),
        "subject_id":  torch.tensor([item["subject_id"]  for item in batch], dtype=torch.long),
        "file":        [item["file"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Per-exercise training
# ---------------------------------------------------------------------------

def train_one_exercise(
    exercise_id: int,
    cfg: TrainingConfig,
    records: list[dict] | None = None,
    split_plan: SplitPlan | None = None,
) -> dict[str, float]:
    set_seed(cfg.seed + exercise_id)

    loader_ds    = UIPRMDLoader(cfg.data_root)
    preprocessor = UIPRMDPreprocessor()
    if records is None:
        records = loader_ds.load_vicon_data(exercise_id=exercise_id)
    if not records:
        raise ValueError(f"No records found for exercise {exercise_id}.")

    if split_plan is None:
        train_records, val_records, test_records = split_by_subject(
            records, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed
        )
        fold_name = "subject_split"
        held_out_subject = None
    else:
        train_records = split_plan.train_records
        val_records = split_plan.val_records
        test_records = split_plan.test_records
        fold_name = split_plan.fold_name
        held_out_subject = split_plan.held_out_subject
    if not train_records:
        raise ValueError(f"Training split is empty for exercise {exercise_id}.")

    template_tensor     = build_template_tensor(train_records, preprocessor)

    # RAW XYZ template for the inference ghost overlay.
    # Must NOT be preprocessed — see build_raw_xyz_template docstring.
    template_xyz_tensor = build_raw_xyz_template(train_records, preprocessor)

    train_dataset = ExercisePairDataset(train_records, template_tensor, preprocessor)
    val_dataset   = ExercisePairDataset(
        val_records  if val_records  else train_records, template_tensor, preprocessor
    )
    test_dataset  = ExercisePairDataset(
        test_records if test_records else val_records if val_records else train_records,
        template_tensor, preprocessor,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_batch,
    )
    val_loader  = DataLoader(
        val_dataset,  batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_batch,
    )

    device = resolve_device(cfg.device)
    model = ModelFactory().create_evaluator(
        in_channels=12,
        hidden_channels=cfg.hidden_channels,
        embedding_dim=cfg.embedding_dim,
        use_phase_decoder=cfg.use_phase_decoder,
        device=device,
    )

    loss_factory = LossFactory()
    criterion = loss_factory.create_contrastive(margin=cfg.margin)
    delta_criterion = loss_factory.create_delta_regression() if cfg.use_phase_decoder else None
    rom_criterion = loss_factory.create_rom_loss(weight=cfg.rom_weight)
    optimizer       = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )


    # Delegate training loop to the shared Trainer implementation
    from Train.trainer import Trainer

    trainer = Trainer(cfg)
    summary = trainer.train(
        exercise_id=exercise_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir=cfg.output_dir,
        template_tensor=template_tensor,
        template_xyz_tensor=template_xyz_tensor,
        delta_criterion=delta_criterion,
        delta_weight=cfg.delta_weight,
        rom_criterion=rom_criterion,
        fold_name=fold_name,
        held_out_subject=held_out_subject,
        feature_method=getattr(cfg, "feature_method", None),
        in_channels=12,
        hidden_channels=cfg.hidden_channels,
        embedding_dim=cfg.embedding_dim,
        use_phase_decoder=cfg.use_phase_decoder,
        feature_channels=["x","y","z","vx","vy","vz","ax","ay","az","angle","angular_vel","bone_ratio"],
    )

    return summary

    summary = {
        "exercise_id":         exercise_id,
        "split_mode":          cfg.split_mode,
        "fold_name":           fold_name,
        "held_out_subject":    held_out_subject,
        "best_epoch":          best_epoch,
        "best_val_f1":         best_val_f1,
        "best_test_loss":      best_test_met.get("loss", 0.0),
        "best_test_accuracy":  best_test_met.get("accuracy", 0.0),
        "best_test_precision": best_test_met.get("precision", 0.0),
        "best_test_recall":    best_test_met.get("recall", 0.0),
        "best_test_f1":        best_test_met.get("f1", 0.0),
        "best_threshold":      best_test_met.get("threshold", 0.5),
    }
    (exercise_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_exercise_ids(raw_values: list[int] | None) -> tuple[int, ...]:
    if not raw_values:
        return tuple(range(10))
    return tuple(int(v) for v in raw_values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one UI-PRMD ST-GAT model per exercise."
    )
    parser.add_argument("--data-root",        type=Path,  default=Path("Datasets") / "UIPRMD")
    parser.add_argument("--output-dir",       type=Path,  default=Path("checkpoints") / "uiprmd")
    parser.add_argument("--exercise-ids",     type=int,   nargs="*", default=None)
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--batch-size",       type=int,   default=8)
    parser.add_argument("--learning-rate",    type=float, default=1e-3)
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--embedding-dim",    type=int,   default=128)
    parser.add_argument("--hidden-channels",  type=int,   nargs="*", default=[64, 128])
    parser.add_argument("--margin",           type=float, default=1.0)
    parser.add_argument("--delta-weight",     type=float, default=0)
    parser.add_argument("--rom-weight",       type=float, default=2.0)
    parser.add_argument("--train-ratio",      type=float, default=0.8)
    parser.add_argument("--val-ratio",        type=float, default=0.1)
    parser.add_argument("--test-ratio",       type=float, default=0.1)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--patience",         type=int,   default=10)
    parser.add_argument("--num-workers",      type=int,   default=0)
    parser.add_argument("--device",           type=str,   default="auto")
    parser.add_argument("--split-mode",       type=str,   default="subject", choices=["subject", "loso"])
    parser.add_argument("--no-phase-decoder", action="store_true",
                        help="Disable FrameDecoder/PhaseAligner (pure similarity mode).")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        exercise_ids=parse_exercise_ids(args.exercise_ids),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_channels=tuple(args.hidden_channels),
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        delta_weight=args.delta_weight,
        rom_weight=args.rom_weight,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        use_phase_decoder=not args.no_phase_decoder,
        split_mode=args.split_mode,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for exercise_id in cfg.exercise_ids:
        if cfg.split_mode == "loso":
            exercise_loader = UIPRMDLoader(cfg.data_root)
            exercise_records = exercise_loader.load_vicon_data(exercise_id=exercise_id)
            split_plans = build_split_plans(
                exercise_records,
                cfg.split_mode,
                cfg.train_ratio,
                cfg.val_ratio,
                cfg.test_ratio,
                cfg.seed + exercise_id,
            )
            for split_plan in split_plans:
                summary = train_one_exercise(exercise_id, cfg, records=exercise_records, split_plan=split_plan)
                summaries.append(summary)
                print(json.dumps(summary, indent=2))
        else:
            summary = train_one_exercise(exercise_id, cfg)
            summaries.append(summary)
            print(json.dumps(summary, indent=2))

    (cfg.output_dir / "all_exercises_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
