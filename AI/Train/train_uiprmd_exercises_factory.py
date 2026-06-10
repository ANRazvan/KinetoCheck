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

from FeaturePipelines import FeatureExtractorFactory, MotionFeatureExtractor
from Models.factory import ModelFactory, LossFactory
from Train.split_strategies import SplitPlan, build_split_plans
from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor
from Preprocessing.UIPRMD_loader import UIPRMDLoader


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    output_dir: Path
    exercise_ids: tuple[int, ...]
    feature_method: str = "angles_v1"
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_channels: tuple[int, ...] = (64, 128)
    embedding_dim: int = 128
    margin: float = 1.0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    patience: int = 10
    num_workers: int = 0
    device: str = "auto"
    split_mode: str = "subject"


class ExercisePairDataset(Dataset):
    def __init__(self, records: list[dict], template_tensor: torch.Tensor, preprocessor: UIPRMDPreprocessor, extractor: MotionFeatureExtractor):
        self.template = template_tensor.detach().clone().float()
        self.samples: list[dict[str, torch.Tensor | float | str | int]] = []

        for record in records:
            user_tensor = self._record_to_tensor(record, preprocessor, extractor)
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
    def _record_to_tensor(record: dict, preprocessor: UIPRMDPreprocessor, extractor: MotionFeatureExtractor) -> torch.Tensor:
        aligned = preprocessor.align_vicon_to_mediapipe(record["sequence"])
        processed = preprocessor.process(aligned)
        features = extractor.build(processed)
        return torch.from_numpy(features).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


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


def split_by_subject(records: list[dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    if not records:
        return [], [], []

    subjects = sorted({int(record["subject_id"]) for record in records})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    total = len(subjects)

    if total == 1:
        subject_to_split = {subjects[0]: "train"}
    else:
        raw = np.array([max(0.0, train_ratio), max(0.0, val_ratio), max(0.0, test_ratio)], dtype=np.float64)
        if raw.sum() <= 0:
            raw = np.array([0.8, 0.1, 0.1], dtype=np.float64)
        ratios = raw / raw.sum()
        n_train = max(1, int(round(total * ratios[0])))
        n_val = max(1, int(round(total * ratios[1]))) if total >= 3 else 0
        n_test = max(1, total - n_train - n_val) if total >= 3 else max(0, total - n_train - n_val)

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

        train_subjects = set(subjects[:n_train])
        val_subjects = set(subjects[n_train : n_train + n_val])
        test_subjects = set(subjects[n_train + n_val : n_train + n_val + n_test])

        subject_to_split = {s: "train" for s in train_subjects}
        subject_to_split.update({s: "val" for s in val_subjects})
        subject_to_split.update({s: "test" for s in test_subjects})

    train_records, val_records, test_records = [], [], []
    for record in records:
        split_name = subject_to_split.get(int(record["subject_id"]), "train")
        if split_name == "train":
            train_records.append(record)
        elif split_name == "val":
            val_records.append(record)
        else:
            test_records.append(record)
    return train_records, val_records, test_records


def build_template_tensor(records: list[dict], preprocessor: UIPRMDPreprocessor, extractor: MotionFeatureExtractor) -> torch.Tensor:
    correct_records = [record for record in records if int(record["label"]) == 0]
    if not correct_records:
        raise ValueError("No correct samples available to build a template.")

    tensors = []
    for record in correct_records:
        aligned = preprocessor.align_vicon_to_mediapipe(record["sequence"])
        processed = preprocessor.process(aligned)
        tensors.append(torch.from_numpy(extractor.build(processed)).float())
    return torch.stack(tensors, dim=0).mean(dim=0)


def collate_batch(batch: list[dict]) -> dict:
    return {
        "template": torch.stack([item["template"] for item in batch], dim=0),
        "user": torch.stack([item["user"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "exercise_id": torch.tensor([item["exercise_id"] for item in batch], dtype=torch.long),
        "subject_id": torch.tensor([item["subject_id"] for item in batch], dtype=torch.long),
        "file": [item["file"] for item in batch],
    }


def train_one_exercise(
    exercise_id: int,
    cfg: TrainingConfig,
    extractor: MotionFeatureExtractor,
    records: list[dict] | None = None,
    split_plan: SplitPlan | None = None,
) -> dict[str, float]:
    set_seed(cfg.seed + exercise_id)

    loader = UIPRMDLoader(cfg.data_root)
    preprocessor = UIPRMDPreprocessor()
    if records is None:
        records = loader.load_vicon_data(exercise_id=exercise_id)
    if not records:
        raise ValueError(f"No records found for exercise {exercise_id}.")

    if split_plan is None:
        train_records, val_records, test_records = split_by_subject(records, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed)
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

    template_tensor = build_template_tensor(train_records, preprocessor, extractor)
    train_dataset = ExercisePairDataset(train_records, template_tensor, preprocessor, extractor)
    val_dataset = ExercisePairDataset(val_records if val_records else train_records, template_tensor, preprocessor, extractor)
    test_dataset = ExercisePairDataset(test_records if test_records else val_records if val_records else train_records, template_tensor, preprocessor, extractor)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_batch)

    device = resolve_device(cfg.device)
    model = ModelFactory().create_evaluator(
        in_channels=extractor.in_channels,
        hidden_channels=cfg.hidden_channels,
        embedding_dim=cfg.embedding_dim,
        device=device,
    )
    criterion = LossFactory().create_contrastive(margin=cfg.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Use the shared Trainer for the epoch loop, evaluation and checkpointing
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
        template_xyz_tensor=None,
        delta_criterion=None,
        delta_weight=getattr(cfg, "delta_weight", 0.1),
        rom_criterion=None,
        fold_name=fold_name,
        held_out_subject=held_out_subject,
        feature_method=getattr(cfg, "feature_method", None),
        in_channels=extractor.in_channels,
        hidden_channels=cfg.hidden_channels,
        embedding_dim=cfg.embedding_dim,
        use_phase_decoder=False,
        feature_channels=None,
    )

    return summary


def parse_exercise_ids(raw_values: list[int] | None) -> tuple[int, ...]:
    if not raw_values:
        return tuple(range(10))
    return tuple(int(v) for v in raw_values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train per-exercise ST-GAT using pluggable feature methods (factory).")
    parser.add_argument("--data-root", type=Path, default=Path("Datasets") / "UIPRMD")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints") / "uiprmd_factory")
    parser.add_argument("--exercise-ids", type=int, nargs="*", default=None)
    parser.add_argument("--feature-method", type=str, default="angles_v1", choices=["baseline_xyz", "angles_v1", "baseline", "angles"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split-mode", type=str, default="subject", choices=["subject", "loso"])
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        exercise_ids=parse_exercise_ids(args.exercise_ids),
        feature_method=args.feature_method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_channels=tuple(args.hidden_channels),
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        split_mode=args.split_mode,
    )

    extractor = FeatureExtractorFactory.create(cfg.feature_method)
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
                summary = train_one_exercise(exercise_id, cfg, extractor, records=exercise_records, split_plan=split_plan)
                summaries.append(summary)
                print(json.dumps(summary, indent=2))
        else:
            summary = train_one_exercise(exercise_id, cfg, extractor)
            summaries.append(summary)
            print(json.dumps(summary, indent=2))

    (cfg.output_dir / "all_exercises_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
