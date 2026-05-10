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
from Models import ContrastiveLoss, ExerciseEvaluator
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


def classification_metrics(scores: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).float()
    targets = targets.float()
    tp = float(((predictions == 1) & (targets == 1)).sum().item())
    tn = float(((predictions == 0) & (targets == 0)).sum().item())
    fp = float(((predictions == 1) & (targets == 0)).sum().item())
    fn = float(((predictions == 0) & (targets == 1)).sum().item())
    accuracy = (tp + tn) / max(1.0, tp + tn + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def search_best_threshold(scores: torch.Tensor, targets: torch.Tensor) -> tuple[float, dict[str, float]]:
    if scores.numel() == 0:
        return 0.5, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    min_score = float(scores.min().item())
    max_score = float(scores.max().item())
    if math.isclose(min_score, max_score):
        return min_score, classification_metrics(scores, targets, min_score)
    best_threshold, best_metrics = min_score, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in torch.linspace(min_score, max_score, steps=101):
        metrics = classification_metrics(scores, targets, float(threshold.item()))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold, best_metrics = float(threshold.item()), metrics
    return best_threshold, best_metrics


def evaluate(model: ExerciseEvaluator, loader: DataLoader, criterion: ContrastiveLoss, device: torch.device) -> dict[str, float | torch.Tensor]:
    model.eval()
    losses, scores, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            template = batch["template"].to(device)
            user = batch["user"].to(device)
            target = batch["target"].to(device)
            outputs = model(template, user)
            loss = criterion(outputs["template_embedding"], outputs["user_embedding"], target)
            losses.append(float(loss.item()))
            scores.append(outputs["similarity_score"].detach().cpu())
            targets.append(target.detach().cpu())

    scores_tensor = torch.cat(scores, dim=0) if scores else torch.empty(0)
    targets_tensor = torch.cat(targets, dim=0) if targets else torch.empty(0)
    threshold, metrics = search_best_threshold(scores_tensor, targets_tensor)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "threshold": threshold,
        "scores": scores_tensor,
        "targets": targets_tensor,
        **metrics,
    }


def train_one_exercise(exercise_id: int, cfg: TrainingConfig, extractor: MotionFeatureExtractor) -> dict[str, float]:
    set_seed(cfg.seed + exercise_id)
    loader = UIPRMDLoader(cfg.data_root)
    preprocessor = UIPRMDPreprocessor()
    records = loader.load_vicon_data(exercise_id=exercise_id)
    if not records:
        raise ValueError(f"No records found for exercise {exercise_id}.")

    train_records, val_records, test_records = split_by_subject(records, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed)
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
    model = ExerciseEvaluator(in_channels=extractor.in_channels, hidden_channels=cfg.hidden_channels, embedding_dim=cfg.embedding_dim).to(device)
    criterion = ContrastiveLoss(margin=cfg.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    exercise_dir = cfg.output_dir / f"exercise_{exercise_id + 1:02d}"
    exercise_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exercise_dir / "metrics.csv"
    best_checkpoint_path = exercise_dir / "best_checkpoint.pt"
    last_checkpoint_path = exercise_dir / "last_checkpoint.pt"
    history_path = exercise_dir / "history.json"

    history: list[dict[str, float | int]] = []
    best_val_f1, best_epoch = -1.0, -1
    best_test_metrics: dict[str, float] = {}
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            template = batch["template"].to(device)
            user = batch["user"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(template, user)
            loss = criterion(outputs["template_embedding"], outputs["user_embedding"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        epoch_metrics = {
            "exercise_id": exercise_id,
            "epoch": epoch,
            "feature_method": extractor.name,
            "in_channels": extractor.in_channels,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_threshold": float(val_metrics["threshold"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
            "test_f1": float(test_metrics["f1"]),
        }
        history.append(epoch_metrics)

        checkpoint = {
            "exercise_id": exercise_id,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "template_tensor": template_tensor,
            "config": {**asdict(cfg), "feature_method": extractor.name, "in_channels": extractor.in_channels},
            "metrics": epoch_metrics,
            "val_threshold": float(val_metrics["threshold"]),
        }
        torch.save(checkpoint, last_checkpoint_path)

        with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(epoch_metrics.keys()))
            writer.writeheader()
            writer.writerows(history)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            best_test_metrics = {
                "loss": float(test_metrics["loss"]),
                "accuracy": float(test_metrics["accuracy"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "f1": float(test_metrics["f1"]),
                "threshold": float(val_metrics["threshold"]),
            }
            torch.save(checkpoint, best_checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if cfg.patience > 0 and patience_counter >= cfg.patience:
            break

    summary = {
        "exercise_id": exercise_id,
        "feature_method": extractor.name,
        "in_channels": extractor.in_channels,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_test_loss": best_test_metrics.get("loss", 0.0),
        "best_test_accuracy": best_test_metrics.get("accuracy", 0.0),
        "best_test_precision": best_test_metrics.get("precision", 0.0),
        "best_test_recall": best_test_metrics.get("recall", 0.0),
        "best_test_f1": best_test_metrics.get("f1", 0.0),
        "best_threshold": best_test_metrics.get("threshold", 0.5),
    }
    (exercise_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    )

    extractor = FeatureExtractorFactory.create(cfg.feature_method)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for exercise_id in cfg.exercise_ids:
        summary = train_one_exercise(exercise_id, cfg, extractor)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    (cfg.output_dir / "all_exercises_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
