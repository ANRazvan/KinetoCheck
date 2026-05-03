"""
Train one ST-GAT ExerciseEvaluator per UI-PRMD exercise.

ROOT CAUSE FIX — "always predicts correct"
==========================================
The original training had three compounding problems that caused the model
to output high similarity for every input regardless of whether it was
correct or not:

1. BLURRY TEMPLATE (mean of all correct samples)
   The template was the element-wise average of every correct execution.
   After z-scoring and averaging, this produces a "mean pose sequence" that
   sits near the centroid of the embedding space.  Any input that is even
   vaguely exercise-shaped ends up close to it in cosine space.
   FIX: Use the single most-representative correct sample (the one with the
   highest average cosine similarity to all other correct samples) as the
   template.  A sharp, real sequence makes a much harder target.

2. NO CROSS-EXERCISE NEGATIVES
   Each model only saw "correct vs incorrect of the same exercise."
   "Incorrect" in UI-PRMD means wrong form of the SAME exercise — not a
   completely different movement.  The model had zero training signal to
   distinguish "this is the wrong exercise entirely."
   FIX: For each exercise, sample a fraction of sequences from ALL OTHER
   exercises and add them as hard negatives (label = 0).  This forces the
   model to learn exercise identity, not just form quality.

3. CLASS IMBALANCE COLLAPSE
   UI-PRMD has many more "incorrect form" samples than "correct form."
   With contrastive loss, if negatives dominate, the easiest optimum is to
   push everything apart — then the threshold search finds a threshold near
   the top of the score distribution, making nearly everything "correct."
   FIX: Oversample positives (correct) to balance the dataset, and use a
   larger contrastive margin so negative pairs must be clearly separated.

Additional changes
------------------
- Added score distribution logging per epoch so you can see if collapse
  is happening (all scores the same → collapse).
- Template is now stored as the raw representative sequence, not the mean.
- Hard-negative ratio and positive oversampling are CLI-configurable.
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from phase_aware import ContrastiveLoss, DeltaRegressionLoss, ExerciseEvaluator
from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor
from Preprocessing.UIPRMD_loader import UIPRMDLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    output_dir: Path
    exercise_ids: tuple[int, ...]
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_channels: tuple[int, ...] = (64, 128)
    embedding_dim: int = 128
    # Larger margin forces negatives to be clearly separated
    margin: float = 0.5
    delta_weight: float = 0.05
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    patience: int = 15
    num_workers: int = 0
    device: str = "auto"
    use_phase_decoder: bool = True
    # Fraction of training samples to replace with cross-exercise hard negatives
    # e.g. 0.3 means ~30% of each batch will be wrong-exercise pairs (label=0)
    hard_negative_ratio: float = 0.3
    # How many times to repeat each positive (correct) sample to balance classes
    positive_oversample: int = 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ExercisePairDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        template_tensor: torch.Tensor,
        preprocessor: UIPRMDPreprocessor,
        hard_negative_records: list[dict] | None = None,
        hard_negative_ratio: float = 0.3,
        positive_oversample: int = 3,
        is_train: bool = True,
    ):
        self.template = template_tensor.detach().clone().float()
        self.samples: list[dict] = []

        # Cache processed tensors to avoid re-processing per epoch
        tensor_cache: dict[str, torch.Tensor] = {}

        def get_tensor(record: dict) -> torch.Tensor:
            key = str(record["file"])
            if key not in tensor_cache:
                tensor_cache[key] = self._record_to_tensor(record, preprocessor)
            return tensor_cache[key]

        # --- Same-exercise samples (correct label=0 → target=1.0, incorrect label!=0 → target=0.0) ---
        same_ex_samples = []
        for record in records:
            user_tensor = get_tensor(record)
            target = 1.0 if int(record["label"]) == 0 else 0.0
            same_ex_samples.append({
                "template": self.template,
                "user": user_tensor,
                "target": torch.tensor(target, dtype=torch.float32),
                "label": int(record["label"]),
                "exercise_id": int(record["exercise_id"]),
                "subject_id": int(record["subject_id"]),
                "file": str(record["file"]),
            })

        if is_train and positive_oversample > 1:
            # Oversample correct samples to balance against (more numerous) incorrect ones
            positives = [s for s in same_ex_samples if s["target"].item() == 1.0]
            negatives = [s for s in same_ex_samples if s["target"].item() == 0.0]
            balanced = negatives + positives * positive_oversample
            random.shuffle(balanced)
            self.samples.extend(balanced)
        else:
            self.samples.extend(same_ex_samples)

        # --- Cross-exercise hard negatives (only for training) ---
        if is_train and hard_negative_records and hard_negative_ratio > 0:
            n_hard = int(len(self.samples) * hard_negative_ratio)
            hard_pool = hard_negative_records.copy()
            random.shuffle(hard_pool)
            for record in hard_pool[:n_hard]:
                user_tensor = get_tensor(record)
                self.samples.append({
                    "template": self.template,
                    "user": user_tensor,
                    "target": torch.tensor(0.0, dtype=torch.float32),  # always negative
                    "label": -1,   # sentinel: cross-exercise hard negative
                    "exercise_id": int(record["exercise_id"]),
                    "subject_id": int(record.get("subject_id", -1)),
                    "file": str(record["file"]),
                })

        random.shuffle(self.samples)

    @staticmethod
    def _record_to_tensor(
        record: dict, preprocessor: UIPRMDPreprocessor
    ) -> torch.Tensor:
        aligned = preprocessor.align_vicon_to_mediapipe(record["sequence"])
        processed = preprocessor.process(aligned)

        velocity     = np.diff(processed, axis=0, prepend=processed[:1])
        acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])

        features = np.concatenate([processed, velocity, acceleration], axis=-1)  # (T, 17, 9)
        features = np.transpose(features, (2, 0, 1)).copy()                      # (9, T, 17)
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


def _record_to_embedding(
    record: dict,
    preprocessor: UIPRMDPreprocessor,
    model: ExerciseEvaluator,
    device: torch.device,
) -> torch.Tensor:
    """Process a single record to a normalised embedding (no grad)."""
    aligned   = preprocessor.align_vicon_to_mediapipe(record["sequence"])
    processed = preprocessor.process(aligned)
    velocity     = np.diff(processed, axis=0, prepend=processed[:1])
    acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
    features = np.concatenate([processed, velocity, acceleration], axis=-1)
    features = np.transpose(features, (2, 0, 1)).copy()
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb, _ = model.encode(tensor, return_attentions=False)
    return emb.squeeze(0).cpu()


def build_template_tensor(
    records: list[dict],
    preprocessor: UIPRMDPreprocessor,
) -> torch.Tensor:
    """
    Choose the single most-representative correct sample as the template.

    We pick the correct sample whose processed feature tensor has the highest
    average cosine similarity to all other correct samples.  A real, sharp
    sequence makes a much better discriminative target than the blurry mean.

    Falls back to simple mean if there is only one correct sample.
    """
    correct_records = [r for r in records if int(r["label"]) == 0]
    if not correct_records:
        raise ValueError("No correct samples to build a template.")

    tensors = []
    for r in correct_records:
        aligned  = preprocessor.align_vicon_to_mediapipe(r["sequence"])
        processed = preprocessor.process(aligned)
        velocity     = np.diff(processed, axis=0, prepend=processed[:1])
        acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
        features = np.concatenate([processed, velocity, acceleration], axis=-1)
        tensors.append(
            torch.from_numpy(np.transpose(features, (2, 0, 1)).copy()).float()
        )

    if len(tensors) == 1:
        return tensors[0]

    # Pool each tensor to a vector and find the medoid
    vecs = torch.stack([t.mean(dim=(1, 2)) for t in tensors], dim=0)   # (N, C)
    vecs = F.normalize(vecs, p=2, dim=-1)
    sim_matrix = vecs @ vecs.T    # (N, N)
    # Best = highest mean similarity to all others (medoid)
    mean_sim = sim_matrix.mean(dim=1)
    best_idx = int(mean_sim.argmax().item())
    print(f"  Template: medoid sample {best_idx+1}/{len(tensors)} "
          f"(mean cosine sim to others: {mean_sim[best_idx]:.3f})")
    return tensors[best_idx]


def _resample_to(arr: np.ndarray, target_T: int) -> np.ndarray:
    T_i = arr.shape[0]
    if T_i == target_T:
        return arr
    if T_i == 1:
        return np.repeat(arr, target_T, axis=0)
    original_trailing = arr.shape[1:]
    flat = arr.reshape(T_i, -1)
    src_t = np.linspace(0.0, 1.0, T_i,      dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_T, dtype=np.float32)
    out = np.zeros((target_T, flat.shape[1]), dtype=np.float32)
    for feat in range(flat.shape[1]):
        out[:, feat] = np.interp(dst_t, src_t, flat[:, feat])
    return out.reshape(target_T, *original_trailing)


def build_raw_xyz_template(records: list[dict], preprocessor: UIPRMDPreprocessor) -> torch.Tensor:
    """
    Build a template in RAW un-preprocessed XY coordinates for the overlay.

    For MediaPipe datasets, record["sequence"] already contains raw
    image-fraction XY in shape (T, 17, 3) or flat (T, 51).
    For Vicon datasets, align first and then normalise each axis to [0, 1]
    so the overlay stays in a screen-like coordinate range.
    Variable-length sequences are resampled to the median length.

    Returns torch.Tensor shape (3, T, J).
    """
    correct_records = [r for r in records if int(r["label"]) == 0]
    if not correct_records:
        raise ValueError("No correct samples to build raw XYZ template.")

    raw_list: list[np.ndarray] = []
    using_raw_image_fractions = True

    for r in correct_records:
        seq = np.asarray(r["sequence"], dtype=np.float32)
        if seq.ndim == 3 and seq.shape[1] == 17:
            raw_list.append(seq)
        elif seq.ndim == 2 and seq.shape[1] == 51:
            raw_list.append(seq.reshape(seq.shape[0], 17, 3))
        else:
            aligned = preprocessor.align_vicon_to_mediapipe(seq)
            axis_min = aligned.min(axis=(0, 1), keepdims=True)
            axis_max = aligned.max(axis=(0, 1), keepdims=True)
            aligned = (aligned - axis_min) / np.maximum(axis_max - axis_min, 1e-6)
            raw_list.append(np.clip(aligned, 0.0, 1.0))

    lengths  = [a.shape[0] for a in raw_list]
    target_T = int(np.median(lengths))

    tensors: list[torch.Tensor] = []
    for seq_arr in raw_list:
        resampled = _resample_to(seq_arr, target_T)
        xyz = np.transpose(resampled, (2, 0, 1)).copy()
        tensors.append(torch.from_numpy(xyz).float())

    return torch.stack(tensors, dim=0).mean(dim=0)


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
# Evaluation
# ---------------------------------------------------------------------------

def search_best_threshold(
    scores: torch.Tensor, targets: torch.Tensor
) -> tuple[float, dict[str, float]]:
    if scores.numel() == 0:
        return 0.5, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    lo = float(scores.min().item())
    hi = float(scores.max().item())

    if math.isclose(lo, hi):
        return lo, classification_metrics(scores, targets, lo)

    best_thr, best_met = lo, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for thr in torch.linspace(lo, hi, steps=201):
        met = classification_metrics(scores, targets, float(thr.item()))
        if met["f1"] > best_met["f1"]:
            best_met, best_thr = met, float(thr.item())

    return best_thr, best_met


def classification_metrics(
    scores: torch.Tensor, targets: torch.Tensor, threshold: float
) -> dict[str, float]:
    preds   = (scores >= threshold).float()
    targets = targets.float()

    tp = float(((preds == 1) & (targets == 1)).sum())
    tn = float(((preds == 0) & (targets == 0)).sum())
    fp = float(((preds == 1) & (targets == 0)).sum())
    fn = float(((preds == 0) & (targets == 1)).sum())

    accuracy  = (tp + tn) / max(1.0, tp + tn + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall    = tp / max(1.0, tp + fn)
    f1        = 2.0 * precision * recall / max(1e-8, precision + recall)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate(
    model: ExerciseEvaluator,
    loader: DataLoader,
    criterion: ContrastiveLoss,
    delta_criterion: DeltaRegressionLoss | None,
    device: torch.device,
    delta_weight: float = 0.05,
) -> dict[str, float | torch.Tensor]:
    model.eval()
    losses, scores, targets = [], [], []

    with torch.no_grad():
        for batch in loader:
            template = batch["template"].to(device)
            user     = batch["user"].to(device)
            target   = batch["target"].to(device)

            outputs = model(template, user)
            loss = criterion(
                outputs["template_embedding"], outputs["user_embedding"], target
            )

            if delta_criterion is not None and "correction_delta" in outputs:
                d_loss = delta_criterion(
                    outputs["correction_delta"],
                    outputs["warped_template_xyz"],
                    user,
                    target,
                )
                loss = loss + delta_weight * d_loss

            losses.append(float(loss.item()))
            scores.append(outputs["similarity_score"].detach().cpu())
            targets.append(target.detach().cpu())

    scores_t  = torch.cat(scores,  dim=0) if scores  else torch.empty(0)
    targets_t = torch.cat(targets, dim=0) if targets else torch.empty(0)
    threshold, metrics = search_best_threshold(scores_t, targets_t)

    return {
        "loss":       float(np.mean(losses)) if losses else 0.0,
        "threshold":  threshold,
        "score_mean": float(scores_t.mean().item()) if scores_t.numel() > 0 else 0.0,
        "score_std":  float(scores_t.std().item())  if scores_t.numel() > 0 else 0.0,
        "scores":     scores_t,
        "targets":    targets_t,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Per-exercise training
# ---------------------------------------------------------------------------

def load_all_records(
    loader_ds: UIPRMDLoader, all_exercise_ids: tuple[int, ...]
) -> dict[int, list[dict]]:
    """Load records for every exercise — used to sample cross-exercise negatives."""
    all_records: dict[int, list[dict]] = {}
    for eid in all_exercise_ids:
        try:
            recs = loader_ds.load_vicon_data(exercise_id=eid)
            all_records[eid] = recs
        except Exception:
            all_records[eid] = []
    return all_records


def train_one_exercise(
    exercise_id: int,
    cfg: TrainingConfig,
    all_records_by_exercise: dict[int, list[dict]] | None = None,
) -> dict[str, float]:
    set_seed(cfg.seed + exercise_id)

    loader_ds    = UIPRMDLoader(cfg.data_root)
    preprocessor = UIPRMDPreprocessor()
    records = loader_ds.load_vicon_data(exercise_id=exercise_id)
    if not records:
        raise ValueError(f"No records found for exercise {exercise_id}.")

    train_records, val_records, test_records = split_by_subject(
        records, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )
    if not train_records:
        raise ValueError(f"Training split is empty for exercise {exercise_id}.")

    # --- Build template (medoid of correct samples, not blurry mean) ---
    template_tensor     = build_template_tensor(train_records, preprocessor)
    template_xyz_tensor = build_raw_xyz_template(train_records, preprocessor)

    # --- Cross-exercise hard negatives ---
    hard_negative_records: list[dict] = []
    if all_records_by_exercise is not None and cfg.hard_negative_ratio > 0:
        for other_id, other_records in all_records_by_exercise.items():
            if other_id == exercise_id:
                continue
            # Take only correct-form samples from other exercises as hard negatives
            # (correct form of the wrong exercise is a harder negative than wrong form)
            correct_others = [r for r in other_records if int(r["label"]) == 0]
            hard_negative_records.extend(correct_others)
        random.Random(cfg.seed).shuffle(hard_negative_records)
        print(f"  Exercise {exercise_id}: {len(hard_negative_records)} cross-exercise hard negatives available.")

    train_dataset = ExercisePairDataset(
        train_records, template_tensor, preprocessor,
        hard_negative_records=hard_negative_records,
        hard_negative_ratio=cfg.hard_negative_ratio,
        positive_oversample=cfg.positive_oversample,
        is_train=True,
    )
    val_dataset = ExercisePairDataset(
        val_records if val_records else train_records,
        template_tensor, preprocessor, is_train=False,
    )
    test_dataset = ExercisePairDataset(
        test_records if test_records else val_records if val_records else train_records,
        template_tensor, preprocessor, is_train=False,
    )

    print(f"  Exercise {exercise_id}: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    pos = sum(1 for s in train_dataset.samples if s["target"].item() == 1.0)
    neg = sum(1 for s in train_dataset.samples if s["target"].item() == 0.0)
    print(f"  Train class balance: {pos} positive, {neg} negative ({pos/(pos+neg)*100:.1f}% pos)")

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
    model = ExerciseEvaluator(
        in_channels=9,
        hidden_channels=cfg.hidden_channels,
        embedding_dim=cfg.embedding_dim,
        use_phase_decoder=cfg.use_phase_decoder,
    ).to(device)

    criterion       = ContrastiveLoss(margin=cfg.margin)
    delta_criterion = DeltaRegressionLoss() if cfg.use_phase_decoder else None
    optimizer       = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    # Cosine annealing so LR doesn't stay stuck high
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate / 20
    )

    exercise_dir         = cfg.output_dir / f"exercise_{exercise_id + 1:02d}"
    exercise_dir.mkdir(parents=True, exist_ok=True)
    metrics_path         = exercise_dir / "metrics.csv"
    best_checkpoint_path = exercise_dir / "best_checkpoint.pt"
    last_checkpoint_path = exercise_dir / "last_checkpoint.pt"
    history_path         = exercise_dir / "history.json"

    history: list[dict] = []
    best_val_f1   = -1.0
    best_epoch    = -1
    best_test_met: dict[str, float] = {}
    patience_ctr  = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []
        all_train_scores: list[float] = []

        for batch in train_loader:
            template = batch["template"].to(device)
            user     = batch["user"].to(device)
            target   = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(template, user)

            loss = criterion(
                outputs["template_embedding"], outputs["user_embedding"], target
            )
            if delta_criterion is not None and "correction_delta" in outputs:
                d_loss = delta_criterion(
                    outputs["correction_delta"],
                    outputs["warped_template_xyz"],
                    user,
                    target,
                )
                loss = loss + cfg.delta_weight * d_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            all_train_scores.extend(
                outputs["similarity_score"].detach().cpu().tolist()
            )

        scheduler.step()

        train_loss   = float(np.mean(train_losses)) if train_losses else 0.0
        score_mean   = float(np.mean(all_train_scores))
        score_std    = float(np.std(all_train_scores))
        val_metrics  = evaluate(model, val_loader,  criterion, delta_criterion, device, cfg.delta_weight)
        test_metrics = evaluate(model, test_loader, criterion, delta_criterion, device, cfg.delta_weight)

        epoch_metrics = {
            "exercise_id":    exercise_id,
            "epoch":          epoch,
            "train_loss":     train_loss,
            # Score distribution diagnostics — std near 0 means collapse
            "train_score_mean": score_mean,
            "train_score_std":  score_std,
            "val_loss":       float(val_metrics["loss"]),
            "val_score_mean": float(val_metrics["score_mean"]),
            "val_score_std":  float(val_metrics["score_std"]),
            "val_threshold":  float(val_metrics["threshold"]),
            "val_accuracy":   float(val_metrics["accuracy"]),
            "val_precision":  float(val_metrics["precision"]),
            "val_recall":     float(val_metrics["recall"]),
            "val_f1":         float(val_metrics["f1"]),
            "test_loss":      float(test_metrics["loss"]),
            "test_accuracy":  float(test_metrics["accuracy"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall":    float(test_metrics["recall"]),
            "test_f1":        float(test_metrics["f1"]),
        }
        history.append(epoch_metrics)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Ex{exercise_id} ep{epoch:3d}: loss={train_loss:.4f} "
                f"score_mean={score_mean:.3f} score_std={score_std:.3f} "
                f"val_f1={val_metrics['f1']:.3f} thr={val_metrics['threshold']:.3f}"
            )
            if score_std < 0.02:
                print(f"  [WARN] Ex{exercise_id}: score std={score_std:.4f} — possible collapse!")

        checkpoint = {
            "exercise_id":          exercise_id,
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "template_tensor":      template_tensor,
            "config":               asdict(cfg),
            "metrics":              epoch_metrics,
            "val_threshold":        float(val_metrics["threshold"]),
            "template_xyz_tensor":  template_xyz_tensor,
            "in_channels":          9,
            "hidden_channels":      list(cfg.hidden_channels),
            "embedding_dim":        cfg.embedding_dim,
            "use_phase_decoder":    cfg.use_phase_decoder,
            "feature_channels":     ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"],
            "preprocessor_config":  {
                "align_method":        "vicon_to_mediapipe",
                "num_joints":          17,
                "template_xyz_is_raw": True,
            },
        }

        torch.save(checkpoint, last_checkpoint_path)

        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_metrics.keys()))
            writer.writeheader()
            writer.writerows(history)

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1  = float(val_metrics["f1"])
            best_epoch   = epoch
            best_test_met = {
                "loss":      float(test_metrics["loss"]),
                "accuracy":  float(test_metrics["accuracy"]),
                "precision": float(test_metrics["precision"]),
                "recall":    float(test_metrics["recall"]),
                "f1":        float(test_metrics["f1"]),
                "threshold": float(val_metrics["threshold"]),
            }
            torch.save(checkpoint, best_checkpoint_path)
            patience_ctr = 0
        else:
            patience_ctr += 1

        if cfg.patience > 0 and patience_ctr >= cfg.patience:
            print(f"  Ex{exercise_id}: early stop at epoch {epoch} (best val_f1={best_val_f1:.3f})")
            break

    summary = {
        "exercise_id":         exercise_id,
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
    parser.add_argument("--data-root",            type=Path,  default=Path("Datasets") / "UIPRMD")
    parser.add_argument("--output-dir",           type=Path,  default=Path("checkpoints") / "uiprmd")
    parser.add_argument("--exercise-ids",         type=int,   nargs="*", default=None)
    parser.add_argument("--epochs",               type=int,   default=50)
    parser.add_argument("--batch-size",           type=int,   default=16)
    parser.add_argument("--learning-rate",        type=float, default=3e-4)
    parser.add_argument("--weight-decay",         type=float, default=1e-4)
    parser.add_argument("--embedding-dim",        type=int,   default=128)
    parser.add_argument("--hidden-channels",      type=int,   nargs="*", default=[64, 128])
    parser.add_argument("--margin",               type=float, default=0.5,
                        help="Contrastive loss margin (default 0.5 — smaller = harder separation).")
    parser.add_argument("--delta-weight",         type=float, default=0.05)
    parser.add_argument("--train-ratio",          type=float, default=0.8)
    parser.add_argument("--val-ratio",            type=float, default=0.1)
    parser.add_argument("--test-ratio",           type=float, default=0.1)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--patience",             type=int,   default=15)
    parser.add_argument("--num-workers",          type=int,   default=0)
    parser.add_argument("--device",               type=str,   default="auto")
    parser.add_argument("--hard-negative-ratio",  type=float, default=0.3,
                        help="Fraction of training samples replaced with cross-exercise hard negatives.")
    parser.add_argument("--positive-oversample",  type=int,   default=3,
                        help="How many times to repeat each positive sample to balance classes.")
    parser.add_argument("--no-phase-decoder",     action="store_true",
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
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        hard_negative_ratio=args.hard_negative_ratio,
        positive_oversample=args.positive_oversample,
        use_phase_decoder=not args.no_phase_decoder,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    loader_ds = UIPRMDLoader(cfg.data_root)
    print("Pre-loading all exercise records for cross-exercise hard negatives...")
    all_records_by_exercise = load_all_records(loader_ds, cfg.exercise_ids)

    summaries = []
    for exercise_id in cfg.exercise_ids:
        print(f"\n=== Exercise {exercise_id} ===")
        summary = train_one_exercise(exercise_id, cfg, all_records_by_exercise)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    (cfg.output_dir / "all_exercises_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()