from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class Trainer:
    """Reusable trainer encapsulating the training loop, evaluation and checkpointing.

    This keeps the epoch/validation/test loop in a single place so different
    CLI wrappers can reuse the same logic (baseline, factory, phase-aware).
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ----- metrics helpers (copied small, stable implementations) -----
    @staticmethod
    def classification_metrics(scores: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
        preds = (scores >= threshold).float()
        targets = targets.float()

        tp = float(((preds == 1) & (targets == 1)).sum())
        tn = float(((preds == 0) & (targets == 0)).sum())
        fp = float(((preds == 1) & (targets == 0)).sum())
        fn = float(((preds == 0) & (targets == 1)).sum())

        accuracy = (tp + tn) / max(1.0, tp + tn + fp + fn)
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def search_best_threshold(scores: torch.Tensor, targets: torch.Tensor) -> tuple[float, dict]:
        if scores.numel() == 0:
            return 0.5, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        lo = float(scores.min().item())
        hi = float(scores.max().item())
        if math.isclose(lo, hi):
            return lo, Trainer.classification_metrics(scores, targets, lo)

        best_thr, best_met = lo, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for thr in torch.linspace(lo, hi, steps=101):
            met = Trainer.classification_metrics(scores, targets, float(thr.item()))
            if met["f1"] > best_met["f1"]:
                best_met, best_thr = met, float(thr.item())
        return best_thr, best_met

    def evaluate(self, model, loader, criterion, device: torch.device, delta_criterion=None, delta_weight: float = 0.1, rom_criterion=None) -> dict:
        model.eval()
        losses, scores, targets = [], [], []

        with torch.no_grad():
            for batch in loader:
                template = batch["template"].to(device)
                user = batch["user"].to(device)
                target = batch["target"].to(device)

                outputs = model(template, user)
                loss = criterion(outputs["template_embedding"], outputs["user_embedding"], target)

                if delta_criterion is not None and "correction_delta" in outputs:
                    d_loss = delta_criterion(
                        outputs["correction_delta"], outputs.get("warped_template_xyz"), user, target
                    )
                    loss = loss + delta_weight * d_loss

                if rom_criterion is not None:
                    loss = loss + rom_criterion(user, template, target)

                losses.append(float(loss.item()))
                scores.append(outputs["similarity_score"].detach().cpu())
                targets.append(target.detach().cpu())

        scores_t = torch.cat(scores, dim=0) if scores else torch.empty(0)
        targets_t = torch.cat(targets, dim=0) if targets else torch.empty(0)
        threshold, metrics = Trainer.search_best_threshold(scores_t, targets_t)

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "threshold": threshold,
            "scores": scores_t,
            "targets": targets_t,
            **metrics,
        }

    # ----- main reusable training loop -----
    def train(
        self,
        exercise_id: int,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        device: torch.device,
        output_dir: Path,
        template_tensor,
        template_xyz_tensor=None,
        delta_criterion=None,
        delta_weight: float = 0.1,
        rom_criterion=None,
        fold_name: str = "subject_split",
        held_out_subject: Optional[int] = None,
        feature_method: Optional[str] = None,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[list] = None,
        embedding_dim: Optional[int] = None,
        use_phase_decoder: bool = False,
        feature_channels: Optional[list] = None,
    ) -> dict:
        exercise_dir = output_dir / f"exercise_{exercise_id + 1:02d}" / fold_name
        exercise_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = exercise_dir / "metrics.csv"
        best_checkpoint_path = exercise_dir / "best_checkpoint.pt"
        last_checkpoint_path = exercise_dir / "last_checkpoint.pt"
        history_path = exercise_dir / "history.json"

        history = []
        best_val_f1 = -1.0
        best_epoch = -1
        best_test_met = {}
        patience_ctr = 0

        epochs = getattr(self.cfg, "epochs", 30)

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []

            for batch in train_loader:
                template = batch["template"].to(device)
                user = batch["user"].to(device)
                target = batch["target"].to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(template, user)

                loss = criterion(outputs["template_embedding"], outputs["user_embedding"], target)

                if delta_criterion is not None and "correction_delta" in outputs:
                    d_loss = delta_criterion(
                        outputs["correction_delta"], outputs.get("warped_template_xyz"), user, target
                    )
                    loss = loss + delta_weight * d_loss

                if rom_criterion is not None:
                    loss = loss + rom_criterion(user, template, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            val_metrics = self.evaluate(model, val_loader, criterion, device, delta_criterion, delta_weight, rom_criterion)
            test_metrics = self.evaluate(model, test_loader, criterion, device, delta_criterion, delta_weight, rom_criterion)

            epoch_metrics = {
                "exercise_id": exercise_id,
                "epoch": epoch,
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
                "config": asdict(self.cfg) if hasattr(self.cfg, "__dataclass_fields__") else dict(self.cfg.__dict__),
                "metrics": epoch_metrics,
                "val_threshold": float(val_metrics["threshold"]),
            }

            # additional metadata for phase-aware / inference overlay
            if feature_channels is not None:
                checkpoint.update({
                    "template_xyz_tensor": template_xyz_tensor,
                    "in_channels": in_channels,
                    "hidden_channels": list(hidden_channels) if hidden_channels is not None else None,
                    "embedding_dim": embedding_dim,
                    "use_phase_decoder": use_phase_decoder,
                    "feature_channels": list(feature_channels),
                })

            torch.save(checkpoint, last_checkpoint_path)

            with metrics_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(epoch_metrics.keys()))
                writer.writeheader()
                writer.writerows(history)

            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            if float(val_metrics["f1"]) > best_val_f1:
                best_val_f1 = float(val_metrics["f1"]) 
                best_epoch = epoch
                best_test_met = {
                    "loss": float(test_metrics["loss"]),
                    "accuracy": float(test_metrics["accuracy"]),
                    "precision": float(test_metrics["precision"]),
                    "recall": float(test_metrics["recall"]),
                    "f1": float(test_metrics["f1"]),
                    "threshold": float(val_metrics["threshold"]),
                }
                torch.save(checkpoint, best_checkpoint_path)
                patience_ctr = 0
            else:
                patience_ctr += 1

            if getattr(self.cfg, "patience", 0) > 0 and patience_ctr >= getattr(self.cfg, "patience", 0):
                break

        summary = {
            "exercise_id": exercise_id,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "best_test_loss": best_test_met.get("loss", 0.0),
            "best_test_accuracy": best_test_met.get("accuracy", 0.0),
            "best_test_precision": best_test_met.get("precision", 0.0),
            "best_test_recall": best_test_met.get("recall", 0.0),
            "best_test_f1": best_test_met.get("f1", 0.0),
            "best_threshold": best_test_met.get("threshold", 0.5),
        }

        (exercise_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
