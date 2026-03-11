"""
Training callbacks for KinetoCheck.

Callbacks are hooks invoked at key points during training.
Implement ``TrainingCallback`` to add custom behaviour (logging, checkpointing,
early stopping, etc.) without touching the training loop.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


# ── Metrics container passed to every callback ──────────────────────

@dataclass
class TrainingMetrics:
    """Snapshot of training state shared with callbacks each epoch."""

    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    lr: float = 0.0
    epoch_time: float = 0.0
    elapsed: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ── Base callback ────────────────────────────────────────────────────

class TrainingCallback(ABC):
    """Override any of the hooks you need."""

    def on_train_begin(self, **kwargs: Any) -> None:
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, metrics: TrainingMetrics) -> None:
        pass

    @property
    def should_stop(self) -> bool:
        """Return True to request early termination."""
        return False


# ── Early stopping ───────────────────────────────────────────────────

class EarlyStopping(TrainingCallback):
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        patience:  How many epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        monitor:   Metric name in ``TrainingMetrics`` (default ``val_loss``).
        mode:      ``"min"`` (lower is better) or ``"max"`` (higher is better).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best: Optional[float] = None
        self.counter: int = 0
        self._stop: bool = False

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def on_epoch_end(self, metrics: TrainingMetrics) -> None:
        current = getattr(metrics, self.monitor, None)
        if current is None:
            return

        if self._is_improvement(current):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(
                    f"  ⏹  Early stopping: no improvement in "
                    f"{self.monitor} for {self.patience} epochs "
                    f"(best={self.best:.4f})"
                )
                self._stop = True

    @property
    def should_stop(self) -> bool:
        return self._stop


# ── Best-model checkpoint ────────────────────────────────────────────

class ModelCheckpoint(TrainingCallback):
    """
    Save model weights whenever a monitored metric improves.
    """

    def __init__(
        self,
        save_path: str,
        model: Any,  # BaseMovementModel
        monitor: str = "val_acc",
        mode: str = "max",
    ):
        self.save_path = save_path
        self.model = model
        self.monitor = monitor
        self.mode = mode
        self.best: Optional[float] = None

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best
        return current > self.best

    def on_epoch_end(self, metrics: TrainingMetrics) -> None:
        current = getattr(metrics, self.monitor, None)
        if current is None:
            return

        if self._is_improvement(current):
            self.best = current
            self.model.save_weights(self.save_path)
            print(f"  💾 Checkpoint: {self.monitor}={current:.4f} → {self.save_path}")


# ── LR-on-plateau scheduler wrapper ─────────────────────────────────

class ReduceLROnPlateauCallback(TrainingCallback):
    """
    Wraps ``torch.optim.lr_scheduler.ReduceLROnPlateau`` as a callback.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        monitor: str = "val_loss",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
    ):
        self.monitor = monitor
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if "loss" in monitor else "max",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    def on_epoch_end(self, metrics: TrainingMetrics) -> None:
        current = getattr(metrics, self.monitor, None)
        if current is not None:
            old_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            self.scheduler.step(current)
            new_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                print(f"  📉 LR reduced: {old_lr:.2e} → {new_lr:.2e}")


# ── Callback runner ──────────────────────────────────────────────────

class CallbackRunner:
    """Aggregates multiple callbacks and dispatches events."""

    def __init__(self, callbacks: List[TrainingCallback] | None = None):
        self.callbacks: List[TrainingCallback] = callbacks or []

    def add(self, cb: TrainingCallback) -> None:
        self.callbacks.append(cb)

    def on_train_begin(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(**kwargs)

    def on_epoch_end(self, metrics: TrainingMetrics) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(metrics)

    @property
    def should_stop(self) -> bool:
        return any(cb.should_stop for cb in self.callbacks)
