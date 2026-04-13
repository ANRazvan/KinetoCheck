from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure src/ is importable when running `python scripts/train.py` from rehab_project/
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.factory import build_dataloaders
from src.models.factory import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.training.optimizer
    name = str(opt_cfg.name).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_criterion(cfg: DictConfig) -> torch.nn.Module:
    loss_cfg = cfg.training.loss
    name = str(loss_cfg.name).lower()

    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    if name == "mse":
        return torch.nn.MSELoss()

    raise ValueError(f"Unsupported loss function: {name}")


def build_scheduler(
    cfg: DictConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    sched_cfg = cfg.training.scheduler
    if not bool(sched_cfg.get("enabled", False)):
        return None

    name = str(sched_cfg.name).lower()
    if name == "cosine":
        t_max = int(sched_cfg.get("t_max", cfg.training.epochs))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=min_lr,
        )
    if name == "step":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    raise ValueError(f"Unsupported scheduler: {name}")


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        x = batch.get("keypoints") or batch.get("inputs") or batch.get("x")
        y = batch.get("label") or batch.get("labels") or batch.get("y")
        if x is None or y is None:
            raise KeyError("Dict batch must contain input and label keys.")
        return x, y

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise ValueError("Unsupported batch format. Expected dict or (inputs, labels).")


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct / max(total, 1))


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0

    for batch in loader:
        inputs, targets = unpack_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=(amp_enabled and device.type == "cuda"),
        ):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        total_acc += compute_accuracy(logits.detach(), targets)
        total_steps += 1

    if total_steps == 0:
        return 0.0, 0.0

    return total_loss / total_steps, total_acc / total_steps


def save_checkpoint(
    cfg: DictConfig,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    ckpt_dir = Path(str(cfg.training.checkpoint.dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        ckpt_path,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.project.seed))

    device = resolve_device(str(cfg.project.device))
    print(f"[INFO] Device: {device}")

    dataloaders = build_dataloaders(cfg)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg).to(device)
    scheduler = build_scheduler(cfg, optimizer)

    epochs = int(cfg.training.epochs)
    amp_enabled = bool(cfg.training.get("amp", False))
    save_every = int(cfg.training.checkpoint.get("save_every", 1))

    print(f"[INFO] Starting training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
        )

        if bool(cfg.evaluation.get("run_val_each_epoch", True)):
            with torch.no_grad():
                val_loss, val_acc = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    optimizer=None,
                    device=device,
                    amp_enabled=amp_enabled,
                )
        else:
            val_loss, val_acc = 0.0, 0.0

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if epoch % save_every == 0:
            save_checkpoint(cfg, epoch, model, optimizer)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
