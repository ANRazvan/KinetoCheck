from __future__ import annotations

import importlib
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


_DATASET_REGISTRY: Dict[str, str] = {
    "uiprmd": "src.data.dataloaders.UIPRMDDataset",
    "intellirehab": "src.data.dataloaders.IntelliRehabDataset",
}


def _load_symbol(target: str) -> Any:
    if "." not in target:
        raise ValueError(f"Invalid target '{target}'. Expected dotted path.")

    module_path, symbol_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f"Symbol '{symbol_name}' not found in '{module_path}'.") from exc


def _to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _build_dataset(cfg: DictConfig, split: str) -> Dataset:
    dataset_cfg = cfg.dataset
    dataset_name = str(dataset_cfg.get("name", "")).lower()
    target = dataset_cfg.get("target") or _DATASET_REGISTRY.get(dataset_name)

    if not target:
        known = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unable to resolve dataset target for '{dataset_name}'. "
            f"Set dataset.target or use one of: {known}"
        )

    dataset_cls = _load_symbol(str(target))
    kwargs = _to_plain_dict(dataset_cfg)

    kwargs.pop("name", None)
    kwargs.pop("target", None)

    splits_cfg = kwargs.pop("splits", {})
    split_name = split
    if isinstance(splits_cfg, dict):
        split_name = str(splits_cfg.get(split, split))

    kwargs["split"] = split_name

    if "preprocessing" in cfg:
        kwargs["preprocessing_cfg"] = _to_plain_dict(cfg.preprocessing)

    dataset = dataset_cls(**kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Constructed dataset from '{target}' is not torch.utils.data.Dataset.")

    return dataset


def build_dataloaders(cfg: DictConfig) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from Hydra config."""
    if "dataset" not in cfg:
        raise ValueError("Missing 'dataset' section in config.")
    if "training" not in cfg:
        raise ValueError("Missing 'training' section in config.")

    train_dataset = _build_dataset(cfg, split="train")
    val_dataset = _build_dataset(cfg, split="val")
    test_dataset = _build_dataset(cfg, split="test")

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.get("num_workers", 0))
    pin_memory = bool(cfg.training.get("pin_memory", False))
    shuffle_train = bool(cfg.training.get("shuffle", True))

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders
