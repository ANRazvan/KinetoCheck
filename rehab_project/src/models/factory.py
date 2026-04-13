from __future__ import annotations

import importlib
from typing import Any, Dict

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


_MODEL_REGISTRY: Dict[str, str] = {
    "stgat": "src.models.stgat.STGATModel",
    "stgcn": "src.models.stgcn.STGCNModel",
}


def _load_symbol(target: str) -> Any:
    """Load a Python symbol from a dotted path: package.module.ClassName."""
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


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a model from Hydra config.

    Resolution order:
    1) cfg.model.target
    2) registry lookup by cfg.model.name
    """
    if "model" not in cfg:
        raise ValueError("Missing 'model' section in config.")

    model_cfg = cfg.model
    model_name = str(model_cfg.get("name", "")).lower()
    target = model_cfg.get("target") or _MODEL_REGISTRY.get(model_name)

    if not target:
        known = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unable to resolve model target for '{model_name}'. "
            f"Set model.target or use one of: {known}"
        )

    model_cls = _load_symbol(str(target))
    kwargs = _to_plain_dict(model_cfg)
    kwargs.pop("name", None)
    kwargs.pop("target", None)

    model = model_cls(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Constructed model from '{target}' is not torch.nn.Module.")

    return model
