from __future__ import annotations

from typing import Any, Dict

import torch


class _Singleton(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ModelFactory(metaclass=_Singleton):
    """Factory producing model instances.

    This is intentionally lightweight: it centralises model construction so the
    training code is clearer and creation logic (e.g. device casting) can be
    adapted in one place.
    """

    def create_evaluator(self, *, in_channels: int, hidden_channels: tuple[int, ...], embedding_dim: int, use_phase_decoder: bool = False, device: torch.device | None = None, **kwargs) -> Any:
        # Import lazily to avoid import-time side-effects in callers
        from . import stgat_temporal_pyramid as base_mod
        from . import stgat_temporal_pyramid_phase_aware as phase_mod

        if use_phase_decoder:
            Evaluator = getattr(phase_mod, "ExerciseEvaluator")
        else:
            Evaluator = getattr(base_mod, "ExerciseEvaluator")

        model = Evaluator(in_channels=in_channels, hidden_channels=hidden_channels, embedding_dim=embedding_dim, **{k: v for k, v in kwargs.items() if k in ("use_phase_decoder",)})
        if device is not None:
            model = model.to(device)
        return model


class LossFactory(metaclass=_Singleton):
    """Factory producing loss objects used in training/evaluation."""

    def create_contrastive(self, *, margin: float = 1.0):
        from .stgat_temporal_pyramid import ContrastiveLoss

        return ContrastiveLoss(margin=margin)

    def create_delta_regression(self):
        from .stgat_temporal_pyramid_phase_aware import DeltaRegressionLoss

        return DeltaRegressionLoss()

    def create_rom_loss(self, *, weight: float = 1.0):
        from .stgat_temporal_pyramid_phase_aware import RangeOfMotionLoss

        return RangeOfMotionLoss(weight=weight)
