"""
Registry-based factory for pose-extraction backends.

To add a new backend (e.g. MediaPipe):
  1. Implement ``BasePoseExtractor`` in a new module
  2. Register it with ``PoseExtractorFactory.register("mediapipe", MediaPipeExtractor)``

The module-level ``create_pose_extractor()`` helper is kept for backward-compat.
"""

from typing import Dict, Type

from app.services.pose_extractor.base import BasePoseExtractor
from config import settings


class PoseExtractorFactory:
    """Registry that maps backend names → BasePoseExtractor subclasses."""

    _registry: Dict[str, Type[BasePoseExtractor]] = {}

    @classmethod
    def _ensure_defaults_registered(cls) -> None:
        """Register built-in extractors lazily to avoid hard deps at import time."""
        if "yolo" not in cls._registry:
            from app.services.pose_extractor.yolo_extractor import YoloPoseExtractor

            cls._registry["yolo"] = YoloPoseExtractor
        if "mediapipe" not in cls._registry:
            try:
                from app.services.pose_extractor.mediapipe_extractor import MediaPipePoseExtractor

                cls._registry["mediapipe"] = MediaPipePoseExtractor
            except ImportError:
                # Keep mediapipe optional unless explicitly selected.
                pass

    @classmethod
    def register(cls, name: str, extractor_cls: Type[BasePoseExtractor]) -> None:
        cls._registry[name] = extractor_cls

    @classmethod
    def create(cls, name: str | None = None) -> BasePoseExtractor:
        cls._ensure_defaults_registered()
        key = (name or settings.POSE_EXTRACTOR).lower()
        extractor_cls = cls._registry.get(key)
        if extractor_cls is None:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown pose extractor '{key}'. Available: {available}")
        return extractor_cls()

    @classmethod
    def list_extractors(cls) -> list[str]:
        cls._ensure_defaults_registered()
        return list(cls._registry.keys())


def create_pose_extractor(backend: str | None = None) -> BasePoseExtractor:
    """Backward-compatible helper to instantiate a pose extractor."""
    return PoseExtractorFactory.create(backend)
