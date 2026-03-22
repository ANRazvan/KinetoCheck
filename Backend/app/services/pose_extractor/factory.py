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
    """
    Registry that maps backend names → BasePoseExtractor subclasses.

    Mirrors the structure of ``ModelFactory`` so the two factories stay
    consistent and both benefit from the same Abstract Factory approach.

    Usage::

        extractor = PoseExtractorFactory.create("yolo")
    """

    _registry: Dict[str, Type[BasePoseExtractor]] = {}

    @classmethod
    def register(cls, name: str, extractor_class: Type[BasePoseExtractor]) -> None:
        """Register a pose-extractor backend by name."""
        cls._registry[name] = extractor_class

    @classmethod
    def create(cls, name: str | None = None) -> BasePoseExtractor:
        """
        Instantiate and return a pose extractor by *name*.

        Args:
            name: Backend key (e.g. ``"yolo"``).  Defaults to
                  ``settings.POSE_EXTRACTOR`` when omitted.
        """
        key = (name or settings.POSE_EXTRACTOR).lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown pose extractor '{key}'. Available: {available}"
            )
        return cls._registry[key]()

    @classmethod
    def list_extractors(cls) -> list[str]:
        return list(cls._registry.keys())


# ── Register built-in backends ────────────────────────────────────────
# Imported lazily inside the lambda so that missing deps (ultralytics, cv2)
# don't break imports of this module on systems where they aren't installed.
from app.services.pose_extractor.yolo_extractor import YoloPoseExtractor  # noqa: E402
PoseExtractorFactory.register("yolo", YoloPoseExtractor)

# Future: just add one line
# from app.services.pose_extractor.mediapipe_extractor import MediaPipeExtractor
# PoseExtractorFactory.register("mediapipe", MediaPipeExtractor)


# ── Backward-compatible helper ────────────────────────────────────────

def create_pose_extractor(backend: str | None = None) -> BasePoseExtractor:
    """
    Instantiate a pose extractor by name.

    Delegates to ``PoseExtractorFactory.create()``; kept for backward
    compatibility with code that imports this function directly.

    Args:
        backend: ``"yolo"`` (default from ``settings.POSE_EXTRACTOR``).
    """
    return PoseExtractorFactory.create(backend)
