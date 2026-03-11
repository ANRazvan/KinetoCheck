"""
Factory that returns the correct pose-extraction backend
based on ``settings.POSE_EXTRACTOR``.

To add a new backend:
  1. Implement ``BasePoseExtractor`` in a new module
  2. Add an ``elif`` branch here (or use a registry dict)
"""

from app.services.pose_extractor.base import BasePoseExtractor
from config import settings


def create_pose_extractor(backend: str | None = None) -> BasePoseExtractor:
    """
    Instantiate a pose extractor by name.

    Args:
        backend: ``"yolo"`` (default from ``settings.POSE_EXTRACTOR``).
                 Future options: ``"mediapipe"``, ``"openpose"``, etc.
    """
    name = (backend or settings.POSE_EXTRACTOR).lower()

    if name == "yolo":
        from app.services.pose_extractor.yolo_extractor import YoloPoseExtractor
        return YoloPoseExtractor()
    else:
        raise ValueError(
            f"Unknown pose extractor '{name}'. "
            f"Available: yolo"
        )
