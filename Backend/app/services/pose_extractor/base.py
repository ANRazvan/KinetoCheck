"""
Abstract interface for pose extraction backends.

To add a new backend (e.g. MediaPipe):
  1. Create ``mediapipe_extractor.py`` implementing ``BasePoseExtractor``
  2. Register it in ``factory.py``
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePoseExtractor(ABC):
    """
    Contract every pose-extraction backend must satisfy.
    All implementations return keypoints in (num_keypoints, C) per frame,
    where C is typically 2 (x, y) or 3 (x, y, z).
    """

    @abstractmethod
    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract keypoints from every frame of a video.

        Returns:
            np.ndarray of shape (num_frames, num_keypoints, C)
        """
        ...

    @abstractmethod
    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract keypoints from a single BGR frame.

        Returns:
            np.ndarray of shape (num_keypoints, C)
        """
        ...
