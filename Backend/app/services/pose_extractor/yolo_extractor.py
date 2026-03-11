"""
YOLO-based pose extraction (ultralytics YOLOv8-pose).
"""

import functools

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.services.pose_extractor.base import BasePoseExtractor
from config import settings


class YoloPoseExtractor(BasePoseExtractor):
    """Extract COCO-17 skeleton keypoints using YOLOv8-pose."""

    def __init__(self, weights_path: str | None = None):
        weights = weights_path or settings.YOLO_WEIGHTS

        # PyTorch 2.6+ defaults weights_only=True in torch.load, but
        # ultralytics 8.x model files require weights_only=False.
        _original_load = torch.load
        torch.load = functools.partial(_original_load, weights_only=False)
        try:
            self.model = YOLO(weights)
        finally:
            torch.load = _original_load

    # ── public API ───────────────────────────────────────────────────

    def extract_from_video(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        all_keypoints: list[np.ndarray] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_keypoints.append(self._extract(frame))

        cap.release()
        return np.array(all_keypoints)  # (num_frames, 17, 2)

    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        return self._extract(frame)

    # ── internals ────────────────────────────────────────────────────

    def _extract(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLO on a single frame, return (COCO_NUM_KEYPOINTS, 2)."""
        results = self.model(frame, verbose=False)
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.xy.cpu().numpy()
            if len(kps) > 0:
                return kps[0][: settings.COCO_NUM_KEYPOINTS, : settings.COCO_KEYPOINT_DIM]
        return np.zeros((settings.COCO_NUM_KEYPOINTS, settings.COCO_KEYPOINT_DIM))
