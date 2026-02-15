import numpy as np
import cv2
from ultralytics import YOLO
from config import settings


class PoseExtractionService:
    """Extract skeleton keypoints from video/images using YOLOv8-pose."""

    def __init__(self):
        self.model = YOLO(settings.YOLO_WEIGHTS)

    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract keypoints from every frame of a video.
        
        Returns:
            np.ndarray of shape (num_frames, num_keypoints, 2)
        """
        cap = cv2.VideoCapture(video_path)
        all_keypoints = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            kps = self._extract_keypoints_from_result(results)
            all_keypoints.append(kps)

        cap.release()
        return np.array(all_keypoints)

    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract keypoints from a single frame. Returns (num_keypoints, 2)."""
        results = self.model(frame, verbose=False)
        return self._extract_keypoints_from_result(results)

    def _extract_keypoints_from_result(self, results) -> np.ndarray:
        """Parse YOLO results and return (num_keypoints, 2)."""
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.xy.cpu().numpy()
            if len(kps) > 0:
                return kps[0][:settings.NUM_KEYPOINTS, :2]  # first person
        return np.zeros((settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM))