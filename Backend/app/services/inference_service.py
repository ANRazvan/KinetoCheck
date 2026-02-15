import os
from typing import Dict, Any

from app.models.base_model import BaseMovementModel
from app.models.model_factory import ModelFactory
from app.preprocessing.skeleton_preprocessor import SkeletonPreprocessor
from app.services.pose_extraction_service import PoseExtractionService
from config import settings


class InferenceService:
    """
    High-level service that ties together:
      1. Pose extraction (YOLO)
      2. Preprocessing (normalize + pad)
      3. Model inference (Strategy pattern)
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or settings.ACTIVE_MODEL

        # Create model via factory
        self.model: BaseMovementModel = ModelFactory.create(model_name)
        self.model.build()

        # Load weights if available
        weights_path = os.path.join(settings.WEIGHTS_DIR, f"{model_name}_best.pt")
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)

        self.preprocessor = SkeletonPreprocessor()
        self.pose_extractor = PoseExtractionService()

    def predict_from_video(self, video_path: str) -> Dict[str, Any]:
        """End-to-end: video → keypoints → preprocess → predict."""
        keypoints = self.pose_extractor.extract_from_video(video_path)
        processed = self.preprocessor.process(keypoints)
        result = self.model.predict(processed)
        result["model_info"] = self.model.get_model_info()
        return result

    def predict_from_keypoints(self, keypoints) -> Dict[str, Any]:
        """If keypoints already extracted (e.g., from frontend or IntelliRehab data)."""
        import numpy as np
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # Handle flat IntelliRehab format: (num_frames, 75) → reshape to (num_frames, 25, 3)
        if keypoints.ndim == 2 and keypoints.shape[1] == settings.NUM_KEYPOINTS * settings.KEYPOINT_DIM:
            keypoints = keypoints.reshape(
                keypoints.shape[0], settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM
            )
        
        processed = self.preprocessor.process(keypoints)
        result = self.model.predict(processed)
        result["model_info"] = self.model.get_model_info()
        return result