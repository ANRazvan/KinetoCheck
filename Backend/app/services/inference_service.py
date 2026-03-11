import os
from typing import Dict, Any

import numpy as np

from app.models.base_model import BaseMovementModel
from app.models.model_factory import ModelFactory
from app.preprocessing.skeleton_preprocessor import SkeletonPreprocessor
from app.services.pose_extractor import BasePoseExtractor, create_pose_extractor
from app.services.explainability_service import ExplainabilityService
from config import settings


class InferenceService:
    """
    High-level service that ties together:
      1. Pose extraction  (pluggable backend via BasePoseExtractor)
      2. Preprocessing    (normalize + pad)
      3. Model inference   (one model *per exercise*)

    Each ``exercise_id`` gets its own weights file, e.g.
    ``weights/stgat_exercise_3_best.pt``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        pose_extractor: BasePoseExtractor | None = None,
    ):
        self.model_name = model_name or settings.ACTIVE_MODEL
        self.preprocessor = SkeletonPreprocessor()
        self.explainability = ExplainabilityService()

        # Pose extractor is injected (DI) or created via factory
        self.pose_extractor = pose_extractor or create_pose_extractor()

        # Cache: exercise_id → built model with loaded weights
        self._models: Dict[int, BaseMovementModel] = {}

    # ── public API ───────────────────────────────────────────────────

    def predict_from_video(
        self, video_path: str, exercise_id: int
    ) -> Dict[str, Any]:
        """End-to-end: video → keypoints → preprocess → predict."""
        model = self._get_model(exercise_id)
        keypoints = self.pose_extractor.extract_from_video(video_path)
        processed = self.preprocessor.process(keypoints)
        result = model.predict(processed)
        result["exercise_id"] = exercise_id
        result["exercise_name"] = settings.exercise_name(exercise_id)
        
        # Add deviation analysis
        deviations = self.explainability.compute_deviations(processed, exercise_id)
        if deviations:
            result["joint_deviations"] = deviations
            result["problematic_joints"] = self.explainability.get_problematic_joints(deviations)
        
        result["model_info"] = model.get_model_info()
        return result

    def predict_from_keypoints(
        self, keypoints, exercise_id: int
    ) -> Dict[str, Any]:
        """If keypoints already extracted (e.g. from frontend / IntelliRehab)."""
        model = self._get_model(exercise_id)
        keypoints = np.array(keypoints, dtype=np.float32)

        # Handle flat IntelliRehab format: (num_frames, 75) → (num_frames, 25, 3)
        if (
            keypoints.ndim == 2
            and keypoints.shape[1]
            == settings.NUM_KEYPOINTS * settings.KEYPOINT_DIM
        ):
            keypoints = keypoints.reshape(
                keypoints.shape[0],
                settings.NUM_KEYPOINTS,
                settings.KEYPOINT_DIM,
            )
        # Add deviation analysis
        deviations = self.explainability.compute_deviations(processed, exercise_id)

        if deviations:
            result["joint_deviations"] = deviations
            result["problematic_joints"] = self.explainability.get_problematic_joints(deviations)
        

        processed = self.preprocessor.process(keypoints)
        result = model.predict(processed)
        result["exercise_id"] = exercise_id
        result["exercise_name"] = settings.exercise_name(exercise_id)
        result["model_info"] = model.get_model_info()
        return result

    # ── internals ────────────────────────────────────────────────────

    def _get_model(self, exercise_id: int) -> BaseMovementModel:
        """Lazy-load and cache a model for the given exercise."""
        if exercise_id in self._models:
            return self._models[exercise_id]

        model = ModelFactory.create(self.model_name)
        model.build()

        weights_path = settings.weights_path_for(self.model_name, exercise_id)
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            # Fall back to the generic (non-exercise-specific) weights
            fallback = os.path.join(
                settings.WEIGHTS_DIR, f"{self.model_name}_best.pt"
            )
            if os.path.exists(fallback):
                model.load_weights(fallback)

        self._models[exercise_id] = model
        return model