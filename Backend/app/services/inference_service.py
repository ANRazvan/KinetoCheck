from typing import Dict, Any

import numpy as np

from app.models.base_model import BaseMovementModel
from app.preprocessing.skeleton_preprocessor import SkeletonPreprocessor
from app.preprocessing.uiprmd_preprocessor import UIPRMDPreprocessor
from app.services.pose_extractor import BasePoseExtractor, create_pose_extractor
from app.services.explainability_service import ExplainabilityService
from app.services.model_repository import ModelRepository
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
        dataset: str = "intellirehab",
        pose_extractor: BasePoseExtractor | None = None,
        model_repository: ModelRepository | None = None,
    ):
        self.model_name = model_name or settings.ACTIVE_MODEL
        self.dataset = settings._normalize_dataset_key(dataset)
        if self.dataset == "uiprmd":
            self.preprocessor = UIPRMDPreprocessor()
        else:
            self.preprocessor = SkeletonPreprocessor()
        self.explainability = ExplainabilityService()

        # Pose extractor is injected (DI) or created via factory
        self.pose_extractor = pose_extractor or create_pose_extractor()
        self.model_repository = model_repository or ModelRepository(
            self.model_name,
            dataset=self.dataset,
        )

    # ── public API ───────────────────────────────────────────────────

    def predict_from_video(
        self, video_path: str, exercise_id: int
    ) -> Dict[str, Any]:
        """End-to-end: video → keypoints → preprocess → predict."""
        model = self._get_model(exercise_id)
        keypoints = self.pose_extractor.extract_from_video(video_path)
        keypoints = self._prepare_keypoints_for_dataset(keypoints)
        processed = self.preprocessor.process(keypoints)
        result = model.predict(processed)
        return self._enrich_result(result, processed, exercise_id, model)

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

        keypoints = self._prepare_keypoints_for_dataset(keypoints)
        processed = self.preprocessor.process(keypoints)
        result = model.predict(processed)
        return self._enrich_result(result, processed, exercise_id, model)

    def _enrich_result(
        self,
        result: Dict[str, Any],
        processed: np.ndarray,
        exercise_id: int,
        model: BaseMovementModel,
    ) -> Dict[str, Any]:
        """Attach shared metadata and explainability details to model output."""
        result["exercise_id"] = exercise_id
        result["dataset"] = self.dataset
        result["exercise_name"] = settings.exercise_name_for(self.dataset, exercise_id)

        deviations = self.explainability.compute_deviations(
            processed,
            exercise_id,
            dataset=self.dataset,
        )
        if deviations:
            result["joint_deviations"] = deviations
            result["problematic_joints"] = self.explainability.get_problematic_joints(
                deviations
            )

        result["model_info"] = model.get_model_info()
        return result

    def _prepare_keypoints_for_dataset(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert extracted keypoints into the expected dataset-specific shape."""
        if self.dataset != "uiprmd":
            return keypoints

        arr = np.array(keypoints, dtype=np.float32)

        # UI-PRMD inference expects 17 joints x 3D. Pose extractors usually provide 17 x 2D.
        if arr.ndim == 3 and arr.shape[1] == settings.UIPRMD_NUM_KEYPOINTS and arr.shape[2] == 2:
            return np.pad(arr, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)

        if arr.ndim == 2 and arr.shape[1] == settings.UIPRMD_NUM_KEYPOINTS * 2:
            arr = arr.reshape(arr.shape[0], settings.UIPRMD_NUM_KEYPOINTS, 2)
            return np.pad(arr, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)

        return arr

    # ── internals ────────────────────────────────────────────────────

    def _get_model(self, exercise_id: int) -> BaseMovementModel:
        """Lazy-load and cache a model for the given exercise."""
        return self.model_repository.get(exercise_id)