import os
from typing import Dict

from app.models.base_model import BaseMovementModel
from app.models.model_factory import ModelFactory
from config import settings


class ModelRepository:
    """
    Repository for model loading and caching.

    Responsibilities:
    - Build model instances for a chosen architecture.
    - Resolve candidate weight paths with backward-compatible fallbacks.
    - Cache one loaded model per exercise for fast repeated inference.
    """

    def __init__(self, model_name: str, dataset: str = "intellirehab"):
        self.model_name = model_name
        self.dataset = settings._normalize_dataset_key(dataset)
        self._cache: Dict[int, BaseMovementModel] = {}

    def get(self, exercise_id: int) -> BaseMovementModel:
        """Return a loaded model for *exercise_id*, using in-memory cache."""
        if exercise_id in self._cache:
            return self._cache[exercise_id]

        model = ModelFactory.create(self.model_name)
        model.build(**self._build_kwargs())

        for path in self._candidate_weight_paths(exercise_id):
            if os.path.exists(path):
                model.load_weights(path)
                break

        self._cache[exercise_id] = model
        return model

    def clear(self) -> None:
        """Drop all cached model instances."""
        self._cache.clear()

    def _build_kwargs(self) -> dict:
        """Return dataset-specific model input shape kwargs."""
        if self.dataset in {"uiprmd", "uiprmd_2d"}:
            return {
                "num_keypoints": settings.UIPRMD_NUM_KEYPOINTS,
                "keypoint_dim": settings.uiprmd_keypoint_dim_for_dataset(self.dataset),
            }
        if self.dataset == "intellirehab_2d":
            return {
                "num_keypoints": settings.NUM_KEYPOINTS,
                "keypoint_dim": settings.INTELLIREHAB_KEYPOINT_DIM_2D,
            }
        return {
            "num_keypoints": settings.NUM_KEYPOINTS,
            "keypoint_dim": settings.KEYPOINT_DIM,
        }

    def _candidate_weight_paths(self, exercise_id: int) -> list[str]:
        """Resolve weights in priority order while preserving legacy locations."""
        paths = [
            # 1. Dataset-specific exercise model (e.g., weights/uiprmd/stgat_exercise_1_best.pt)
            settings.weights_path_for(self.model_name, exercise_id, dataset=self.dataset),
            # 2. Dataset-specific general model (e.g., weights/uiprmd/stgat_best.pt)
            os.path.join(settings.weights_dir_for(self.dataset), f"{self.model_name}_best.pt"),
        ]

        # Only check legacy/root paths if we are compatible with the default IntelliRehab structure (25 keypoints)
        if self.dataset in {"intellirehab", "intellirehab_2d"}:
            paths.extend([
                # 3. Legacy exercise path (defaults to intellirehab)
                settings.weights_path_for(self.model_name, exercise_id),
                # 4. Legacy general model (defaults to weights/stgat_best.pt)
                os.path.join(settings.WEIGHTS_DIR, f"{self.model_name}_best.pt"),
            ])

        unique_paths: list[str] = []
        for path in paths:
            if path not in unique_paths:
                unique_paths.append(path)
        return unique_paths
