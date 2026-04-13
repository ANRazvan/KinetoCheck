from typing import Dict, Any

import numpy as np

from app.models.base_model import BaseMovementModel
from app.preprocessing.intellirehab_2d_preprocessor import IntelliRehab2DPreprocessor
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
        if self.dataset in {"uiprmd", "uiprmd_2d"}:
            self.preprocessor = UIPRMDPreprocessor(
                keypoint_dim=settings.uiprmd_keypoint_dim_for_dataset(self.dataset)
            )
        elif self.dataset == "intellirehab_2d":
            self.preprocessor = IntelliRehab2DPreprocessor()
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

    def score_exercises_for_keypoints(
        self,
        keypoints,
        exercise_ids: list[int] | None = None,
    ) -> list[Dict[str, Any]]:
        """
        Score one keypoint sequence against multiple exercise-specific models.

        This is used by timeline inference for videos containing multiple exercises.
        Returns sorted candidates (highest score first).
        """
        keypoints = np.array(keypoints, dtype=np.float32)

        # Handle flat IntelliRehab format: (num_frames, 75) -> (num_frames, 25, 3)
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

        ids = exercise_ids or sorted(settings.exercises_for(self.dataset).keys())
        candidates: list[Dict[str, Any]] = []

        for exercise_id in ids:
            model = self._get_model(exercise_id)
            result = model.predict(processed)
            probs = (result.get("details") or {}).get("raw_probs") or []

            max_prob = float(max(probs)) if probs else float(result.get("confidence", 0.0))
            correct_prob = float(probs[0]) if len(probs) > 0 else 0.0
            incorrect_prob = float(probs[1]) if len(probs) > 1 else 0.0

            candidates.append(
                {
                    "exercise_id": exercise_id,
                    "exercise_name": settings.exercise_name_for(self.dataset, exercise_id),
                    "score": max_prob,
                    "predicted_label": result.get("label"),
                    "correct_prob": correct_prob,
                    "incorrect_prob": incorrect_prob,
                }
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

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
        if self.dataset == "intellirehab_2d":
            return self._prepare_intellirehab_2d_keypoints(keypoints)

        if self.dataset not in {"uiprmd", "uiprmd_2d"}:
            return keypoints

        arr = np.array(keypoints, dtype=np.float32)

        if self.dataset == "uiprmd_2d":
            # Keep UI-PRMD 2D data as (frames, 17, 2) or (frames, 34).
            if arr.ndim == 3 and arr.shape[1] == settings.UIPRMD_NUM_KEYPOINTS and arr.shape[2] == 3:
                return arr[:, :, :2]
            if arr.ndim == 2 and arr.shape[1] == settings.UIPRMD_NUM_KEYPOINTS * 3:
                return arr.reshape(arr.shape[0], settings.UIPRMD_NUM_KEYPOINTS, 3)[:, :, :2]
            return arr

        # UI-PRMD inference expects 17 joints x 3D. Extractors may provide 17x2 or 17x3.
        if arr.ndim == 3 and arr.shape[1] == settings.UIPRMD_NUM_KEYPOINTS and arr.shape[2] == 2:
            return np.pad(arr, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)

        if arr.ndim == 2 and arr.shape[1] == settings.UIPRMD_NUM_KEYPOINTS * 2:
            arr = arr.reshape(arr.shape[0], settings.UIPRMD_NUM_KEYPOINTS, 2)
            return np.pad(arr, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)

        return arr

    def _prepare_intellirehab_2d_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Map incoming keypoints to Kinect-25 2D layout used by IntelliRehab 2D."""
        arr = np.asarray(keypoints, dtype=np.float32)

        if arr.ndim == 2 and arr.shape[1] == settings.NUM_KEYPOINTS * settings.INTELLIREHAB_KEYPOINT_DIM_2D:
            return arr.reshape(arr.shape[0], settings.NUM_KEYPOINTS, settings.INTELLIREHAB_KEYPOINT_DIM_2D)

        if arr.ndim == 2 and arr.shape[1] == settings.NUM_KEYPOINTS * settings.KEYPOINT_DIM:
            return arr.reshape(arr.shape[0], settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM)[:, :, :2]

        if arr.ndim == 3 and arr.shape[1] == settings.NUM_KEYPOINTS and arr.shape[2] == settings.KEYPOINT_DIM:
            return arr[:, :, :2]

        if arr.ndim == 3 and arr.shape[1] == settings.NUM_KEYPOINTS and arr.shape[2] == settings.INTELLIREHAB_KEYPOINT_DIM_2D:
            return arr

        if arr.ndim == 2 and arr.shape[1] == settings.COCO_NUM_KEYPOINTS * settings.COCO_KEYPOINT_DIM:
            arr = arr.reshape(arr.shape[0], settings.COCO_NUM_KEYPOINTS, settings.COCO_KEYPOINT_DIM)

        if arr.ndim == 3 and arr.shape[1] == settings.COCO_NUM_KEYPOINTS and arr.shape[2] == settings.COCO_KEYPOINT_DIM:
            return self._map_coco_to_kinect_2d(arr)

        if arr.ndim == 3 and arr.shape[1] == settings.COCO_NUM_KEYPOINTS and arr.shape[2] == 3:
            # MediaPipe can provide COCO-mapped 3D; IntelliRehab 2D consumes XY.
            return self._map_coco_to_kinect_2d(arr[:, :, :2])

        return arr

    def _map_coco_to_kinect_2d(self, coco_xy: np.ndarray) -> np.ndarray:
        """Convert COCO-17 XY frames to Kinect-25 XY by anatomical mapping."""
        frames = coco_xy.shape[0]
        out = np.zeros((frames, settings.NUM_KEYPOINTS, settings.INTELLIREHAB_KEYPOINT_DIM_2D), dtype=np.float32)

        # Direct semantic mappings where joints are available in COCO.
        mapping = {
            3: 0,    # Head <- Nose
            4: 5,    # ShoulderLeft <- LeftShoulder
            5: 7,    # ElbowLeft <- LeftElbow
            6: 9,    # WristLeft <- LeftWrist
            8: 6,    # ShoulderRight <- RightShoulder
            9: 8,    # ElbowRight <- RightElbow
            10: 10,  # WristRight <- RightWrist
            12: 11,  # HipLeft <- LeftHip
            13: 13,  # KneeLeft <- LeftKnee
            14: 15,  # AnkleLeft <- LeftAnkle
            16: 12,  # HipRight <- RightHip
            17: 14,  # KneeRight <- RightKnee
            18: 16,  # AnkleRight <- RightAnkle
        }

        for kinect_idx, coco_idx in mapping.items():
            out[:, kinect_idx, :] = coco_xy[:, coco_idx, :]

        left_hip = coco_xy[:, 11, :]
        right_hip = coco_xy[:, 12, :]
        left_shoulder = coco_xy[:, 5, :]
        right_shoulder = coco_xy[:, 6, :]

        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        out[:, 0, :] = hip_center            # SpineBase
        out[:, 20, :] = shoulder_center      # SpineShoulder
        out[:, 2, :] = shoulder_center       # Neck (approximation)
        out[:, 1, :] = (hip_center + shoulder_center) / 2.0  # SpineMid

        out[:, 7, :] = out[:, 6, :]   # HandLeft <- WristLeft
        out[:, 11, :] = out[:, 10, :] # HandRight <- WristRight
        out[:, 15, :] = out[:, 14, :] # FootLeft <- AnkleLeft
        out[:, 19, :] = out[:, 18, :] # FootRight <- AnkleRight
        out[:, 21, :] = out[:, 7, :]  # HandTipLeft
        out[:, 22, :] = out[:, 7, :]  # ThumbLeft
        out[:, 23, :] = out[:, 11, :] # HandTipRight
        out[:, 24, :] = out[:, 11, :] # ThumbRight

        return out

    # ── internals ────────────────────────────────────────────────────

    def _get_model(self, exercise_id: int) -> BaseMovementModel:
        """Lazy-load and cache a model for the given exercise."""
        return self.model_repository.get(exercise_id)