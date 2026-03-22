import os
import tempfile
import shutil
import time

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
import numpy as np

from app.schemas.request_response import (
    KeypointSequenceRequest,
    PredictionResponse,
    ModelInfoResponse,
    ExerciseInfo,
)
from app.services.inference_service import InferenceService
from app.services.model_repository import ModelRepository
from app.models.model_factory import ModelFactory
from config import settings

router = APIRouter(prefix="/api/v1", tags=["Movement Analysis"])


class ServiceRegistry:
    """
    Singleton registry that lazily creates and caches one
    ``InferenceService`` per model architecture.

    Using a Singleton class (rather than a module-level dict) makes the
    lifecycle explicit and keeps the pattern consistent with the rest of
    the codebase.  The cache is never cleared at runtime since models are
    stateless once loaded.

    Usage::

        service = ServiceRegistry().get("stgat")
    """

    _instance: "ServiceRegistry | None" = None
    _services: dict[tuple[str, str], InferenceService]
    _repositories: dict[tuple[str, str], ModelRepository]

    def __new__(cls) -> "ServiceRegistry":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._services = {}
            inst._repositories = {}
            cls._instance = inst
        return cls._instance

    def get(self, model_name: str | None = None, dataset: str = "intellirehab") -> InferenceService:
        """Return (or lazily create) the service for *(model_name, dataset)*."""
        name = model_name or settings.ACTIVE_MODEL
        dataset_key = settings._normalize_dataset_key(dataset)
        cache_key = (name, dataset_key)
        if cache_key not in self._services:
            # Use a dedicated ModelRepository for this model architecture
            if cache_key not in self._repositories:
                self._repositories[cache_key] = ModelRepository(model_name=name, dataset=dataset_key)
            repo = self._repositories[cache_key]
            self._services[cache_key] = InferenceService(
                model_name=name,
                dataset=dataset_key,
                model_repository=repo,
            )
        return self._services[cache_key]


def _get_service(model_name: str | None = None, dataset: str = "intellirehab") -> InferenceService:
    """Backward-compatible helper — delegates to ``ServiceRegistry``."""
    return ServiceRegistry().get(model_name, dataset=dataset)


def _validate_exercise(dataset: str, exercise_id: int) -> str:
    dataset_key = settings._normalize_dataset_key(dataset)
    exercises = settings.exercises_for(dataset_key)
    if exercise_id not in exercises:
        raise HTTPException(
            400,
            f"Unknown exercise_id {exercise_id} for dataset '{dataset_key}'. "
            f"Available: {list(exercises.keys())}",
        )
    return dataset_key


def _exercise_payload(model_name: str, dataset: str) -> list[dict]:
    dataset_key = settings._normalize_dataset_key(dataset)
    return [
        {
            "dataset": dataset_key,
            "id": eid,
            "name": ename,
            "has_weights": os.path.exists(
                settings.weights_path_for(model_name, eid, dataset=dataset_key)
            ),
        }
        for eid, ename in settings.exercises_for(dataset_key).items()
    ]


def _safe_unlink(path: str, retries: int = 5, delay_sec: float = 0.05) -> None:
    """Best-effort deletion for temp files, tolerant to transient Windows locks."""
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                os.unlink(path)
            return
        except PermissionError:
            if attempt == retries - 1:
                return
            time.sleep(delay_sec)


@router.post("/predict/video", response_model=PredictionResponse)
async def predict_from_video(
    file: UploadFile = File(...),
    exercise_id: int = Form(..., description="Exercise type (0-8)"),
    dataset: str = Form(default="intellirehab", description="Dataset key: intellirehab | uiprmd"),
    model_name: str | None = Form(default=None, description="Model architecture to use"),
):
    """Upload a video and get movement correctness prediction for a specific exercise."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video.")

    dataset_key = _validate_exercise(dataset, exercise_id)

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        service = _get_service(model_name, dataset=dataset_key)
        result = service.predict_from_video(tmp_path, exercise_id=exercise_id)
        return PredictionResponse(**result)
    finally:
        _safe_unlink(tmp_path)


@router.post("/predict/keypoints", response_model=PredictionResponse)
async def predict_from_keypoints(request: KeypointSequenceRequest):
    """Send pre-extracted keypoints and get prediction for a specific exercise."""
    dataset_key = _validate_exercise(request.dataset, request.exercise_id)

    frames = [f.keypoints for f in request.frames]
    keypoints = np.array(frames, dtype=np.float32)

    service = _get_service(request.model_name, dataset=dataset_key)
    result = service.predict_from_keypoints(
        keypoints, exercise_id=request.exercise_id
    )
    return PredictionResponse(**result)


@router.get("/models", response_model=ModelInfoResponse)
async def list_models():
    """List registered models and available exercises."""
    model_name = settings.ACTIVE_MODEL
    exercises = [
        ExerciseInfo(**entry)
        for entry in (
            _exercise_payload(model_name, "intellirehab")
            + _exercise_payload(model_name, "uiprmd")
        )
    ]
    return ModelInfoResponse(
        available_models=ModelFactory.list_models(),
        active_model=model_name,
        exercises=exercises,
    )


@router.get("/exercises")
async def list_exercises():
    """Return the exercise registry so the frontend can populate dropdowns."""
    model_name = settings.ACTIVE_MODEL
    return _exercise_payload(model_name, "intellirehab") + _exercise_payload(model_name, "uiprmd")


@router.post("/predict/video_annotated")
async def predict_with_annotated_video(
    file: UploadFile = File(...),
    exercise_id: int = Form(..., description="Exercise type (0-8)"),
    dataset: str = Form(default="intellirehab", description="Dataset key: intellirehab | uiprmd"),
    model_name: str | None = Form(default=None, description="Model architecture to use"),
):
    """
    Upload a video and get:
    1. Movement correctness prediction
    2. Annotated video with color-coded skeleton overlay
    
    Returns annotated video file.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video.")

    dataset_key = _validate_exercise(dataset, exercise_id)

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        service = _get_service(model_name, dataset=dataset_key)
        
        # Extract keypoints and compute prediction + deviations
        keypoints = service.pose_extractor.extract_from_video(tmp_path)
        keypoints = service._prepare_keypoints_for_dataset(keypoints)
        processed = service.preprocessor.process(keypoints)
        
        model = service._get_model(exercise_id)
        result = model.predict(processed)
        result["dataset"] = dataset_key
        
        deviations = service.explainability.compute_deviations(
            processed,
            exercise_id,
            dataset=dataset_key,
        )
        if not deviations:
            # Keep annotated-video endpoint usable even when reference files are unavailable.
            deviations = {}
        
        # Create annotated video
        output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = output_tmp.name
        output_tmp.close()
        
        service.explainability.create_annotated_video(
            tmp_path,
            keypoints,
            deviations,
            output_path
        )
        
        # Return the annotated video
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"annotated_{exercise_id}_{result['label']}.mp4",
            headers={
                "X-Prediction-Label": result["label"],
                "X-Prediction-Confidence": str(result["confidence"]),
                "X-Problematic-Joints": ",".join(service.explainability.get_problematic_joints(deviations)),
            }
        )
    finally:
        # Clean up input file (output cleaned by FileResponse background task)
        _safe_unlink(tmp_path)


@router.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}