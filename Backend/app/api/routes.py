import os
import tempfile
import shutil

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
from app.models.model_factory import ModelFactory
from config import settings

router = APIRouter(prefix="/api/v1", tags=["Movement Analysis"])

# Lazy-loaded service cache (one per model architecture)
_services: dict[str, InferenceService] = {}


def _get_service(model_name: str | None = None) -> InferenceService:
    name = model_name or settings.ACTIVE_MODEL
    if name not in _services:
        _services[name] = InferenceService(model_name=name)
    return _services[name]


@router.post("/predict/video", response_model=PredictionResponse)
async def predict_from_video(
    file: UploadFile = File(...),
    exercise_id: int = Form(..., description="Exercise type (0-8)"),
    model_name: str | None = Form(default=None, description="Model architecture to use"),
):
    """Upload a video and get movement correctness prediction for a specific exercise."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video.")

    if exercise_id not in settings.EXERCISES:
        raise HTTPException(
            400,
            f"Unknown exercise_id {exercise_id}. "
            f"Available: {list(settings.EXERCISES.keys())}",
        )

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        service = _get_service(model_name)
        result = service.predict_from_video(tmp_path, exercise_id=exercise_id)
        return PredictionResponse(**result)
    finally:
        os.unlink(tmp_path)


@router.post("/predict/keypoints", response_model=PredictionResponse)
async def predict_from_keypoints(request: KeypointSequenceRequest):
    """Send pre-extracted keypoints and get prediction for a specific exercise."""
    if request.exercise_id not in settings.EXERCISES:
        raise HTTPException(
            400,
            f"Unknown exercise_id {request.exercise_id}. "
            f"Available: {list(settings.EXERCISES.keys())}",
        )

    frames = [f.keypoints for f in request.frames]
    keypoints = np.array(frames, dtype=np.float32)

    service = _get_service(request.model_name)
    result = service.predict_from_keypoints(
        keypoints, exercise_id=request.exercise_id
    )
    return PredictionResponse(**result)


@router.get("/models", response_model=ModelInfoResponse)
async def list_models():
    """List registered models and available exercises."""
    exercises = []
    model_name = settings.ACTIVE_MODEL
    for eid, ename in settings.EXERCISES.items():
        wp = settings.weights_path_for(model_name, eid)
        exercises.append(
            ExerciseInfo(id=eid, name=ename, has_weights=os.path.exists(wp))
        )
    return ModelInfoResponse(
        available_models=ModelFactory.list_models(),
        active_model=model_name,
        exercises=exercises,
    )


@router.get("/exercises")
async def list_exercises():
    """Return the exercise registry so the frontend can populate dropdowns."""
    model_name = settings.ACTIVE_MODEL
    return [
        {
            "id": eid,
            "name": ename,
            "has_weights": os.path.exists(
                settings.weights_path_for(model_name, eid)
            ),
        }
        for eid, ename in settings.EXERCISES.items()
    ]


@router.post("/predict/video_annotated")
async def predict_with_annotated_video(
    file: UploadFile = File(...),
    exercise_id: int = Form(..., description="Exercise type (0-8)"),
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

    if exercise_id not in settings.EXERCISES:
        raise HTTPException(
            400,
            f"Unknown exercise_id {exercise_id}. "
            f"Available: {list(settings.EXERCISES.keys())}",
        )

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        service = _get_service(model_name)
        
        # Extract keypoints and compute prediction + deviations
        keypoints = service.pose_extractor.extract_from_video(tmp_path)
        processed = service.preprocessor.process(keypoints)
        
        model = service._get_model(exercise_id)
        result = model.predict(processed)
        
        deviations = service.explainability.compute_deviations(processed, exercise_id)
        
        if not deviations:
            raise HTTPException(
                404, 
                f"No reference data found for exercise {exercise_id}. "
                f"Run: python -m tools.compute_reference_keypoints"
            )
        
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
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}