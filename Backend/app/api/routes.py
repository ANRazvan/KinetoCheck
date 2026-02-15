import os
import tempfile
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import numpy as np

from app.schemas.request_response import (
    KeypointSequenceRequest,
    PredictionResponse,
    ModelInfoResponse,
)
from app.services.inference_service import InferenceService
from app.models.model_factory import ModelFactory
from config import settings

router = APIRouter(prefix="/api/v1", tags=["Movement Analysis"])

# Lazy-loaded service cache (one per model)
_services: dict = {}


def _get_service(model_name: str = None) -> InferenceService:
    name = model_name or settings.ACTIVE_MODEL
    if name not in _services:
        _services[name] = InferenceService(model_name=name)
    return _services[name]


@router.post("/predict/video", response_model=PredictionResponse)
async def predict_from_video(
    file: UploadFile = File(...),
    model_name: str = Query(default=None, description="Model to use"),
):
    """Upload a video and get movement correctness prediction."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video.")

    # Save to temp file
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        service = _get_service(model_name)
        result = service.predict_from_video(tmp_path)
        return PredictionResponse(**result)
    finally:
        os.unlink(tmp_path)


@router.post("/predict/keypoints", response_model=PredictionResponse)
async def predict_from_keypoints(request: KeypointSequenceRequest):
    """Send pre-extracted keypoints and get prediction."""
    frames = [f.keypoints for f in request.frames]
    keypoints = np.array(frames, dtype=np.float32)

    service = _get_service(request.model_name)
    result = service.predict_from_keypoints(keypoints)
    return PredictionResponse(**result)


@router.get("/models", response_model=ModelInfoResponse)
async def list_models():
    """List all registered models."""
    return ModelInfoResponse(
        available_models=ModelFactory.list_models(),
        active_model=settings.ACTIVE_MODEL,
    )


@router.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}