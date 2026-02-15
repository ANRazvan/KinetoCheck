from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class KeypointFrame(BaseModel):
    """Single frame: list of [x, y] for each keypoint."""
    keypoints: List[List[float]]


class KeypointSequenceRequest(BaseModel):
    """Full sequence of frames with keypoints."""
    frames: List[KeypointFrame]
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    details: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None


class ModelInfoResponse(BaseModel):
    available_models: List[str]
    active_model: str