from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class KeypointFrame(BaseModel):
    """Single frame: list of [x, y] for each keypoint."""
    keypoints: List[List[float]]


class KeypointSequenceRequest(BaseModel):
    """Full sequence of frames with keypoints."""
    frames: List[KeypointFrame]
    exercise_id: int
    dataset: str = "intellirehab"
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    exercise_id: Optional[int] = None
    exercise_name: Optional[str] = None
    dataset: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None
    problematic_joints: Optional[List[str]] = None
    joint_deviations: Optional[Dict[str, float]] = None


class ExerciseInfo(BaseModel):
    dataset: str
    id: int
    name: str
    has_weights: bool = False


class ModelInfoResponse(BaseModel):
    available_models: List[str]
    active_model: str
    exercises: List[ExerciseInfo] = []