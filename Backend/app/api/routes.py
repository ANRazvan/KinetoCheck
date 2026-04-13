import os
import tempfile
import shutil
import time
from collections import Counter
from typing import Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
import numpy as np
import cv2

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


def _probe_video(path: str) -> tuple[str, int]:
    """Return (fourcc, frame_count) for a video file, best-effort."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return "", 0
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fourcc, frame_count


def _transcode_reference_for_browser(reference_video: str) -> tuple[str, str] | None:
    """
    Transcode legacy reference previews to browser-friendly codecs and cache them.

    Returns:
        Tuple[path, media_type] if a converted file is available, else None.
    """
    src_mtime = os.path.getmtime(reference_video)
    root, _ = os.path.splitext(reference_video)

    candidates = [
        ("video/mp4", ".browser.mp4", "avc1"),
        ("video/webm", ".browser.webm", "VP90"),
        ("video/webm", ".browser.webm", "VP80"),
    ]

    # Reuse up-to-date browser preview if already generated.
    for media_type, suffix, _ in candidates:
        out_path = root + suffix
        if os.path.exists(out_path) and os.path.getmtime(out_path) >= src_mtime:
            codec, frames = _probe_video(out_path)
            if frames > 0:
                return out_path, media_type

    cap = cv2.VideoCapture(reference_video)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        return None

    # Load frames once, then try multiple encoders/containers.
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    for media_type, suffix, fourcc_tag in candidates:
        out_path = root + suffix
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(tmp_fd)

        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*fourcc_tag),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            _safe_unlink(tmp_path)
            continue

        for frame in frames:
            writer.write(frame)
        writer.release()

        codec, out_frames = _probe_video(tmp_path)
        if out_frames <= 0:
            _safe_unlink(tmp_path)
            continue

        if media_type == "video/mp4" and codec.upper() == "FMP4":
            _safe_unlink(tmp_path)
            continue

        shutil.move(tmp_path, out_path)
        return out_path, media_type

    return None


def _transcode_h264_mp4(source_video: str) -> str | None:
    """Transcode a video to H.264 MP4 and return output path on success."""
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        return None

    root, _ = os.path.splitext(source_video)
    out_path = root + ".h264.mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        _safe_unlink(tmp_path)
        return None

    wrote_any = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        wrote_any = True

    cap.release()
    writer.release()

    if not wrote_any:
        _safe_unlink(tmp_path)
        return None

    codec, frame_count = _probe_video(tmp_path)
    if frame_count <= 0 or codec.upper() == "FMP4":
        _safe_unlink(tmp_path)
        return None

    shutil.move(tmp_path, out_path)
    return out_path


def _extract_video_segment(source_video: str, start_frame: int, end_frame: int) -> str | None:
    """Extract a frame-range clip from source video and return temp output path."""
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0 or total_frames <= 0:
        cap.release()
        return None

    start = max(0, min(start_frame, total_frames - 1))
    end = max(start + 1, min(end_frame, total_frames))

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    try:
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
        if not writer.isOpened():
            writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            _safe_unlink(tmp_path)
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        wrote_any = False
        for _ in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            wrote_any = True

        cap.release()
        writer.release()

        if not wrote_any:
            _safe_unlink(tmp_path)
            return None

        return tmp_path
    except Exception:
        cap.release()
        _safe_unlink(tmp_path)
        return None


def _window_starts(total_frames: int, window_size: int, stride: int) -> list[int]:
    if total_frames <= window_size:
        return [0]

    starts = list(range(0, total_frames - window_size + 1, stride))
    last_start = total_frames - window_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _majority_smooth(labels: list[int], window: int) -> list[int]:
    if window <= 1 or len(labels) <= 2:
        return labels

    radius = window // 2
    out: list[int] = []
    for idx, current in enumerate(labels):
        start = max(0, idx - radius)
        end = min(len(labels), idx + radius + 1)
        votes = labels[start:end]
        counts = Counter(votes)
        best_count = max(counts.values())
        tied = [k for k, v in counts.items() if v == best_count]
        out.append(current if current in tied else tied[0])
    return out


def _labels_to_segments(labels: list[int], starts: list[int], window_size: int) -> list[dict[str, int]]:
    if not labels:
        return []

    segments: list[dict[str, int]] = []
    seg_label = labels[0]
    seg_start = starts[0]
    seg_end = starts[0] + window_size

    for i in range(1, len(labels)):
        current_label = labels[i]
        current_start = starts[i]
        current_end = current_start + window_size
        if current_label == seg_label:
            seg_end = max(seg_end, current_end)
            continue

        segments.append({"exercise_id": seg_label, "start_frame": seg_start, "end_frame": seg_end})
        seg_label = current_label
        seg_start = current_start
        seg_end = current_end

    segments.append({"exercise_id": seg_label, "start_frame": seg_start, "end_frame": seg_end})
    return segments


@router.post("/predict/video", response_model=PredictionResponse)
async def predict_from_video(
    file: UploadFile = File(...),
    exercise_id: int = Form(..., description="Exercise type (0-8)"),
    dataset: str = Form(default="intellirehab", description="Dataset key: intellirehab | intellirehab_2d | uiprmd | uiprmd_2d"),
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


@router.post("/predict/video_timeline")
async def predict_video_timeline(
    file: UploadFile = File(...),
    dataset: str = Form(default="intellirehab", description="Dataset key"),
    model_name: str | None = Form(default=None, description="Model architecture to use"),
    window_size: int = Form(default=120, description="Window size in frames"),
    stride: int = Form(default=30, description="Window stride in frames"),
    smoothing_window: int = Form(default=5, description="Majority smoothing window over window-label sequence"),
    min_segment_frames: int = Form(default=60, description="Minimum segment length (frames) to keep"),
):
    """Predict exercise timeline and per-segment correctness for a mixed-exercise video."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video.")

    dataset_key = settings._normalize_dataset_key(dataset)
    exercises = settings.exercises_for(dataset_key)

    if window_size < 10:
        raise HTTPException(400, "window_size must be >= 10.")
    if stride < 1:
        raise HTTPException(400, "stride must be >= 1.")

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        service = _get_service(model_name, dataset=dataset_key)

        keypoints = service.pose_extractor.extract_from_video(tmp_path)
        keypoints = service._prepare_keypoints_for_dataset(keypoints)

        total_frames = int(keypoints.shape[0]) if keypoints.ndim >= 1 else 0
        if total_frames == 0:
            raise HTTPException(400, "No frames/keypoints extracted from video.")

        # Keep only exercises that have available weights.
        candidate_exercise_ids: list[int] = []
        for exercise_id in sorted(exercises.keys()):
            if os.path.exists(settings.weights_path_for(service.model_name, exercise_id, dataset=dataset_key)):
                candidate_exercise_ids.append(exercise_id)

        # Fallback: if no dataset-specific weights found, still attempt all exercises.
        if not candidate_exercise_ids:
            candidate_exercise_ids = sorted(exercises.keys())

        starts = _window_starts(total_frames, window_size, stride)
        window_predictions: list[dict[str, Any]] = []
        raw_labels: list[int] = []

        for start in starts:
            end = min(total_frames, start + window_size)
            clip = keypoints[start:end]
            candidates = service.score_exercises_for_keypoints(clip, exercise_ids=candidate_exercise_ids)
            best = candidates[0]
            raw_labels.append(int(best["exercise_id"]))
            window_predictions.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "exercise_id": int(best["exercise_id"]),
                    "exercise_name": str(best["exercise_name"]),
                    "score": float(best["score"]),
                    "predicted_label": str(best["predicted_label"]),
                }
            )

        smoothed_labels = _majority_smooth(raw_labels, smoothing_window)

        for i, label in enumerate(smoothed_labels):
            window_predictions[i]["smoothed_exercise_id"] = int(label)
            window_predictions[i]["smoothed_exercise_name"] = exercises.get(int(label), f"Exercise {label}")

        raw_segments = _labels_to_segments(smoothed_labels, starts, window_size)

        segments: list[dict[str, Any]] = []
        for seg in raw_segments:
            start_frame = max(0, int(seg["start_frame"]))
            end_frame = min(total_frames, int(seg["end_frame"]))
            if end_frame - start_frame < min_segment_frames:
                continue

            exercise_id = int(seg["exercise_id"])
            segment_keypoints = keypoints[start_frame:end_frame]
            result = service.predict_from_keypoints(segment_keypoints, exercise_id=exercise_id)

            segments.append(
                {
                    "exercise_id": exercise_id,
                    "exercise_name": exercises.get(exercise_id, f"Exercise {exercise_id}"),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": end_frame - start_frame,
                    "prediction": {
                        "label": result.get("label"),
                        "confidence": result.get("confidence"),
                        "problematic_joints": result.get("problematic_joints") or [],
                    },
                }
            )

        return {
            "dataset": dataset_key,
            "model_name": service.model_name,
            "total_frames": total_frames,
            "window_size": window_size,
            "stride": stride,
            "smoothing_window": smoothing_window,
            "min_segment_frames": min_segment_frames,
            "candidate_exercise_ids": candidate_exercise_ids,
            "num_windows": len(window_predictions),
            "num_segments": len(segments),
            "window_predictions": window_predictions,
            "segments": segments,
        }
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
            + _exercise_payload(model_name, "intellirehab_2d")
            + _exercise_payload(model_name, "uiprmd")
            + _exercise_payload(model_name, "uiprmd_2d")
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
    return (
        _exercise_payload(model_name, "intellirehab")
        + _exercise_payload(model_name, "intellirehab_2d")
        + _exercise_payload(model_name, "uiprmd")
        + _exercise_payload(model_name, "uiprmd_2d")
    )


@router.get("/reference/visualization")
async def get_reference_visualization(
    exercise_id: int = Query(..., description="Exercise id to preview"),
    dataset: str = Query("intellirehab", description="Dataset key"),
):
    """Return reference visualization video for a dataset/exercise pair."""
    dataset_key = _validate_exercise(dataset, exercise_id)

    candidates = [
        os.path.join(
            settings.weights_dir_for(dataset_key),
            f"reference_exercise_{exercise_id}_visualization.mp4",
        )
    ]

    # IntelliRehab 2D can reuse IntelliRehab reference preview videos.
    if dataset_key == "intellirehab_2d":
        candidates.append(
            os.path.join(
                settings.weights_dir_for("intellirehab"),
                f"reference_exercise_{exercise_id}_visualization.mp4",
            )
        )

    # Backward compatibility with older layouts that stored files in root weights dir.
    if dataset_key in {"intellirehab", "intellirehab_2d"}:
        candidates.append(
            os.path.join(
                settings.WEIGHTS_DIR,
                f"reference_exercise_{exercise_id}_visualization.mp4",
            )
        )

    reference_video = next((p for p in candidates if os.path.exists(p)), None)
    if reference_video is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Reference visualization not found for "
                f"dataset='{dataset_key}', exercise_id={exercise_id}."
            ),
        )

    media_type = "video/mp4"
    serve_path = reference_video

    # Existing reference previews in this repo are typically FMP4 (mp4v),
    # which some browsers fail to decode in HTML5 <video>.
    codec, _ = _probe_video(reference_video)
    if codec.upper() == "FMP4":
        converted = _transcode_reference_for_browser(reference_video)
        if converted is not None:
            serve_path, media_type = converted

    return FileResponse(
        serve_path,
        media_type=media_type,
        filename=os.path.basename(serve_path),
    )


@router.post("/predict/video_annotated")
async def predict_with_annotated_video(
    file: UploadFile = File(...),
    exercise_id: int = Form(..., description="Exercise type (0-8)"),
    dataset: str = Form(default="intellirehab", description="Dataset key: intellirehab | intellirehab_2d | uiprmd | uiprmd_2d"),
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
            output_path,
            exercise_id=exercise_id,
            dataset=dataset_key,
            deviation_window_size=5,
        )

        serve_path = output_path
        codec, _ = _probe_video(output_path)
        if codec.upper() == "FMP4":
            converted_h264 = _transcode_h264_mp4(output_path)
            if converted_h264 is not None:
                serve_path = converted_h264
        
        # Return the annotated video
        return FileResponse(
            serve_path,
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


@router.post("/predict/video_annotated_segment")
async def predict_with_annotated_video_segment(
    file: UploadFile = File(...),
    exercise_id: int = Form(..., description="Exercise type (0-8)"),
    start_frame: int = Form(..., description="Start frame (inclusive)"),
    end_frame: int = Form(..., description="End frame (exclusive)"),
    dataset: str = Form(default="intellirehab", description="Dataset key"),
    model_name: str | None = Form(default=None, description="Model architecture to use"),
):
    """Annotate only a frame-range segment from uploaded video."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video.")
    if start_frame < 0 or end_frame <= start_frame:
        raise HTTPException(400, "Invalid frame range.")

    dataset_key = _validate_exercise(dataset, exercise_id)

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    segment_path: str | None = None
    try:
        segment_path = _extract_video_segment(tmp_path, start_frame, end_frame)
        if segment_path is None:
            raise HTTPException(400, "Failed to extract requested video segment.")

        service = _get_service(model_name, dataset=dataset_key)

        keypoints = service.pose_extractor.extract_from_video(segment_path)
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
            deviations = {}

        output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = output_tmp.name
        output_tmp.close()

        service.explainability.create_annotated_video(
            segment_path,
            keypoints,
            deviations,
            output_path,
            exercise_id=exercise_id,
            dataset=dataset_key,
            deviation_window_size=5,
        )

        serve_path = output_path
        codec, _ = _probe_video(output_path)
        if codec.upper() == "FMP4":
            converted_h264 = _transcode_h264_mp4(output_path)
            if converted_h264 is not None:
                serve_path = converted_h264

        return FileResponse(
            serve_path,
            media_type="video/mp4",
            filename=f"annotated_seg_{exercise_id}_{start_frame}_{end_frame}.mp4",
            headers={
                "X-Prediction-Label": result["label"],
                "X-Prediction-Confidence": str(result["confidence"]),
                "X-Problematic-Joints": ",".join(service.explainability.get_problematic_joints(deviations)),
            },
        )
    finally:
        _safe_unlink(tmp_path)
        if segment_path:
            _safe_unlink(segment_path)


@router.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}