from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path
from collections.abc import Sequence
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KinetoCheck Inference API",
    description="Upload a video and get AI-powered exercise form analysis.",
)

# Find workspace root: api.py is in AI/ subfolder, so go up one level
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

TMP_DIR = BASE_DIR / "tmp_api_uploads"
TMP_DIR.mkdir(exist_ok=True)


def _coerce_scalar(value: object, default: str = "auto") -> str:
    current = value
    while isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
        if len(current) == 0:
            return default
        current = current[0]

    if current is None:
        return default

    return str(current)


@app.get("/")
def root():
    """Health check and API info."""
    return {
        "service": "KinetoCheck Inference API",
        "status": "running",
        "docs": "http://127.0.0.1:8000/docs",
        "endpoint": "POST /analyze-video/",
    }


@app.post("/analyze-video/")
async def analyze_video(video: UploadFile = File(...), exercise_id: str = Form("auto")):
    if not video.filename.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save uploaded file to a temp path
    session_id = uuid.uuid4().hex
    tmp_dir = TMP_DIR / session_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / video.filename

    with tmp_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    # Create a persistent output directory for annotated videos
    PERSISTENT_UPLOADS = BASE_DIR / "tmp_api_uploads" / "annotated_videos"
    PERSISTENT_UPLOADS.mkdir(parents=True, exist_ok=True)

    # Import local processing function
    try:
        from inference.inference_engine import (
            process_video,
            get_cached_models,
        )
        from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to import processing module: {exc}")

    # Minimal runtime setup mirroring app.get runtime
    try:
        from inference.inference_engine import ensure_pose_task_model, resolve_device
        device = resolve_device("auto")
        checkpoints_root = BASE_DIR / "checkpoints" / "uiprmd_phase_aware_rom"
        # checkpoints_root = BASE_DIR / "checkpoints" / "uiprmd"
        all_models = get_cached_models(checkpoints_root, device)
        preprocessor = UIPRMDPreprocessor()
        pose_model_path = ensure_pose_task_model(None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model runtime: {exc}")

    # Filter models based on selected exercise
    models_to_run = all_models
    exercise_id = _coerce_scalar(exercise_id)
    logger.info(f"Received exercise_id: {exercise_id}")

    if exercise_id and exercise_id != "auto":
        try:
            target_id = int(exercise_id) - 1  # Convert 1-indexed to 0-indexed
            logger.info(f"Filtering models for exercise_id={exercise_id} (target_id={target_id})")
            models_to_run = [m for m in all_models if m.exercise_id == target_id]
            for m in models_to_run:
                logger.info(f"Selected model for exercise {m.exercise_id}")
            if not models_to_run:
                raise HTTPException(status_code=400, detail=f"No model found for exercise {exercise_id}")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid exercise_id: {exercise_id}")

    # Run processing (synchronous call)
    try:
        report = process_video(
            video_path=tmp_path,
            models=models_to_run,
            preprocessor=preprocessor,
            output_dir=PERSISTENT_UPLOADS,
            device=device,
            pose_model_path=pose_model_path,
        )
    except Exception as exc:
        logger.exception("Processing failed")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    # removing the temporary directory (but NOT the persistent uploads)
    try:
        shutil.rmtree(tmp_dir)
    except Exception as exc:
        logger.warning(f"Failed to clean up temp directory {tmp_dir}: {exc}")

    # Return JSON summary; include session id and paths
    report["session_id"] = session_id
    return JSONResponse(content=report)


def __main__():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    __main__()