from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_ROOT = BASE_DIR / "checkpoints" / "uiprmd_phase_aware_rom"
UPLOAD_ROOT = BASE_DIR / "Video-kineto-annotated" / "web_uploads"
ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv", "avi", "webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@lru_cache(maxsize=1)
def get_runtime() -> tuple[list, UIPRMDPreprocessor, tuple[object, Path]]:
    from phase_aware.video_checkpoint_inference_phase_aware import ensure_pose_task_model, load_models, resolve_device

    device = resolve_device("auto")
    models = load_models(CHECKPOINTS_ROOT, device)
    preprocessor = UIPRMDPreprocessor()
    pose_model_path = ensure_pose_task_model(None)
    return models, preprocessor, (device, pose_model_path)


def get_available_exercises(models: list) -> list[dict]:
    """Extracts a unique list of available exercises from the loaded models."""
    unique_models = {m.exercise_id: m for m in models}.values()
    return [
        {"id": m.exercise_id, "name": f"Exercise {m.exercise_id + 1:02d}"}
        for m in sorted(unique_models, key=lambda x: x.exercise_id)
    ]


def prepare_template_args(report: dict | None = None, error: str | None = None, models: list | None = None) -> dict:
    """Helper to structure the variables needed by the HTML template."""
    args = {
        "report": report,
        "error": error,
        "best": report.get("best", {}) if report else {},
        "all_results": report.get("all", []) if report else [],
        "worst_summary": "N/A",
        "feedback_summary": report.get("feedback_summary") if report else None,
        "annotated_url": None,
        "codec": None,
        "codec_warning": None,
        "available_exercises": get_available_exercises(models) if models else []
    }

    if report:
        worst_joints = report.get("worst_joints", [])
        if worst_joints:
            args["worst_summary"] = ", ".join(item["joint"] for item in worst_joints[:3])
            
        args["annotated_url"] = url_for(
            "uploaded_file", 
            session_id=report["session_id"], 
            filename=report["annotated_video_rel"]
        )
        args["codec"] = report.get("annotated_video_codec")
        if args["codec"] == "mp4v":
            args["codec_warning"] = (
                "This file was encoded with mp4v, which some browsers cannot play inline. "
                "Use the download link below, or install ffmpeg and re-run to transcode into H.264."
            )
            
    return args


@app.get("/")
def index() -> str:
    models, _, _ = get_runtime()
    template_args = prepare_template_args(models=models)
    return render_template("index.html", **template_args)


@app.post("/")
def upload_video() -> str:
    models, preprocessor, runtime = get_runtime()
    device, pose_model_path = runtime
    template_args = prepare_template_args(models=models)

    # 1. Validate File
    if "video" not in request.files:
        template_args["error"] = "No file field was submitted."
        return render_template("index.html", **template_args), 400

    uploaded = request.files["video"]
    if uploaded.filename == "":
        template_args["error"] = "Choose a video file before uploading."
        return render_template("index.html", **template_args), 400

    if not allowed_file(uploaded.filename):
        template_args["error"] = "Unsupported file type. Use a video file such as MP4 or MOV."
        return render_template("index.html", **template_args), 400

    # 2. Handle Exercise Selection
    selected_ex_id = request.form.get("exercise_id")
    models_to_run = models
    print(selected_ex_id)

    if selected_ex_id and selected_ex_id != "auto":
        target_id = int(selected_ex_id) - 1  # Convert from 1-indexed to 0-indexed
        # Filter the models to ONLY include the selected exercise
        models_to_run = [m for m in models if m.exercise_id == target_id]
        for m in models_to_run:
            print(f"Selected model for exercise {m.exercise_id}: {m.model_path}")
        if not models_to_run:
            template_args["error"] = "Selected exercise model not found."
            return render_template("index.html", **template_args), 400

    # 3. Setup Directories & Save File
    session_id = uuid.uuid4().hex
    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(uploaded.filename)
    input_path = session_dir / filename
    uploaded.save(input_path)

    # 4. Process Video
    from phase_aware.video_checkpoint_inference_phase_aware import process_video
    
    report = process_video(
        video_path=input_path,
        models=models_to_run,  # Pass the filtered models list here
        preprocessor=preprocessor,
        output_dir=session_dir,
        device=device,
        pose_model_path=pose_model_path,
    )

    annotated_path = Path(report["annotated_video"])
    report["annotated_video_rel"] = annotated_path.name
    report["session_id"] = session_id

    report_path = session_dir / f"{input_path.stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Re-render with success
    template_args = prepare_template_args(report=report, models=models)
    return render_template("index.html", **template_args)


@app.get("/uploads/<session_id>/<path:filename>")
def uploaded_file(session_id: str, filename: str):
    directory = UPLOAD_ROOT / session_id
    if not directory.exists():
        abort(404)
    return send_from_directory(directory, filename, as_attachment=False)


if __name__ == "__main__":
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)