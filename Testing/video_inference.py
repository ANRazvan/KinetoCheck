import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Import your local modules
from Models.stgat_temporal_pyramid import ExerciseEvaluator
from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor

# COCO 17-joint mapping for MediaPipe
MP_COCO17_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

def ensure_pose_task_model(model_path: Path) -> Path:
    """Downloads the MediaPipe pose task file if it doesn't exist."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe pose model to {model_path}...")
    urllib.request.urlretrieve(DEFAULT_POSE_MODEL_URL, str(model_path))
    return model_path

def extract_mediapipe_sequence(video_path: Path, pose_model_path: Path) -> np.ndarray:
    """Runs MediaPipe on a video and returns raw (T, 17, 3) joint coordinates."""
    base_options = mp_python.BaseOptions(model_asset_path=str(pose_model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sequence = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            det = landmarker.detect_for_video(mp_img, ts_ms)

            if det.pose_landmarks:
                lms = det.pose_landmarks[0]
                pts = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in MP_COCO17_IDXS], dtype=np.float32)
                
                # Invert Y axis to match standard 3D coordinate spaces (optional based on your Vicon data)
                # pts[:, 1] = -pts[:, 1] 
                
                sequence.append(pts)
            frame_idx += 1

    cap.release()
    if not sequence:
        raise RuntimeError(f"No human detected in video: {video_path}")
    
    return np.stack(sequence, axis=0)

def prepare_tensor(sequence: np.ndarray, preprocessor: UIPRMDPreprocessor) -> torch.Tensor:
    """Formats the MediaPipe sequence exactly how the ST-GAT model expects it."""
    # 1. Align and center (Your preprocessor handles shape (T, 17, 3) automatically)
    aligned = preprocessor.align_vicon_to_mediapipe(sequence)
    
    # 2. Resample and normalize
    processed = preprocessor.process(aligned)  # (T, 17, 3)

    # 3. Calculate kinematics
    velocity = np.diff(processed, axis=0, prepend=processed[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])

    # 4. Stack into 9 channels and reshape
    features = np.concatenate([processed, velocity, acceleration], axis=-1)  # (T, 17, 9)
    features = np.transpose(features, (2, 0, 1)).copy()  # (9, T, 17)
    
    # Add batch dimension (1, 9, T, 17)
    return torch.from_numpy(features).float().unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a video using a trained ST-GAT checkpoint.")
    parser.add_argument("--video", type=Path, required=True, default = Path("Video-kineto/UIPRMD-videos/ex1-1-rep.mp4"), help="Path to the input video (.mp4)")
    parser.add_argument("--exercise-id", type=int, required=True, default = 1,help="Exercise ID to load (e.g., 0 for exercise_01)")
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints/uiprmd"), help="Directory containing exercise_XX folders")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cpu, cuda, auto)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Resolve Checkpoint Path
    exercise_dir = args.checkpoints_root / f"exercise_{args.exercise_id + 1:02d}"
    checkpoint_path = exercise_dir / "best_checkpoint.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: Could not find checkpoint at {checkpoint_path}")
        return

    print(f"Loading checkpoint for Exercise {args.exercise_id}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract Configuration and Parameters
    cfg_dict = checkpoint.get("config", {})
    hidden_channels = tuple(cfg_dict.get("hidden_channels", (64, 128)))
    embedding_dim = cfg_dict.get("embedding_dim", 128)
    val_threshold = checkpoint.get("val_threshold", 0.5)
    
    # The template tensor is saved as (9, T, 17). We need the exact T to configure the preprocessor.
    template_tensor = checkpoint["template_tensor"].to(device).unsqueeze(0) # Add batch dim: (1, 9, T, 17)
    target_seq_length = template_tensor.shape[2] 

    # Initialize Model
    model = ExerciseEvaluator(
        in_channels=9, 
        hidden_channels=hidden_channels, 
        embedding_dim=embedding_dim
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Extract & Preprocess Video
    print("Extracting MediaPipe landmarks from video...")
    pose_model_path = ensure_pose_task_model(Path(".cache/mediapipe/pose_landmarker_full.task"))
    raw_sequence = extract_mediapipe_sequence(args.video, pose_model_path)
    
    print(f"Preprocessing {raw_sequence.shape[0]} frames...")
    preprocessor = UIPRMDPreprocessor(seq_length=target_seq_length)
    user_tensor = prepare_tensor(raw_sequence, preprocessor).to(device)

    # Run Inference
    print("Evaluating movement...")
    with torch.no_grad():
        outputs = model(template_tensor, user_tensor)
        similarity_score = float(outputs["similarity_score"].item())

    # Assess result
    is_correct = similarity_score >= val_threshold

    # Print Report
    print("\n" + "="*40)
    print("         ASSESSMENT RESULTS")
    print("="*40)
    print(f"Video File       : {args.video.name}")
    print(f"Exercise ID      : {args.exercise_id}")
    print("-" * 40)
    print(f"Similarity Score : {similarity_score:.4f}")
    print(f"Model Threshold  : {val_threshold:.4f}")
    print("-" * 40)
    if is_correct:
        print("Verdict          : CORRECT \033[92m(Passed)\033[0m")
    else:
        print("Verdict          : INCORRECT \033[91m(Failed)\033[0m")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()