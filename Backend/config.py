import os
from typing import Dict


class Settings:
    PROJECT_NAME: str = "KinetoCheck Backend"
    VERSION: str = "1.0.0"

    # Model selection: "stgat", "inception_time", "lstm", etc.
    ACTIVE_MODEL: str = os.getenv("ACTIVE_MODEL", "stgat")

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_DIR: str = os.path.join(BASE_DIR, "weights")
    YOLO_WEIGHTS: str = os.path.join(BASE_DIR, "..", "MovementCorrectness-main", "yolov8n-pose.pt")
    DATA_DIR: str = os.path.join(BASE_DIR, "..", "SkeletonData", "Simplified")

    # ── Exercise registry ────────────────────────────────────────────
    # Maps exercise ID (from IntelliRehab filename) → human-readable name
    EXERCISES: Dict[int, str] = {
        0: "Shoulder Flexion",
        1: "Shoulder Abduction",
        2: "Shoulder Forward Extension",
        3: "Elbow Flexion",
        4: "Shoulder Horizontal Abduction",
        5: "Shoulder Rotation",
        6: "Forearm Pronation/Supination",
        7: "Wrist Flexion/Extension",
        8: "Hand to Mouth",
    }

    # Pose extractor backend: "yolo" (swap to e.g. "mediapipe" later)
    POSE_EXTRACTOR: str = os.getenv("POSE_EXTRACTOR", "yolo")

    # Model hyperparameters
    NUM_KEYPOINTS: int = 25  # Kinect skeleton joints (IntelliRehab)
    KEYPOINT_DIM: int = 3  # x, y, z
    NUM_CLASSES: int = 2  # correct / incorrect
    SEQUENCE_LENGTH: int = 120  # number of frames per clip

    # COCO keypoints (for YOLO pose extraction at inference)
    COCO_NUM_KEYPOINTS: int = 17
    COCO_KEYPOINT_DIM: int = 2  # x, y

    # GAT specific
    GAT_HIDDEN_DIM: int = 64
    GAT_NUM_HEADS: int = 4
    GAT_NUM_LAYERS: int = 3
    GAT_DROPOUT: float = 0.3

    # Training
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 32
    EPOCHS: int = 100

    # Optimisation knobs
    GRAD_CLIP_NORM: float = 1.0          # max-norm for gradient clipping (0 = off)
    USE_AMP: bool = True                  # automatic mixed-precision (CUDA only)
    PIN_MEMORY: bool = True               # pin_memory in DataLoaders (CUDA only)
    DATALOADER_WORKERS: int = 0           # num_workers (0 is safest on Windows)

    # LR scheduler (ReduceLROnPlateau)
    LR_SCHEDULER_FACTOR: float = 0.5
    LR_SCHEDULER_PATIENCE: int = 7
    LR_SCHEDULER_MIN_LR: float = 1e-6

    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 15
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # Device: "cuda", "cpu", or "auto" (auto-detect)
    DEVICE: str = os.getenv("DEVICE", "auto")

    # ── Helpers ──────────────────────────────────────────────────────
    def weights_path_for(self, model_name: str, exercise_id: int) -> str:
        """Return the weights file path for a given model + exercise."""
        return os.path.join(self.WEIGHTS_DIR, f"{model_name}_exercise_{exercise_id}_best.pt")

    def exercise_name(self, exercise_id: int) -> str:
        return self.EXERCISES.get(exercise_id, f"Exercise {exercise_id}")


settings = Settings()