import os
from typing import Dict


class Settings:
    """
    Application-wide settings.

    Implemented as a **Singleton** via ``__new__``: every call to
    ``Settings()`` returns the same instance, so there is always exactly
    one configuration object in the process.
    """

    _instance: "Settings | None" = None

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    PROJECT_NAME: str = "KinetoCheck Backend"
    VERSION: str = "1.0.0"

    # Model selection: "stgat", "inception_time", "lstm", etc.
    ACTIVE_MODEL: str = os.getenv("ACTIVE_MODEL", "stgat")

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_DIR: str = os.path.join(BASE_DIR, "weights")
    YOLO_WEIGHTS: str = os.getenv("YOLO_WEIGHTS", "yolov8n-pose.pt")
    DATASET_ROOT: str = os.path.join(BASE_DIR, "..", "Datasets")
    INTELLIREHAB_DATA_DIR: str = os.getenv(
        "INTELLIREHAB_DATA_DIR",
        os.path.join(DATASET_ROOT, "SkeletonData", "Simplified"),
    )
    UIPRMD_DATA_DIR: str = os.getenv(
        "UIPRMD_DATA_DIR",
        os.path.join(DATASET_ROOT, "uiprmd"),
    )
    # Backward-compatible alias (legacy code may still read settings.DATA_DIR)
    DATA_DIR: str = INTELLIREHAB_DATA_DIR

    # ── Exercise registries ──────────────────────────────────────────
    # IntelliRehab maps exercise ID (from filename) → human-readable name
    INTELLIREHAB_EXERCISES: Dict[int, str] = {
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

    # UI-PRMD maps movement IDs (m01/e01 .. m10/e10) to names.
    # IDs are normalized to 0-based in the loader, so 0..9.
    UIPRMD_EXERCISES: Dict[int, str] = {
        0: "Deep Squat",
        1: "Hurdle Step",
        2: "Inline Lunge",
        3: "Side Lunge",
        4: "Sit To Stand",
        5: "Standing Active Straight Leg Raise",
        6: "Standing Shoulder Abduction",
        7: "Standing Shoulder Extension",
        8: "Standing Shoulder Internal-External Rotation",
        9: "Standing Trunk Rotation",
    }

    # Backward-compatible alias used by older call-sites.
    EXERCISES: Dict[int, str] = INTELLIREHAB_EXERCISES

    # Pose extractor backend: "yolo" (swap to e.g. "mediapipe" later)
    POSE_EXTRACTOR: str = os.getenv("POSE_EXTRACTOR", "yolo")

    # Model hyperparameters
    NUM_KEYPOINTS: int = 25  # Kinect skeleton joints (IntelliRehab)
    KEYPOINT_DIM: int = 3  # x, y, z
    UIPRMD_NUM_KEYPOINTS: int = 17
    UIPRMD_KEYPOINT_DIM: int = 3
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

    # ── Angle-based Training (UI-PRMD) ────────────────────────────────
    # Feature dimensions for angle-based models
    ANGLES_VICON_DIM: int = 130          # Vicon: ~130 angles from 17 joints
    ANGLES_KINECT_DIM: int = 65          # Kinect: ~65 angles (coarser)

    # Augmentation for angle data (video robustness)
    AUGMENT_ANGLE_NOISE_STD: float = 0.5  # Gaussian noise on angle values (degrees)
    AUGMENT_FRAME_DROPOUT: float = 0.05   # Probability of dropping each frame
    AUGMENT_SCALE_JITTER: tuple[float, float] = (0.95, 1.05)  # Scale variation range

    # Three-stage training rates
    LR_STAGE1_ANGLES: float = 1e-3        # Stage 1: Vicon pretraining
    LR_STAGE2_ANGLES: float = 5e-4        # Stage 2: Kinect fine-tuning
    LR_STAGE3_ANGLES: float = 1e-4        # Stage 3: Mixed + augmentation

    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 15
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # Device: "cuda", "cpu", or "auto" (auto-detect)
    DEVICE: str = os.getenv("DEVICE", "auto")

    # ── Helpers ──────────────────────────────────────────────────────
    def _normalize_dataset_key(self, dataset: str) -> str:
        key = (dataset or "intellirehab").strip().lower()
        aliases = {
            "intelli": "intellirehab",
            "intelli_rehab": "intellirehab",
            "ui-prmd": "uiprmd",
            "ui_prmd": "uiprmd",
            "ui": "uiprmd",
        }
        return aliases.get(key, key)

    def data_dir_for(self, dataset: str = "intellirehab") -> str:
        """Return dataset-specific data directory."""
        key = self._normalize_dataset_key(dataset)
        if key == "intellirehab":
            return self.INTELLIREHAB_DATA_DIR
        if key == "uiprmd":
            return self.UIPRMD_DATA_DIR
        raise ValueError(f"Unknown dataset '{dataset}'. Expected: intellirehab | uiprmd")

    def weights_dir_for(self, dataset: str = "intellirehab") -> str:
        """Return dataset-specific weights directory under Backend/weights/."""
        key = self._normalize_dataset_key(dataset)
        return os.path.join(self.WEIGHTS_DIR, key)

    def weights_path_for(
        self,
        model_name: str,
        exercise_id: int,
        dataset: str = "intellirehab",
    ) -> str:
        """Return dataset-specific weights path for a given model + exercise."""
        weights_dir = self.weights_dir_for(dataset)
        return os.path.join(weights_dir, f"{model_name}_exercise_{exercise_id}_best.pt")

    def exercises_for(self, dataset: str = "intellirehab") -> Dict[int, str]:
        """Return exercise registry for the selected dataset."""
        key = self._normalize_dataset_key(dataset)
        if key == "intellirehab":
            return self.INTELLIREHAB_EXERCISES
        if key == "uiprmd":
            return self.UIPRMD_EXERCISES
        raise ValueError(f"Unknown dataset '{dataset}'. Expected: intellirehab | uiprmd")

    def exercise_name_for(self, dataset: str, exercise_id: int) -> str:
        """Return dataset-specific exercise name."""
        return self.exercises_for(dataset).get(exercise_id, f"Exercise {exercise_id}")

    def exercise_name(self, exercise_id: int) -> str:
        # Legacy behavior: IntelliRehab mapping.
        return self.EXERCISES.get(exercise_id, f"Exercise {exercise_id}")


settings = Settings()