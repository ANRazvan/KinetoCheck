import os


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
    BATCH_SIZE: int = 32
    EPOCHS: int = 100

    # Device: "cuda", "cpu", or "auto" (auto-detect)
    DEVICE: str = os.getenv("DEVICE", "auto")


settings = Settings()