"""
Training configuration and utilities for Temporal Pyramid STGAT.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PyramidSTGATConfig:
    """Configuration for Temporal Pyramid STGAT model."""
    
    # Dataset
    dataset_root: str = "Datasets/UIPRMD"
    exercise_id: Optional[int] = None  # None = all exercises
    seq_length: int = 240
    
    # Model architecture
    in_channels_coord: int = 3
    in_channels_angle: int = 13  # Number of angles computed
    hidden_channels: int = 64
    num_heads: int = 4
    num_joints: int = 17
    num_scales: int = 4
    temporal_scales: List[int] = None
    dropout: float = 0.3
    num_classes: int = 2
    frame_head_type: str = "gru"  # "mlp" | "gru" | "1dcnn"
    frame_head_hidden: int = 128
    frame_aggregation: str = "topk_mean"  # "mean" | "max" | "topk_mean" | "noisy_or"
    frame_topk_ratio: float = 0.2
    
    # Training
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 20
    validation_split: float = 0.2
    seed: int = 42
    
    # Data preprocessing
    normalize_coords: bool = True
    normalize_angles: bool = True
    use_temporal_pyramid: bool = True
    pad_mode: str = "constant"  # or "replicate"
    
    # Training strategy
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    
    # Inference
    model_save_dir: str = "temporal_pyramid_stgat/weights"
    checkpoint_name: str = "pyramid_stgat_best.pt"
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.temporal_scales is None:
            self.temporal_scales = [1, 2, 4, 8]
    
    @staticmethod
    def for_ui_prmd() -> "PyramidSTGATConfig":
        """Pre-configured for UI-PRMD dataset."""
        return PyramidSTGATConfig(
            dataset_root="Datasets/UIPRMD",
            seq_length=240,
            num_scales=4,
            temporal_scales=[1, 2, 4, 8],
        )
    
    @staticmethod
    def for_uiprmd_single_exercise(exercise_id: int) -> "PyramidSTGATConfig":
        """Pre-configured for single UI-PRMD exercise."""
        config = PyramidSTGATConfig.for_ui_prmd()
        config.exercise_id = exercise_id
        config.checkpoint_name = f"pyramid_stgat_exercise_{exercise_id}_best.pt"
        return config
    
    @staticmethod
    def for_uiprmd_mediapipe_33joint(exercise_id: Optional[int] = None) -> "PyramidSTGATConfig":
        """
        Pre-configured for UI-PRMD retraining with MediaPipe 33-joint representation.
        
        This configuration matches the MediaPipe 33-landmark format used in video inference,
        ensuring consistent feature representation across training and deployment.
        
        Args:
            exercise_id: Optional filter by exercise ID (None = train on all exercises)
            
        Returns:
            PyramidSTGATConfig with num_joints=33 and angle features from 12 triplets
        """
        config = PyramidSTGATConfig.for_ui_prmd()
        config.num_joints = 33
        # Model expects per-joint coordinate channels (x, y, z), not flattened J*3.
        config.in_channels_coord = 3
        config.in_channels_angle = 12  # 12 anatomical angle triplets from MediaPipe
        config.exercise_id = exercise_id
        config.checkpoint_name = (
            f"pyramid_stgat_mediapipe33_exercise_{exercise_id}_best.pt"
            if exercise_id is not None
            else "pyramid_stgat_mediapipe33_best.pt"
        )
        return config
