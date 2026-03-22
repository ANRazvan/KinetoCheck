"""
Training factory for UI-PRMD angles dataset.

Supports both Vicon (high-quality) and Kinect (realistic noise) modalities.
Enables two-stage training: pretrain on Vicon → fine-tune on Kinect.
"""

from training.training_factory import AbstractTrainingFactory
from torch.utils.data import Dataset
from app.models.base_model import BaseMovementModel
from app.models.model_factory import ModelFactory


class UIPRMDAnglesTrainingFactory(AbstractTrainingFactory):
    """
    Creates training components for UI-PRMD angles dataset.

    Compared to UIPRMDTrainingFactory (which uses positions):
    - Loads pre-computed joint angles instead of positions
    - Supports Vicon (professional) and Kinect (consumer) modalities
    - Enables domain adaptation: Vicon pretraining → Kinect fine-tuning
    """

    def __init__(self, modality: str = "vicon", feature_dim: int | None = None):
        """
        Args:
            modality: 'vicon' or 'kinect' — which data modality to use.
        """
        if modality.lower() not in ("vicon", "kinect"):
            raise ValueError(f"Unknown modality: {modality}. Use 'vicon' or 'kinect'.")
        self._modality = modality.lower()
        self._feature_dim = feature_dim

    @property
    def dataset_name(self) -> str:
        """Human-readable name including modality."""
        return f"UI-PRMD ({self._modality.capitalize()}) Angles"

    @property
    def num_joints(self) -> int:
        """
        Number of angle features (not joints!).

        Vicon: ~130-150 angles derived from 17 joints
        Kinect: ~60-70 angles (coarser model)
        """
        # Dynamically inferred during dataset creation
        return 17  # Placeholder; actual value determined by data

    def create_dataset(
        self,
        data_dir: str,
        exercise_id: int | None = None,
        seq_length: int | None = None,
        use_segmented: bool = True,
    ) -> Dataset:
        """
        Create angles dataset.

        Args:
            data_dir: Root UI-PRMD directory
            exercise_id: Filter by exercise (0-9) or None for all
            seq_length: Override default sequence length
            use_segmented: If True, load only Segmented Movements/

        Returns:
            UIPRMDAnglesDataset
        """
        from training.uiprmd_angles_dataset import UIPRMDAnglesDataset

        return UIPRMDAnglesDataset(
            data_dir,
            modality=self._modality,
            exercise_id=exercise_id,
            seq_length=seq_length,
            use_segmented=use_segmented,
            feature_dim=self._feature_dim,
        )

    def create_preprocessor(self, seq_length: int | None = None):
        """Create angles preprocessor."""
        from app.preprocessing.uiprmd_angles_preprocessor import UIPRMDAnglesPreprocessor

        return UIPRMDAnglesPreprocessor(seq_length, target_dim=self._feature_dim)

    def create_model(self, model_name: str) -> BaseMovementModel:
        """Create model (same architecture for both modalities)."""
        return ModelFactory.create(model_name)
