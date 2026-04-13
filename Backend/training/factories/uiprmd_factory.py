from training.training_factory import AbstractTrainingFactory
from torch.utils.data import Dataset
from app.models.base_model import BaseMovementModel
from app.models.model_factory import ModelFactory
from config import settings

class UIPRMDTrainingFactory(AbstractTrainingFactory):
    """
    Creates training components for the UI-PRMD Vicon dataset.
    """
    def __init__(self, keypoint_dim: int | None = None):
        self._keypoint_dim = keypoint_dim or settings.UIPRMD_KEYPOINT_DIM

    @property
    def dataset_name(self) -> str:
        return "UI-PRMD"

    @property
    def num_joints(self) -> int:
        return 17  # Vicon anatomical landmarks

    @property
    def keypoint_dim(self) -> int:
        return self._keypoint_dim

    def create_dataset(self, data_dir: str, exercise_id: int | None = None, seq_length: int | None = None) -> Dataset:
        from training.uiprmd_dataset import UIPRMDDataset
        return UIPRMDDataset(
            data_dir,
            exercise_id=exercise_id,
            seq_length=seq_length,
            keypoint_dim=self._keypoint_dim,
        )

    def create_preprocessor(self, seq_length: int | None = None):
        from app.preprocessing.uiprmd_preprocessor import UIPRMDPreprocessor
        return UIPRMDPreprocessor(seq_length=seq_length, keypoint_dim=self._keypoint_dim)

    def create_model(self, model_name: str) -> BaseMovementModel:
        return ModelFactory.create(model_name)
