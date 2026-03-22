from training.training_factory import AbstractTrainingFactory
from torch.utils.data import Dataset
from app.models.base_model import BaseMovementModel
from app.models.model_factory import ModelFactory

class UIPRMDTrainingFactory(AbstractTrainingFactory):
    """
    Creates training components for the UI-PRMD Vicon dataset.
    """
    @property
    def dataset_name(self) -> str:
        return "UI-PRMD"

    @property
    def num_joints(self) -> int:
        return 17  # Vicon anatomical landmarks

    def create_dataset(self, data_dir: str, exercise_id: int | None = None, seq_length: int | None = None) -> Dataset:
        from training.uiprmd_dataset import UIPRMDDataset
        return UIPRMDDataset(data_dir, exercise_id=exercise_id, seq_length=seq_length)

    def create_preprocessor(self, seq_length: int | None = None):
        from app.preprocessing.uiprmd_preprocessor import UIPRMDPreprocessor
        return UIPRMDPreprocessor(seq_length)

    def create_model(self, model_name: str) -> BaseMovementModel:
        return ModelFactory.create(model_name)
