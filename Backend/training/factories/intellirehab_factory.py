from training.training_factory import AbstractTrainingFactory
from torch.utils.data import Dataset
from app.models.base_model import BaseMovementModel
from app.models.model_factory import ModelFactory

class IntelliRehabTrainingFactory(AbstractTrainingFactory):
    """
    Creates training components for the IntelliRehab Kinect dataset.
    """
    @property
    def dataset_name(self) -> str:
        return "IntelliRehab"

    @property
    def num_joints(self) -> int:
        return 25  # Kinect skeleton

    def create_dataset(self, data_dir: str, exercise_id: int | None = None, seq_length: int | None = None) -> Dataset:
        from training.skeleton_dataset import SkeletonDataset
        return SkeletonDataset(data_dir, exercise_id=exercise_id, seq_length=seq_length)

    def create_preprocessor(self, seq_length: int | None = None):
        from app.preprocessing.skeleton_preprocessor import SkeletonPreprocessor
        return SkeletonPreprocessor(seq_length)

    def create_model(self, model_name: str) -> BaseMovementModel:
        return ModelFactory.create(model_name)
