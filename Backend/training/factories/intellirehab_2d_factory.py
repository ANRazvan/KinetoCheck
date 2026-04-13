from training.training_factory import AbstractTrainingFactory
from torch.utils.data import Dataset
from app.models.base_model import BaseMovementModel
from app.models.model_factory import ModelFactory


class IntelliRehab2DTrainingFactory(AbstractTrainingFactory):
    """Creates training components for IntelliRehab 2D (XY only)."""

    @property
    def dataset_name(self) -> str:
        return "IntelliRehab 2D"

    @property
    def num_joints(self) -> int:
        return 25

    @property
    def keypoint_dim(self) -> int:
        return 2

    def create_dataset(
        self,
        data_dir: str,
        exercise_id: int | None = None,
        seq_length: int | None = None,
    ) -> Dataset:
        from training.intellirehab_2d_dataset import IntelliRehab2DDataset

        return IntelliRehab2DDataset(data_dir, exercise_id=exercise_id, seq_length=seq_length)

    def create_preprocessor(self, seq_length: int | None = None):
        from app.preprocessing.intellirehab_2d_preprocessor import IntelliRehab2DPreprocessor

        return IntelliRehab2DPreprocessor(seq_length)

    def create_model(self, model_name: str) -> BaseMovementModel:
        return ModelFactory.create(model_name)
