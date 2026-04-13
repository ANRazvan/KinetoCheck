import numpy as np

from app.preprocessing.uiprmd_preprocessor import UIPRMDPreprocessor
from app.services.inference_service import InferenceService
from training.training_factory import get_training_factory


class _DummyPoseExtractor:
    def extract_from_video(self, video_path: str):
        return np.zeros((10, 17, 2), dtype=np.float32)


class _DummyModel:
    def predict(self, input_data):
        return {"label": "correct", "confidence": 0.5}

    def get_model_info(self):
        return {"name": "dummy"}


class _DummyRepo:
    def get(self, exercise_id: int):
        return _DummyModel()


def test_uiprmd_2d_preprocessor_shapes_flat_input():
    pre = UIPRMDPreprocessor(seq_length=30, keypoint_dim=2)
    flat = np.random.rand(40, 34).astype(np.float32)  # 17 joints x 2 dims
    out = pre.process(flat)
    assert out.shape == (30, 17, 2)


def test_training_factory_uiprmd_2d_uses_dim_2():
    factory = get_training_factory("uiprmd_2d")
    assert factory.dataset_name == "UI-PRMD"
    assert factory.num_joints == 17
    assert factory.keypoint_dim == 2


def test_inference_service_uiprmd_2d_drops_z():
    service = InferenceService(
        dataset="uiprmd_2d",
        pose_extractor=_DummyPoseExtractor(),
        model_repository=_DummyRepo(),
    )

    xyz = np.random.rand(20, 17, 3).astype(np.float32)
    prepared = service._prepare_keypoints_for_dataset(xyz)

    assert prepared.shape == (20, 17, 2)
