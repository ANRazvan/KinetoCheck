"""
Tests for the inference pipeline.
Run with:  cd Backend && python -m pytest tests/ -v
"""
import numpy as np
import pytest

from app.models.model_factory import ModelFactory
from app.models.base_model import BaseMovementModel
from app.preprocessing.skeleton_preprocessor import SkeletonPreprocessor
from config import settings


# ── Model Factory Tests ──────────────────────────────────────────────

class TestModelFactory:
    def test_stgat_is_registered(self):
        assert "stgat" in ModelFactory.list_models()

    def test_create_stgat(self):
        model = ModelFactory.create("stgat")
        assert isinstance(model, BaseMovementModel)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("nonexistent_model")


# ── Preprocessor Tests ───────────────────────────────────────────────

class TestSkeletonPreprocessor:
    def setup_method(self):
        self.preprocessor = SkeletonPreprocessor(seq_length=30)

    def test_output_shape_longer_input(self):
        """Input with more frames than seq_length should be downsampled."""
        keypoints = np.random.rand(100, settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM)
        result = self.preprocessor.process(keypoints)
        assert result.shape == (30, settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM)

    def test_output_shape_shorter_input(self):
        """Input with fewer frames than seq_length should be padded."""
        keypoints = np.random.rand(10, settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM)
        result = self.preprocessor.process(keypoints)
        assert result.shape == (30, settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM)

    def test_normalization_centered(self):
        """After normalization, values should be in [-1, 1] range."""
        keypoints = np.random.rand(20, settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM) * 640
        result = self.preprocessor.process(keypoints)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_dtype_is_float32(self):
        keypoints = np.random.rand(20, settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM)
        result = self.preprocessor.process(keypoints)
        assert result.dtype == np.float32


# ── ST-GAT Model Tests ───────────────────────────────────────────────

class TestSTGATModel:
    def setup_method(self):
        self.model = ModelFactory.create("stgat")
        self.model.build(
            num_keypoints=settings.NUM_KEYPOINTS,
            keypoint_dim=settings.KEYPOINT_DIM,
            hidden_dim=32,      # smaller for fast tests
            num_classes=2,
            num_layers=1,       # fewer layers for speed
            num_heads=2,
            seq_length=16,
            dropout=0.1,
        )

    def test_predict_returns_dict(self):
        dummy = np.random.randn(16, settings.NUM_KEYPOINTS, settings.KEYPOINT_DIM).astype(np.float32)
        result = self.model.predict(dummy)
        assert "label" in result
        assert "confidence" in result
        assert result["label"] in ("correct", "incorrect")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_get_model_info(self):
        info = self.model.get_model_info()
        assert info["name"] == "ST-GAT"
        assert info["parameters"] > 0

    def test_predict_without_build_raises(self):
        fresh_model = ModelFactory.create("stgat")
        with pytest.raises(RuntimeError):
            fresh_model.predict(np.zeros((16, 17, 2), dtype=np.float32))


# ── FastAPI Endpoint Tests ────────────────────────────────────────────

class TestAPIEndpoints:
    def setup_method(self):
        from fastapi.testclient import TestClient
        from app import create_app
        self.client = TestClient(create_app())

    def test_health(self):
        resp = self.client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_models(self):
        resp = self.client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "stgat" in data["available_models"]
