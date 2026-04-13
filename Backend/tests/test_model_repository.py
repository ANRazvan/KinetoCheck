import os
from pathlib import Path

from app.services.model_repository import ModelRepository
from config import settings


class _FakeModel:
    def __init__(self):
        self.built = False
        self.loaded_paths: list[str] = []

    def build(self, **kwargs):
        self.built = True

    def load_weights(self, path: str):
        self.loaded_paths.append(path)


def test_model_repository_loads_dataset_specific_weights_first(monkeypatch, tmp_path):
    model_name = "stgat"
    exercise_id = 3
    fake_model = _FakeModel()

    dataset_dir = tmp_path / "weights" / "uiprmd"
    dataset_dir.mkdir(parents=True)
    preferred = dataset_dir / f"{model_name}_exercise_{exercise_id}_best.pt"
    preferred.write_text("ok", encoding="utf-8")

    legacy_dir = tmp_path / "weights"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy = legacy_dir / f"{model_name}_best.pt"
    legacy.write_text("legacy", encoding="utf-8")

    monkeypatch.setattr(
        "app.services.model_repository.ModelFactory.create",
        lambda name: fake_model,
    )
    monkeypatch.setattr(settings, "WEIGHTS_DIR", str(legacy_dir))
    monkeypatch.setattr(
        settings,
        "weights_dir_for",
        lambda dataset="intellirehab": str(dataset_dir if dataset == "uiprmd" else legacy_dir),
    )
    monkeypatch.setattr(
        settings,
        "weights_path_for",
        lambda model_name, exercise_id, dataset="intellirehab": str(
            Path(dataset_dir if dataset == "uiprmd" else legacy_dir)
            / f"{model_name}_exercise_{exercise_id}_best.pt"
        ),
    )

    repo = ModelRepository(model_name=model_name, dataset="uiprmd")
    model = repo.get(exercise_id)

    assert model is fake_model
    assert fake_model.built is True
    assert fake_model.loaded_paths == [str(preferred)]


def test_model_repository_caches_per_exercise(monkeypatch):
    created = []

    def _create(name):
        model = _FakeModel()
        created.append(model)
        return model

    monkeypatch.setattr("app.services.model_repository.ModelFactory.create", _create)
    monkeypatch.setattr(settings, "weights_path_for", lambda *args, **kwargs: "does_not_exist")
    monkeypatch.setattr(settings, "weights_dir_for", lambda *args, **kwargs: "does_not_exist")
    monkeypatch.setattr(settings, "WEIGHTS_DIR", "does_not_exist")

    repo = ModelRepository(model_name="stgat")
    m1 = repo.get(1)
    m2 = repo.get(1)

    assert m1 is m2
    assert len(created) == 1
