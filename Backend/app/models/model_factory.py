from typing import Dict, Type
from app.models.base_model import BaseMovementModel
from app.models.stgat_model import STGATModel


class ModelFactory:
    """
    Factory + Registry for model strategies.
    Register new models here when you add them in the future.
    """

    _registry: Dict[str, Type[BaseMovementModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseMovementModel]) -> None:
        cls._registry[name] = model_class

    @classmethod
    def create(cls, name: str) -> BaseMovementModel:
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model '{name}'. Available: {available}")
        return cls._registry[name]()

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())


# ── Register all available models ────────────────────────────────────
ModelFactory.register("stgat", STGATModel)

# Future: just add one line
# from app.models.inception_time_model import InceptionTimeModel
# ModelFactory.register("inception_time", InceptionTimeModel)