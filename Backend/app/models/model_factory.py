"""
ModelFactory — registry that maps model names to AbstractModelFactory subclasses.

This bridges the Abstract Factory pattern (app/models/abstract_factory.py) with
the rest of the codebase, which calls ``ModelFactory.create(name)`` to get a
fresh model instance without knowing the concrete factory or model class.

Registering a new model family requires only one line::

    ModelFactory.register("my_model", MyModelFactory)
"""

from typing import Dict, Type

from app.models.base_model import BaseMovementModel
from app.models.abstract_factory import (
    AbstractModelFactory,
    MODEL_FACTORIES,
)


class ModelFactory:
    """
    Registry that maps model names → AbstractModelFactory subclasses.

    All creation is delegated to the respective AbstractModelFactory so
    that model-family-specific construction logic stays encapsulated.
    """

    # Maps name → factory *class* (not instance, not model class directly)
    _registry: Dict[str, Type[AbstractModelFactory]] = {}

    @classmethod
    def register(cls, name: str, factory_class: Type[AbstractModelFactory]) -> None:
        """Register a model name with its AbstractModelFactory class."""
        cls._registry[name] = factory_class

    @classmethod
    def create(cls, name: str) -> BaseMovementModel:
        """
        Instantiate and return a model for the given *name*.

        The factory is created fresh each call so any per-construction
        state is isolated, then ``create_model()`` is called on it.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model '{name}'. Available: {available}")
        factory: AbstractModelFactory = cls._registry[name]()
        return factory.create_model()

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._registry.keys())


# ── Register all available model families ────────────────────────────
# Import factories from abstract_factory.py (already defined in MODEL_FACTORIES).
for _name, _factory_cls in MODEL_FACTORIES.items():
    ModelFactory.register(_name, _factory_cls)