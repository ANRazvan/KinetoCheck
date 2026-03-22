
"""
Abstract Factory for movement-analysis model families.

Pattern explained
-----------------
AbstractModelFactory   — declares the contract (create_model)
STGATModelFactory      — concrete factory that produces STGATModel instances
InceptionTimeModelFactory — stub for a future model family

Why Abstract Factory here?
  The factory not only creates the model object but can also supply
  family-specific supporting objects (e.g. a custom edge builder,
  a specific normalizer, etc.) without the client needing to know
  which concrete family is in use.

Usage::

    factory: AbstractModelFactory = STGATModelFactory()
    model = factory.create_model()
    model.build()
    model.load_weights(path)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.base_model import BaseMovementModel


# ── Abstract Factory ─────────────────────────────────────────────────

class AbstractModelFactory(ABC):
    """
    Abstract Factory for model families.

    Each concrete subclass represents one model *family* and is
    responsible for creating a fully configured ``BaseMovementModel``
    instance that fits that family's architecture.
    """

    @abstractmethod
    def create_model(self) -> BaseMovementModel:
        """
        Instantiate and return a concrete movement model.

        The returned object is *not* built yet; callers must invoke
        ``model.build()`` before using it for inference or training.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Concrete Factories ───────────────────────────────────────────────

class STGATModelFactory(AbstractModelFactory):
    """
    Factory for the Spatial-Temporal Graph Attention (ST-GAT) family.

    ST-GAT processes skeleton sequences as spatio-temporal graphs,
    applying Graph Attention Networks spatially (per frame) and
    temporal convolutions across the time axis.
    """

    def create_model(self) -> BaseMovementModel:
        from app.models.stgat_model import STGATModel
        return STGATModel()


class InceptionTimeModelFactory(AbstractModelFactory):
    """
    Factory for the InceptionTime family (stub — not yet implemented).

    InceptionTime applies multi-scale temporal convolutions directly
    on the flattened keypoint sequence and is a lighter alternative
    to graph-based models.
    """

    def create_model(self) -> BaseMovementModel:
        # Import lazily so that missing deps don't break the rest of the app.
        from app.models.inception_time_mode import InceptionTimeModel  # type: ignore
        return InceptionTimeModel()


# ── Factory registry mapping name → factory class ───────────────────
# Used by ModelFactory (model_factory.py) to look up factories by name.

MODEL_FACTORIES: dict[str, type[AbstractModelFactory]] = {
    "stgat": STGATModelFactory,
    "inception_time": InceptionTimeModelFactory,
}
