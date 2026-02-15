from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseMovementModel(ABC):
    """
    Abstract base class for all movement correctness models.
    Implements the Strategy Pattern — any model that inherits this
    can be swapped in without changing the rest of the codebase.
    """

    @abstractmethod
    def build(self, **kwargs) -> None:
        """Build/initialize the model architecture."""
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Load pretrained weights from disk."""
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Save model weights to disk."""
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on preprocessed input.

        Args:
            input_data: Preprocessed skeleton/pose data.
                        Shape depends on the concrete model.

        Returns:
            Dictionary with at least:
                - "label": str (e.g., "correct", "incorrect")
                - "confidence": float
                - "details": optional dict with extra info
        """
        pass

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single training step.

        Returns:
            Dictionary with loss and any metrics.
        """
        pass

    @abstractmethod
    def eval_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single evaluation step.

        Returns:
            Dictionary with loss and any metrics.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the model (name, params, etc.)."""
        pass