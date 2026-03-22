"""
Abstract Factory for training-pipeline component families.

Problem
-------
The training loop in ``training/train.py`` needs three coordinated objects:
  1. A **Dataset**       that knows how to load a specific data format
  2. A **Preprocessor**  that knows how to normalise that format's keypoints
  3. A **Model**         to train on the preprocessed data

When adding support for the **UI-PRMD** dataset we need a different Dataset
and Preprocessor, but the *training loop itself shouldn't change*.  The
Abstract Factory pattern solves this: swap the factory → get compatible,
coordinated components without touching the training code.

Structure
---------

AbstractTrainingFactory(ABC)          ← declares the contract
    ├─ IntelliRehabTrainingFactory      ← creates IntelliRehab components (from training.factories.intellirehab_factory)
    └─ UIPRMDTrainingFactory            ← creates UI-PRMD components (from training.factories.uiprmd_factory)

The factory also exposes ``dataset_name`` and ``num_joints`` descriptors
so the training loop can log which dataset family is in use.

Usage::

    # Switch dataset by swapping one line:
    from training.factories.intellirehab_factory import IntelliRehabTrainingFactory
    from training.factories.uiprmd_factory import UIPRMDTrainingFactory
    factory: AbstractTrainingFactory = IntelliRehabTrainingFactory()
    # factory: AbstractTrainingFactory = UIPRMDTrainingFactory()

    dataset     = factory.create_dataset(data_dir, exercise_id=3)
    preprocessor = factory.create_preprocessor()
    model        = factory.create_model("stgat")
    model.build(num_keypoints=factory.num_joints)
"""

from __future__ import annotations

from abc import ABC, abstractmethod



from torch.utils.data import Dataset
from app.models.base_model import BaseMovementModel

# ── Abstract Factory ─────────────────────────────────────────────────

class AbstractTrainingFactory(ABC):
    """
    Abstract Factory that creates a *family* of compatible training components.

    All three factory methods should be called together; mixing components
    from different concrete factories is not supported.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Human-readable name of the dataset this factory targets."""

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of skeleton joints this dataset/preprocessor produces."""

    @abstractmethod
    def create_dataset(
        self,
        data_dir: str,
        exercise_id: int | None = None,
        seq_length: int | None = None,
    ) -> Dataset:
        """
        Create a dataset for the given *data_dir* (and optional exercise filter).

        Args:
            data_dir:    Path to the directory that contains the raw data files.
            exercise_id: Filter to a single exercise (None = all exercises).
            seq_length:  Override the target sequence length (None = from config).
        """

    @abstractmethod
    def create_preprocessor(self, seq_length: int | None = None):
        """
        Create the matching preprocessor for this dataset family.

        Args:
            seq_length: Override the target sequence length (None = from config).

        Returns:
            An object with a ``process(keypoints) -> np.ndarray`` method.
        """

    @abstractmethod
    def create_model(self, model_name: str) -> BaseMovementModel:
        """
        Create an *unbuilt* model for the given *model_name*.

        The caller must invoke ``model.build()`` before using it.

        Args:
            model_name: Registry key (e.g. ``"stgat"``).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset_name!r})"


# ── Concrete Factory: IntelliRehab ───────────────────────────────────


# Import concrete factories from new files
from training.factories.intellirehab_factory import IntelliRehabTrainingFactory
from training.factories.uiprmd_factory import UIPRMDTrainingFactory
from training.factories.uiprmd_angles_factory import UIPRMDAnglesTrainingFactory


# ── Registry ─────────────────────────────────────────────────────────

TRAINING_FACTORIES: dict[str, type[AbstractTrainingFactory] | tuple[type[AbstractTrainingFactory], dict]] = {
    "intellirehab": IntelliRehabTrainingFactory,
    "uiprmd": UIPRMDTrainingFactory,
    "uiprmd_angles_vicon": (UIPRMDAnglesTrainingFactory, {"modality": "vicon"}),
    "uiprmd_angles_kinect": (UIPRMDAnglesTrainingFactory, {"modality": "kinect"}),
}


def get_training_factory(dataset: str = "intellirehab", **kwargs) -> AbstractTrainingFactory:
    """
    Return an ``AbstractTrainingFactory`` instance for the given *dataset* name.

    Args:
        dataset: ``"intellirehab"`` (default), ``"uiprmd"``, 
                 ``"uiprmd_angles_vicon"``, or ``"uiprmd_angles_kinect"``.
        **kwargs: Additional arguments passed to factory constructor.

    Raises:
        ValueError: If *dataset* is not a registered factory name.
    """
    key = dataset.lower()
    if key not in TRAINING_FACTORIES:
        available = list(TRAINING_FACTORIES.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")
    
    factory_spec = TRAINING_FACTORIES[key]
    
    # Handle both simple factories and parameterized factories
    if isinstance(factory_spec, tuple):
        factory_class, default_kwargs = factory_spec
        # Merge default kwargs with provided kwargs (provided takes precedence)
        merged_kwargs = {**default_kwargs, **kwargs}
        return factory_class(**merged_kwargs)
    else:
        return factory_spec()
