"""
Dataset loader for UI-PRMD angle data (Vicon or Kinect modalities).

Supports loading pre-computed joint angles from Angles/ directories.
These angles are derived from 17 anatomical landmarks and represent
joint rotations in degrees.

Modalities:
  - Vicon Angles: ~130-150 angles (official high-quality)
  - Kinect Angles: ~60-70 angles (realistic, sensor-like noise)

Directory structure (same for both modalities):
  <data_dir>/
    Segmented Movements/
      {Vicon|Kinect}/Angles/*.txt
    Incorrect Segmented Movements/
      {Vicon|Kinect}/Angles/*.txt
    Movements/
      {Vicon|Kinect}/Angles/*.txt
    Incorrect Movements/
      {Vicon|Kinect}/Angles/*.txt
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from app.preprocessing.uiprmd_angles_preprocessor import UIPRMDAnglesPreprocessor


class UIPRMDAnglesDataset(Dataset):
    """
    Load UI-PRMD angle files (.txt) with binary correctness labels.

    Args:
        data_dir: Root UI-PRMD directory (where Segmented Movements/ exists)
        modality: 'vicon' or 'kinect' — which capture system's angles to load
        exercise_id: Filter by exercise (0-9). None = load all
        seq_length: Override target sequence length
        use_segmented: If True, load only Segmented Movements/.
                      If False, mix both Segmented and non-Segmented.
    """

    def __init__(
        self,
        data_dir: str,
        modality: str = "vicon",
        exercise_id: int | None = None,
        seq_length: int | None = None,
        use_segmented: bool = True,
        feature_dim: int | None = None,
        swap_labels: bool = False,
    ):
        if modality.lower() not in ("vicon", "kinect"):
            raise ValueError(f"Unknown modality: {modality}. Choose 'vicon' or 'kinect'.")
        
        self.swap_labels = swap_labels

        self.modality = modality.lower()
        self.preprocessor = UIPRMDAnglesPreprocessor(seq_length, target_dim=feature_dim)
        self.exercise_id = exercise_id
        self.samples: list[str] = []
        self.labels: list[int] = []
        self.exercise_ids: list[int] = []
        self.num_angles: int | None = None

        root = Path(data_dir)
        if not root.is_dir():
            print(f"ERROR: UI-PRMD data directory does not exist: {data_dir}")
            return

        # Build source paths for angle files
        sources: list[tuple[Path, int]] = []

        if use_segmented:
            # Load from Segmented Movements/ only
            sources.extend([
                (root / "Segmented Movements" / self.modality.capitalize() / "Angles", 0),
                (root / "Incorrect Segmented Movements" / self.modality.capitalize() / "Angles", 1),
            ])
        else:
            # Load from both Segmented and non-Segmented
            sources.extend([
                (root / "Segmented Movements" / self.modality.capitalize() / "Angles", 0),
                (root / "Incorrect Segmented Movements" / self.modality.capitalize() / "Angles", 1),
                (root / "Movements" / self.modality.capitalize() / "Angles", 0),
                (root / "Incorrect Movements" / self.modality.capitalize() / "Angles", 1),
            ])

        # Load file paths
        for source_dir, label_int in sources:
            if not source_dir.is_dir():
                continue
            for sample_file in source_dir.glob("*.txt"):
                eid = self._parse_exercise_id(sample_file.name)
                if eid is None:
                    continue
                if exercise_id is not None and eid != exercise_id:
                    continue
                self.samples.append(str(sample_file))
                self.labels.append(label_int)
                self.exercise_ids.append(eid)

        # Infer number of angles from first sample
        if self.samples:
            try:
                raw = np.loadtxt(self.samples[0], dtype=np.float32)
                if raw.ndim == 1:
                    raw = np.expand_dims(raw, axis=0)
                self.num_angles = raw.shape[1]
            except Exception:
                pass

    def _parse_exercise_id(self, filename: str) -> int | None:
        """Extract exercise_id from filename (e01-e10 or m01-m10)."""
        try:
            stem = Path(filename).stem

            # Prefer 'e' marker (e.g., m01_s01_e03_angles)
            m = re.search(r"(?:^|[_-])e(\d+)(?:[_-]|$)", stem, flags=re.IGNORECASE)
            if m:
                value = int(m.group(1))
                return value - 1 if value > 0 else value

            # Fallback to 'm' marker (e.g., m07_s03_angles)
            m = re.search(r"(?:^|[_-])m(\d+)(?:[_-]|$)", stem, flags=re.IGNORECASE)
            if m:
                value = int(m.group(1))
                return value - 1 if value > 0 else value

            return None
        except (IndexError, ValueError, TypeError):
            return None

    def _load_raw_sample(self, sample_path: str) -> np.ndarray:
        """Load angle file with delimiter fallback."""
        try:
            raw = np.loadtxt(sample_path, delimiter=",", dtype=np.float32)
        except ValueError:
            # Fallback to whitespace-delimited (scientific notation)
            raw = np.loadtxt(sample_path, dtype=np.float32)

        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        return raw

    def label_distribution(self) -> dict[int, int]:
        """Return {label: count}."""
        return dict(Counter(self.labels))

    def exercise_distribution(self) -> dict[int, int]:
        """Return {exercise_id: count}."""
        return dict(Counter(self.exercise_ids))

    # ── Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw = self._load_raw_sample(self.samples[idx])
        processed = self.preprocessor.process(raw)
        x = torch.tensor(processed, dtype=torch.float32)
        label = self.labels[idx]
        if self.swap_labels:
            label = 1 - label  # Flip 0↔1 for diagnostic testing
        y = torch.tensor(label, dtype=torch.long)
        return x, y
