"""
Dataset loader for the UI-PRMD (University of Idaho – Physical Rehabilitation
Movement Dataset).

UI-PRMD file layout
-------------------
Files are stored as CSV/TXT with **no header row**.
Each row is one frame and may contain either:
- 117 values = 39 Vicon markers × 3 coordinates
- 51 values  = 17 MediaPipe-compatible joints × 3 coordinates

Expected directory structure::

    <data_dir>/
        correct/
            E<exercise_id>_S<subject>_R<rep>_C.csv
            ...
        incorrect/
            E<exercise_id>_S<subject>_R<rep>_I.csv
            ...

Label encoding  →  0 = correct,  1 = incorrect   (same as IntelliRehab)

This class exposes the same ``__len__`` / ``__getitem__`` / ``label_distribution``
interface as ``SkeletonDataset`` so it is a drop-in product for
``AbstractTrainingFactory.create_dataset()``.

If your UI-PRMD copy uses a different naming scheme, subclass this and
override ``_parse_label_from_path()``.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from app.preprocessing.uiprmd_preprocessor import UIPRMDPreprocessor


class UIPRMDDataset(Dataset):
    """
    Load UI-PRMD skeleton CSV files with binary correctness labels.

    Args:
                data_dir:    Root UI-PRMD directory. Supports both:
                                         - legacy ``correct/`` and ``incorrect/`` CSV layout
                                         - official UI-PRMD Vicon positions layout under
                                             ``Segmented Movements/`` and ``Incorrect Segmented Movements/``
                                             (and non-segmented ``Movements/`` variants).
        exercise_id: Only load files whose name starts with
                     ``E<exercise_id>_`` (0-indexed).  Pass ``None``
                     to load *all* exercises.
        seq_length:  Override the target sequence length (default from
                     config via ``UIPRMDPreprocessor``).
    """

    def __init__(
        self,
        data_dir: str,
        exercise_id: int | None = None,
        seq_length: int | None = None,
        keypoint_dim: int = 3,
    ):
        self.keypoint_dim = keypoint_dim
        self.preprocessor = UIPRMDPreprocessor(seq_length=seq_length, keypoint_dim=keypoint_dim)
        self.exercise_id = exercise_id
        self.samples: list[str] = []
        self.labels: list[int] = []
        self.exercise_ids: list[int] = []
        # Model is always trained on 17 aligned landmarks.
        self.num_joints: int = 17

        root = Path(data_dir)
        if not root.is_dir():
            print(f"ERROR: UI-PRMD data directory does not exist: {data_dir}")
            return

        # Supported layouts (first existing paths are used; duplicates are okay to include)
        sources: list[tuple[Path, int, str]] = [
            # Legacy custom layout
            (root / "correct", 0, "*.csv"),
            (root / "incorrect", 1, "*.csv"),
            # Official UI-PRMD segmented files (recommended)
            (root / "Segmented Movements" / "Vicon" / "Positions", 0, "*.txt"),
            (root / "Incorrect Segmented Movements" / "Vicon" / "Positions", 1, "*.txt"),
            # Official UI-PRMD non-segmented files
            (root / "Movements" / "Vicon" / "Positions", 0, "*.txt"),
            (root / "Incorrect Movements" / "Vicon" / "Positions", 1, "*.txt"),
        ]

        for source_dir, label_int, pattern in sources:
            if not source_dir.is_dir():
                continue
            for sample_file in source_dir.glob(pattern):
                eid = self._parse_exercise_id(sample_file.name)
                if eid is None:
                    continue
                if exercise_id is not None and eid != exercise_id:
                    continue
                self.samples.append(str(sample_file))
                self.labels.append(label_int)
                self.exercise_ids.append(eid)

        # Keep num_joints fixed at 17 because samples are aligned to MediaPipe 17.

    def _infer_num_joints(self, sample_path: str) -> int | None:
        """Infer number of joints from first frame feature count."""
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                return None
            # UI-PRMD files can be comma-delimited or whitespace-delimited.
            feature_count = len([p for p in re.split(r"[\s,]+", first_line) if p])
            if feature_count % 3 == 0:
                return feature_count // 3
            if feature_count % self.keypoint_dim == 0:
                return feature_count // self.keypoint_dim
            return None
        except OSError:
            return None

    def _load_raw_sample(self, sample_path: str) -> np.ndarray:
        """Load one sample with delimiter fallback (comma -> whitespace)."""
        # First try comma-delimited files.
        try:
            raw = np.loadtxt(sample_path, delimiter=",", dtype=np.float32)
            if raw.ndim == 1:
                raw = np.expand_dims(raw, axis=0)
        except ValueError:
            # Fallback for space-delimited scientific-notation text files.
            raw = np.loadtxt(sample_path, dtype=np.float32)
            if raw.ndim == 1:
                raw = np.expand_dims(raw, axis=0)
            
        return raw

    # ── helpers ─────────────────────────────────────────────────────

    def _parse_exercise_id(self, filename: str) -> int | None:
        """
        Extract exercise_id from a UI-PRMD filename.

        Supported patterns include:
            - ``E<id>_...`` (legacy/custom exports)
            - ``..._e<id>_...`` (segmented official files)
            - ``m<id>_...`` (official movement files)

        Returns ``None`` if the filename cannot be parsed.
        """
        try:
            stem = Path(filename).stem

            # Prefer explicit segmented marker first (e.g. m01_s01_e03_positions)
            m = re.search(r"(?:^|[_-])e(\d+)(?:[_-]|$)", stem, flags=re.IGNORECASE)
            if m:
                value = int(m.group(1))
                return value - 1 if value > 0 else value

            # Fallback to movement marker (e.g. m07_s03_positions)
            m = re.search(r"(?:^|[_-])m(\d+)(?:[_-]|$)", stem, flags=re.IGNORECASE)
            if m:
                value = int(m.group(1))
                return value - 1 if value > 0 else value

            # Legacy pattern support (e.g. E3_S01_R2_C)
            if stem.startswith("E") and len(stem) > 1 and stem[1].isdigit():
                value = int(stem.split("_")[0][1:])
                return value

            return None
        except (IndexError, ValueError, TypeError):
            return None

    def _parse_label_from_path(self, path: str) -> int:
        """Return 0 (correct) or 1 (incorrect) based on the parent folder name."""
        parent = Path(path).parent.name.lower()
        return 0 if parent == "correct" else 1

    def label_distribution(self) -> dict[int, int]:
        """Return ``{label: count}`` for a quick sanity check."""
        return dict(Counter(self.labels))

    def exercise_distribution(self) -> dict[int, int]:
        """Return ``{exercise_id: count}``."""
        return dict(Counter(self.exercise_ids))

    # ── Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw = self._load_raw_sample(self.samples[idx])

        # Align Vicon/legacy positions to the 17 MediaPipe-compatible landmarks.
        aligned_3d = self.preprocessor.align_vicon_to_mediapipe(raw)
        if self.keypoint_dim == 2:
            aligned = aligned_3d[:, :, :2]
        else:
            aligned = aligned_3d

        processed = self.preprocessor.process(aligned)
        x = torch.tensor(processed, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
