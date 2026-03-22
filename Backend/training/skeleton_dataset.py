import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from app.preprocessing.skeleton_preprocessor import SkeletonPreprocessor
from config import settings


class SkeletonDataset(Dataset):
    """
    Loads pre-extracted skeleton .txt files with labels from filenames.

    Filename format:
        SubjectID_DateID_GestureID_RepNum_CorrectLabel_Position.txt
        e.g. ``101_18_3_1_1_stand.txt``
              ─── ── ─ ─ ─ ─────
              │   │  │ │ │ └─ Position
              │   │  │ │ └── CorrectLabel (1=correct, 2=incorrect, 3=poorly)
              │   │  │ └─── RepNum
              │   │  └──── GestureID  ← exercise type (0-8)
              │   └────── DateID
              └────────── SubjectID

    Args:
        data_dir:    Path to the folder with ``.txt`` skeleton files.
        exercise_id: If given, only load samples for that exercise (GestureID).
                     ``None`` loads *all* exercises (legacy behaviour).
        seq_length:  Override the target sequence length (default from config).
    """

    def __init__(
        self,
        data_dir: str,
        exercise_id: int | None = None,
        seq_length: int | None = None,
    ):
        self.preprocessor = SkeletonPreprocessor(seq_length)
        self.exercise_id = exercise_id
        self.samples: list[str] = []
        self.labels: list[int] = []
        self.exercise_ids: list[int] = []

        if not os.path.isdir(data_dir):
            print(f"ERROR: Directory does not exist: {data_dir}")
            return

        for fname in os.listdir(data_dir):
            if not fname.endswith(".txt"):
                continue
            try:
                pieces = fname.replace(".txt", "").split("_")
                if len(pieces) < 5:
                    continue

                gesture_id = int(pieces[2])
                correct_label_raw = int(pieces[4])
                label = correct_label_raw - 1  # 1→0, 2→1, 3→2

                # Skip poorly-executed samples (CorrectLabel == 3 → label == 2)
                if label == 2:
                    continue

                # Filter by exercise if requested
                if exercise_id is not None and gesture_id != exercise_id:
                    continue

                self.samples.append(os.path.join(data_dir, fname))
                self.labels.append(label)
                self.exercise_ids.append(gesture_id)
            except (ValueError, IndexError):
                print(f"Warning: Cannot parse label from filename: {fname}")
                continue

    # ── helpers ───────────────────────────────────────────────────────

    def label_distribution(self) -> dict[int, int]:
        """Return {label: count} for a quick sanity check."""
        return dict(Counter(self.labels))

    def exercise_distribution(self) -> dict[int, int]:
        """Return {exercise_id: count}."""
        return dict(Counter(self.exercise_ids))

    # ── Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw = np.loadtxt(self.samples[idx], delimiter=",", dtype=np.float32)
        processed = self.preprocessor.process(raw)
        x = torch.tensor(processed, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
