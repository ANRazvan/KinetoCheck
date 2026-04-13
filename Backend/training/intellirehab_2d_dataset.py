"""
Dataset loader for IntelliRehab 2D.

Reads IntelliRehab Kinect files (25 joints x 3D), strips Z, and returns pure 2D
(25 joints x 2D) sequences processed with the IntelliRehab2D preprocessor.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from app.preprocessing.intellirehab_2d_preprocessor import IntelliRehab2DPreprocessor


class IntelliRehab2DDataset(Dataset):
    """Load IntelliRehab files and convert them to 2D (X, Y) for training.
    
    Includes 'Hard Negatives' from other exercises to teach the model that
    doing a Squat when you should do a Shoulder Flexion is WRONG.
    """

    def __init__(
        self,
        data_dir: str,
        exercise_id: int | None = None,
        seq_length: int | None = None,
        include_foreign_exercises: bool = True, # Enable this for robustness
    ):
        self.preprocessor = IntelliRehab2DPreprocessor(seq_length=seq_length)
        self.exercise_id = exercise_id
        self.samples: list[str] = []
        self.labels: list[int] = []
        self.exercise_ids: list[int] = []
        self.num_joints: int = 25
        self.keypoint_dim: int = 2

        if not os.path.isdir(data_dir):
            print(f"ERROR: Directory does not exist: {data_dir}")
            return

        # Gather positive samples first to determine balance
        target_positives = []
        target_negatives = []
        foreign_negatives = []

        for fname in os.listdir(data_dir):
            if not fname.endswith(".txt"):
                continue

            try:
                parts = fname.replace(".txt", "").split("_")
                if len(parts) < 5:
                    continue

                gesture_id = int(parts[2])
                label_str = parts[3] # Correct or Incorrect

                path = os.path.join(data_dir, fname)

                if self.exercise_id is not None:
                    if gesture_id == self.exercise_id:
                        # This is the target exercise
                        is_correct = 1 if label_str == "Correct" else 0
                        if is_correct:
                            target_positives.append((path, 1))
                        else:
                            target_negatives.append((path, 0))
                    elif include_foreign_exercises:
                        # This is a DIFFERENT exercise -> Wrong!
                        foreign_negatives.append((path, 0))
                else:
                    # Load everything (for multi-class? Not used here usually)
                    is_correct = 1 if label_str == "Correct" else 0
                    self.samples.append(path)
                    self.labels.append(is_correct)
                    self.exercise_ids.append(gesture_id)

            except ValueError:
                continue
        
        if self.exercise_id is not None:
            # Add all target samples
            self.samples.extend([s[0] for s in target_positives])
            self.labels.extend([s[1] for s in target_positives])
            self.samples.extend([s[0] for s in target_negatives])
            self.labels.extend([s[1] for s in target_negatives])
            
            # Balance Foreign Negatives
            # We don't want 90% of dataset to be foreign. 
            # Let's take count equal to total target samples (pos + neg)
            total_target = len(target_positives) + len(target_negatives)
            if include_foreign_exercises and foreign_negatives:
                import random
                # Deterministic shuffle for reproducibility
                random.seed(42) 
                random.shuffle(foreign_negatives)
                
                # Take subset
                limit = min(len(foreign_negatives), total_target)
                selected_foreign = foreign_negatives[:limit]
                
                self.samples.extend([s[0] for s in selected_foreign])
                self.labels.extend([s[1] for s in selected_foreign])

                print(f"Loaded for Ex {exercise_id}: {len(target_positives)} Pos, {len(target_negatives)} Neg, {len(selected_foreign)} Foreign Neg (Hard)")

                correct_label_raw = int(parts[4])
                label = correct_label_raw - 1  # 1->0, 2->1, 3->2

                if label == 2:
                    continue

                if exercise_id is not None and gesture_id != exercise_id:
                    continue

                self.samples.append(os.path.join(data_dir, fname))
                self.labels.append(label)
                self.exercise_ids.append(gesture_id)
            except (ValueError, IndexError):
                print(f"Warning: Cannot parse label from filename: {fname}")
                continue

    def label_distribution(self) -> dict[int, int]:
        return dict(Counter(self.labels))

    def exercise_distribution(self) -> dict[int, int]:
        return dict(Counter(self.exercise_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def _strip_z(self, raw: np.ndarray) -> np.ndarray:
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)

        if raw.shape[1] % 3 != 0:
            raise ValueError(
                f"Expected IntelliRehab 3D features divisible by 3, got {raw.shape[1]}."
            )

        joints = raw.shape[1] // 3
        reshaped = raw.reshape(raw.shape[0], joints, 3)
        xy = reshaped[:, :, :2]
        return xy.reshape(raw.shape[0], joints * 2)

    def __getitem__(self, idx: int):
        raw = np.loadtxt(self.samples[idx], delimiter=",", dtype=np.float32)
        raw_2d = self._strip_z(raw)
        processed = self.preprocessor.process(raw_2d)

        x = torch.tensor(processed, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
