"""
UI-PRMD Dataset Loader for Temporal Pyramid STGAT.

Loads skeleton data from UI-PRMD official layout or legacy CSV format.
Infers joint dimensionality from file columns and preserves all available
features (e.g., 39 joints -> 117 values) for end-to-end consistency.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import re


class UIPRMDLoader:
    """
    Load UI-PRMD skeleton sequences with binary correctness labels.
    
    File structure expected:
    - Segmented Movements/Vicon/Positions/*.txt (correct)
    - Incorrect Segmented Movements/Vicon/Positions/*.txt (incorrect)
    OR
    - correct/*.csv (legacy)
    - incorrect/*.csv (legacy)
    
    File naming: E<exercise_id>_S<subject>_R<rep>_C.csv (correct)
                 E<exercise_id>_S<subject>_R<rep>_I.csv (incorrect)
    """
    
    # Reference 17-joint names used by some pipelines.
    JOINT_NAMES_17 = [
        "Pelvis", "L5_Lower_Spine", "L3_Mid_Spine", "T12",
        "T8_Upper_Spine", "Neck", "Head",
        "Right_Shoulder", "Left_Shoulder",
        "Right_Arm", "Right_Forearm", "Right_Hand",
        "Left_Arm", "Left_Forearm", "Left_Hand",
        "Right_Leg", "Left_Leg"
    ]
    
    JOINT_DIM = 3  # X, Y, Z
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: Path to UI-PRMD root directory
        """
        self.data_root = Path(data_root)
        self.samples: List[str] = []
        self.labels: List[int] = []
        self.exercise_ids: List[int] = []
        self.subject_ids: List[int] = []
        self.metadata: List[Dict] = []
        self.num_joints: int = None
        self.target_features: int = None
        
    def load_all(self, exercise_id: int = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load all sequences.
        
        Args:
            exercise_id: Filter by exercise (None = all exercises)
            
        Returns:
            data: (N, T, J, 3) - N sequences, T timesteps, J inferred joints
            labels: (N,) - 0=correct, 1=incorrect
            metadata: List of dicts with sequence info
        """
        self._scan_files(exercise_id)
        data_list = []
        
        for file_path in self.samples:
            seq = self._load_sequence(file_path)
            if seq is not None:
                data_list.append(seq)
        
        if not data_list:
            raise ValueError(f"No sequences found in {self.data_root}")
        
        # Pad/truncate to common length
        data = self._normalize_lengths(data_list)
        labels = np.array(self.labels, dtype=np.int64)
        
        return data, labels, self.metadata
    
    def _scan_files(self, exercise_id: int = None):
        """Scan directory for data files."""
        self.samples = []
        self.labels = []
        self.exercise_ids = []
        self.subject_ids = []
        self.metadata = []

        sources = [
            # Official UI-PRMD segmented
            (self.data_root / "Segmented Movements" / "Vicon" / "Positions", 0),
            (self.data_root / "Incorrect Segmented Movements" / "Vicon" / "Positions", 1),
            # Non-segmented
            (self.data_root / "Movements" / "Vicon" / "Positions", 0),
            (self.data_root / "Incorrect Movements" / "Vicon" / "Positions", 1),
            # Legacy CSV format
            (self.data_root / "correct", 0),
            (self.data_root / "incorrect", 1),
        ]
        
        for source_dir, label in sources:
            if not source_dir.exists():
                continue
            
            pattern = "*.txt" if "Vicon" in str(source_dir) else "*.csv"
            for fpath in source_dir.glob(pattern):
                eid, sid, rid = self._parse_filename(fpath.name)
                
                if eid is None:
                    continue
                if exercise_id is not None and eid != exercise_id:
                    continue
                
                self.samples.append(str(fpath))
                self.labels.append(label)
                self.exercise_ids.append(eid)
                self.subject_ids.append(sid if sid is not None else -1)
                
                self.metadata.append({
                    "file": str(fpath),
                    "exercise_id": eid,
                    "subject_id": sid,
                    "rep": rid,
                    "label": label,
                    "label_name": "correct" if label == 0 else "incorrect"
                })
    
    def _parse_filename(self, filename: str) -> Tuple[int, int, int]:
        """
        Parse E<ex>_S<sub>_R<rep>_[C|I] format.
        Returns: (exercise_id, subject_id, rep_id)
        """
        # Legacy format: E<exercise>_S<subject>_R<rep>
        match = re.match(r"E(\d+)_S(\d+)_R(\d+)", filename, flags=re.IGNORECASE)
        if match:
            # Keep 0-based exercise indexing for CLI --exercise argument
            return int(match.group(1)) - 1, int(match.group(2)), int(match.group(3))

        # UI-PRMD segmented format: m<movement>_s<subject>_e<rep>_positions(.txt/_inc.txt)
        match = re.match(r"m(\d+)_s(\d+)_e(\d+)_", filename, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)) - 1, int(match.group(2)), int(match.group(3))

        # UI-PRMD non-segmented format: m<movement>_s<subject>_positions(.txt/_inc.txt)
        match = re.match(r"m(\d+)_s(\d+)_", filename, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)) - 1, int(match.group(2)), 1

        return None, None, None
    
    def _load_sequence(self, filepath: str) -> np.ndarray:
        """
        Load a single skeleton sequence from file.
        
        Returns:
            array of shape (T, 17, 3) or None if loading failed
        """
        try:
            # UI-PRMD txt files are often comma-delimited; try whitespace first, then comma.
            lower_path = str(filepath).lower()
            if lower_path.endswith(".csv"):
                data = np.loadtxt(filepath, delimiter=",")
            else:
                try:
                    data = np.loadtxt(filepath)
                except ValueError:
                    data = np.loadtxt(filepath, delimiter=",")
            
            # Ensure (T, F) shape
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Infer feature dimensionality from first valid file and keep it fixed.
            if data.shape[1] % self.JOINT_DIM != 0:
                return None

            if self.target_features is None:
                self.target_features = data.shape[1]
                self.num_joints = self.target_features // self.JOINT_DIM

            # Enforce consistent dimensionality across files.
            if data.shape[1] < self.target_features:
                return None
            if data.shape[1] > self.target_features:
                data = data[:, :self.target_features]
            
            # Reshape to (T, J, 3)
            T = data.shape[0]
            seq = data.reshape(T, self.num_joints, self.JOINT_DIM)
            
            return seq.astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            return None
    
    def _normalize_lengths(self, sequences: List[np.ndarray], target_len: int = 240) -> np.ndarray:
        """
        Pad or truncate sequences to fixed length.
        
        Args:
            sequences: List of (T, J, 3) arrays
            target_len: Target sequence length
            
        Returns:
            (N, target_len, J, 3) array
        """
        normalized = []
        
        for seq in sequences:
            T = seq.shape[0]
            
            if T >= target_len:
                # Truncate from center
                start = (T - target_len) // 2
                normalized.append(seq[start:start + target_len])
            else:
                # Pad with zeros
                padded = np.zeros((target_len, self.num_joints, self.JOINT_DIM),
                                 dtype=np.float32)
                padded[:T] = seq
                normalized.append(padded)
        
        return np.stack(normalized)


class DatasetStatistics:
    """Compute statistics on UI-PRMD dataset."""
    
    @staticmethod
    def compute_stats(data: np.ndarray) -> Dict:
        """
        Compute mean and std per joint and axis.
        
        Args:
            data: (N, T, J, 3) array
            
        Returns:
            Dict with mean, std, min, max
        """
        # Flatten to (N*T*J, 3)
        flat = data.reshape(-1, 3)
        
        return {
            "mean": np.mean(flat, axis=0),
            "std": np.std(flat, axis=0),
            "min": np.min(flat, axis=0),
            "max": np.max(flat, axis=0),
        }
    
    @staticmethod
    def label_distribution(labels: np.ndarray) -> Dict:
        """Count label distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        return {
            int(u): int(c) for u, c in zip(unique, counts)
        }
