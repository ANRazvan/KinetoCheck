from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import Config.config as cfg
from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor


class UIPRMDLoader:
    def __init__(self, data_root: Path | None = None):
        self.data_root: Path = Path(data_root) if data_root is not None else cfg.UIPRMD_PATH
        self.sources_vicon = [
            (self.data_root / "Segmented Movements" / "Vicon" / "Positions", 0, "segmented_correct", "*.txt"),
            (self.data_root / "Incorrect Segmented Movements" / "Vicon" / "Positions", 1, "segmented_incorrect", "*.txt"),
            # (self.data_root / "Movements" / "Vicon" / "Positions", 0, "full_correct", "*.txt"),
            # (self.data_root / "Incorrect Movements" / "Vicon" / "Positions", 1, "full_incorrect", "*.txt"),
        ]

    def _parse_filename(self, name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        # Segmented: m01_s02_e03_positions(.txt/_inc.txt)
        m = re.match(r"m(\d+)_s(\d+)_e(\d+)_", name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)) - 1, int(m.group(2)), int(m.group(3))

        # Non-segmented: m01_s02_positions(.txt/_inc.txt)
        m = re.match(r"m(\d+)_s(\d+)_", name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)) - 1, int(m.group(2)), 1

        return None, None, None

    def _load_vicon_txt(self, file_path: Path) -> Optional[np.ndarray]:
        try:
            arr = np.loadtxt(file_path, delimiter=",", ndmin=2)
        except Exception:
            print(f"Error loading {file_path}")
            return None

        # Expect at least 117 columns; keep first 117.
        if arr.ndim != 2 or arr.shape[1] < 117:
            print(f"Warning: {file_path} has unexpected shape {arr.shape}, skipping.")
            return None

        coords = arr[:, :117].reshape(-1, 39, 3)
        return coords.astype(np.float32)

    def load_vicon_data(self, exercise_id: Optional[int] = None) -> List[Dict]:
        records: List[Dict] = []

        for source_dir, label, subset, pattern in self.sources_vicon:
            source_dir = Path(source_dir)
            if not source_dir.exists():
                continue

            for fpath in sorted(source_dir.glob(pattern)):
                eid, sid, rid = self._parse_filename(fpath.name)
                if eid is None:
                    continue
                if exercise_id is not None and eid != exercise_id:
                    continue

                seq = self._load_vicon_txt(fpath)
                if seq is None:
                    continue

                records.append(
                    {
                        "file": str(fpath),
                        "sequence": seq,                 # shape: (frames, 39, 3)
                        "label": int(label),             # 0 correct, 1 incorrect
                        "subset": str(subset),           # segmented_correct, etc.
                        "exercise_id": int(eid),         # 0-based
                        "subject_id": int(sid),
                        "rep": int(rid),
                        "is_incorrect_file": fpath.name.lower().endswith("_inc.txt"),
                    }
                )

        return records


if __name__ == "__main__":
    loader = UIPRMDLoader()
    data = loader.load_vicon_data()
    print(f"Loaded samples: {len(data)}")
    if data:
        print(f"First sample shape: {data[0]['sequence'].shape}")
        print(f"First sample label/subset: {data[0]['label']} / {data[0]['subset']}")

    preprocessor = UIPRMDPreprocessor()
    if data:
        aligned = preprocessor.align_vicon_to_mediapipe(data[0]["sequence"])
        print(f"Aligned shape: {aligned.shape}")
