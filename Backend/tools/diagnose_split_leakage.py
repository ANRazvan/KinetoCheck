import hashlib
from collections import Counter
from pathlib import Path
import sys

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from temporal_pyramid_stgat.preprocessing.mediapipe_uiprmd_loader import MediaPipeUIsprmdLoader


def main() -> None:
    loader = MediaPipeUIsprmdLoader(r"D:\\Programming\\KinetoCheck\\Datasets\\UIPRMD")
    coords, angles, labels, meta = loader.load_with_angles(0)

    n = len(labels)
    print(f"N={n}")

    uniq_labels, label_counts = np.unique(labels, return_counts=True)
    print("label_dist=", {int(k): int(v) for k, v in zip(uniq_labels, label_counts)})

    subjects = np.array([m.get("subject_id", -1) for m in meta], dtype=np.int64)
    uniq_subj, subj_counts = np.unique(subjects, return_counts=True)
    print(f"subjects={len(uniq_subj)} min_per_subject={int(subj_counts.min())} max_per_subject={int(subj_counts.max())}")

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_subj = set(subjects[train_idx].tolist())
    val_subj = set(subjects[val_idx].tolist())
    overlap = train_subj.intersection(val_subj)
    print(f"random_split_subject_overlap={len(overlap)} / {len(val_subj)} val-subjects")

    hashes = [hashlib.md5(coords[i].tobytes()).hexdigest() for i in range(n)]
    counter = Counter(hashes)
    dup_groups = sum(1 for v in counter.values() if v > 1)
    dup_items = sum(v for v in counter.values() if v > 1)
    print(f"exact_duplicate_groups={dup_groups} exact_duplicate_items={dup_items}")


if __name__ == "__main__":
    main()
