import os
import numpy as np
import torch
from torch.utils.data import Dataset
from app.preprocessing.skeleton_preprocessor import SkeletonPreprocessor
from config import settings


class SkeletonDataset(Dataset):
    """
    Loads pre-extracted skeleton .npy files with labels from filenames.
    
    Expected filename format: SubjectID_DateID_GestureID_RepNum_CorrectLabel_Position.npy
    - CorrectLabel: 1 = correct, 2 = incorrect, 3 = poorly executed (skipped)
    - After label-1: 0 = correct, 1 = incorrect, 2 = poorly executed (skipped)
    
    Data format (IntelliRehab): each .npy has shape (num_frames, 75)
    where 75 = 25 Kinect joints × 3 coordinates (x, y, z).
    
    Expected structure:
        data_dir/
            101_18_0_1_1_stand.npy   # CorrectLabel=1 → label=0 (correct)
            101_18_0_2_2_stand.npy   # CorrectLabel=2 → label=1 (incorrect)
            ...
    """

    def __init__(self, data_dir: str, seq_length: int = None):
        self.preprocessor = SkeletonPreprocessor(seq_length)
        self.samples = []
        self.labels = []

        # Load all .txt files from the directory (simplified format: 75 CSV floats per frame)
        if not os.path.isdir(data_dir):
            print(f"ERROR: Directory does not exist: {data_dir}")
            return
        
        for fname in os.listdir(data_dir):
            if fname.endswith(".txt"):
                # Parse label from filename
                # Format: SubjectID_DateID_GestureID_RepNum_CorrectLabel_Position.txt
                try:
                    pieces = fname.replace('.txt', '').split('_')
                    if len(pieces) >= 5:
                        correct_label_raw = int(pieces[4])
                        label = correct_label_raw - 1  # 1→0 (correct), 2→1 (incorrect), 3→2 (poorly executed)
                        
                        # Skip poorly executed samples (label=2, i.e. CorrectLabel=3)
                        # matching the original intellirehab_helper behavior
                        if label == 2:
                            continue
                        
                        self.samples.append(os.path.join(data_dir, fname))
                        self.labels.append(label)
                except (ValueError, IndexError):
                    print(f"Warning: Cannot parse label from filename: {fname}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load simplified .txt: each line is 75 comma-separated floats (25 joints × 3 coords)
        raw = np.loadtxt(self.samples[idx], delimiter=',', dtype=np.float32)
        processed = self.preprocessor.process(raw)
        x = torch.tensor(processed, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
