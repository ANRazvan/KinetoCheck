# MediaPipe 33-Joint Retraining Guide

## Overview

This guide explains how to retrain the Temporal Pyramid STGAT model using **MediaPipe 33-landmark representation** instead of the original Vicon 39-joint format. This retraining is critical for resolving the domain mismatch between training and video-based inference.

### Why Retrain with MediaPipe?

**Problem**: The original model was trained on 39-joint Vicon lab data, but video inference uses:
- **YOLO**: 17 COCO joints
- **MediaPipe**: 33 landmarks

This massive domain mismatch causes the model to confidently predict "Incorrect" even for correct exercises performed in videos.

**Solution**: Retrain on the same 33-landmark representation used at inference, ensuring consistent feature space.

---

## Pipeline Architecture

### 1. Data Transformation

```
Vicon 39-joint sequences (UI-PRMD) 
    ↓
[MediaPipeMapper] 
    ↓
MediaPipe 33-landmark sequences
    ↓
[MediaPipeAngleCalculator]
    ↓
12 anatomical angles (extracted from 33 landmarks)
```

### 2. Key Components

#### `mediapipe_mapper.py`
- **Purpose**: Convert Vicon 39 joints → MediaPipe 33 landmarks
- **Method**: Heuristic joint mapping (e.g., Vicon 0 → MP 0, Vicon 3 → MP 11)
- **Output**: (T, 33, 3) sequences

#### `mediapipe_angle_calculator.py`
- **Purpose**: Extract 12 anatomical angles from 33-landmark sequences
- **Angles**: Shoulder-elbow-wrist, hip-knee-ankle, torso (left/right variations)
- **Output**: (T, 12) angle sequences, standardized across dataset

#### `mediapipe_uiprmd_loader.py`
- **Purpose**: Orchestrate loading with automatic Vicon→MediaPipe conversion
- **Returns**: (N, T, 33, 3) coords + (N, T, 12) angles + labels

#### `config.py` (updated)
- **New Method**: `PyramidSTGATConfig.for_uiprmd_mediapipe_33joint(exercise_id)`
- **Sets**:
  - `num_joints = 33`
  - `in_channels_coord = 99` (3 × 33)
  - `in_channels_angle = 12`
  - `checkpoint_name = "pyramid_stgat_mediapipe33_best.pt"`

#### `train_triplet.py` (updated)
- **Auto-Detection**: Checks `config.num_joints == 33` to enable MediaPipe mode
- **Data Loading**: Uses `MediaPipeUIsprmdLoader` instead of `UIPRMDLoader`
- **Feature Extraction**: Skips redundant angle extraction (already in loader)

---

## Quick Start

### 1. Validate Pipeline

Before training, verify all components work:

```bash
cd Backend
python -m temporal_pyramid_stgat.validation.validate_mediapipe_pipeline
```

Expected output:
```
Mapper: ✓ PASSED
Angle Calculator: ✓ PASSED
Configuration: ✓ PASSED
Data Loader: ✓ PASSED
✓ All tests passed! Ready for training.
```

### 2. Train on All Exercises

```bash
cd Backend
python temporal_pyramid_stgat/training/train_mediapipe.py
```

Options:
- `--exercise 0`: Train single exercise
- `--loso-subject 1`: Use Leave-One-Subject-Out validation with subject ID 1
- `--validation random`: Force random split (default)

### 3. Training Examples

#### Train all exercises with random validation:
```bash
python temporal_pyramid_stgat/training/train_mediapipe.py
```

#### Train exercise 0 with LOSO (subject 1):
```bash
python temporal_pyramid_stgat/training/train_mediapipe.py --exercise 0 --loso-subject 1
```

#### Train all exercises with LOSO (subject 3):
```bash
python temporal_pyramid_stgat/training/train_mediapipe.py --loso-subject 3
```

---

## Model Configuration

### Default MediaPipe Config

```python
from temporal_pyramid_stgat.config import PyramidSTGATConfig

config = PyramidSTGATConfig.for_uiprmd_mediapipe_33joint()

# Resulting config:
# - num_joints: 33
# - in_channels_coord: 99 (3 × 33)
# - in_channels_angle: 12
# - batch_size: 16
# - learning_rate: 0.001
# - epochs: 100
# - early_stopping_patience: 20
```

### Customization

```python
# Single exercise training
config = PyramidSTGATConfig.for_uiprmd_mediapipe_33joint(exercise_id=0)
```

---

## Training Output

After successful training, check:

```
temporal_pyramid_stgat/weights/
├── pyramid_stgat_mediapipe33_best.pt          # Best loss checkpoint
├── pyramid_stgat_mediapipe33_best_acc.pt      # Best accuracy checkpoint
└── pyramid_stgat_mediapipe33_metrics.csv      # Training metrics
```

### Metrics File

The CSV contains:
- `epoch`: Training epoch
- `train_loss`, `train_acc`, `train_triplet`, `train_ce`: Training metrics
- `val_loss`, `val_acc`, `val_triplet`, `val_ce`: Validation metrics
- `learning_rate`: Current LR (with warmup/cosine annealing)

---

## Inference with New Checkpoint

Once training completes, use the new checkpoint with:

### Option 1: Direct Config

```python
from temporal_pyramid_stgat.config import PyramidSTGATConfig
from temporal_pyramid_stgat.inference import PyramidSTGATInference

config = PyramidSTGATConfig.for_uiprmd_mediapipe_33joint()
inference = PyramidSTGATInference(config)
inference.load_checkpoint("temporal_pyramid_stgat/weights/pyramid_stgat_mediapipe33_best.pt")

# Prepare 33-landmark sequence
score = inference.score_sequence(coords_33, angles_12)
```

### Option 2: Video Inference

The `infer_video.py` script automatically detects checkpoint compatibility:

```bash
python Backend/temporal_pyramid_stgat/infer_video.py \
  --video my_video.mp4 \
  --checkpoint Backend/temporal_pyramid_stgat/weights/pyramid_stgat_mediapipe33_best.pt \
  --pose-backend mediapipe
```

---

## Architecture Details

### MediaPipe 33 Landmarks Layout

```
Head:        0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear, ...
Torso:       10=left_shoulder, 11=right_shoulder, 12=left_hip, 13=right_hip
Left Arm:    9=left_wrist, 7=left_elbow, 5=left_shoulder
Right Arm:   10=right_shoulder, 8=right_elbow, 6=right_wrist
Left Leg:    15=left_knee, 17=left_ankle, 19=left_foot_index, 25=left_hip
Right Leg:   16=right_knee, 18=right_ankle, 20=right_foot_index, 26=right_hip
```

### 12 Angle Triplets

| Angle | Joints | Body Part |
|-------|--------|-----------|
| 0 | (12,13,15) | L-shoulder-elbow-wrist |
| 1 | (13,15,17) | R-shoulder-elbow-wrist |
| 2 | (11,14,16) | L-hip-knee-ankle |
| 3 | (14,16,18) | R-hip-knee-ankle |
| 4 | (25,27,29) | L-torso-hip |
| 5 | (26,28,30) | R-torso-hip |
| 6 | (25,26,12) | L-torso-coord |
| 7 | (26,25,11) | R-torso-coord |
| 8 | (11,12,25) | L-coord-hip |
| 9 | (12,11,26) | R-coord-hip |
| 10 | (11,12,25) | L-upper-torso |
| 11 | (12,11,26) | R-upper-torso |

(Actual triplets defined in `MediaPipeAngleCalculator.ANGLE_TRIPLETS_33`)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: mediapipe_mapper"

**Solution**: Ensure `Backend/temporal_pyramid_stgat/preprocessing/mediapipe_mapper.py` exists.

```bash
ls -la Backend/temporal_pyramid_stgat/preprocessing/mediapipe_mapper.py
```

### Issue: "Expected shape (T, 33, 3) but got (T, 39, 3)"

**Solution**: Verify `config.num_joints == 33` in your config factory call.

```python
config = PyramidSTGATConfig.for_uiprmd_mediapipe_33joint()  # Correct
config = PyramidSTGATConfig.for_ui_prmd()  # Wrong (uses old defaults)
```

### Issue: Training fails on "ModuleNotFoundError: mediapipe_uiprmd_loader"

**Solution**: Ensure imports are in `train_triplet.py`. Check lines 26-31 for:

```python
from temporal_pyramid_stgat.preprocessing.mediapipe_uiprmd_loader import MediaPipeUIsprmdLoader
from temporal_pyramid_stgat.preprocessing.mediapipe_angle_calculator import MediaPipeAngleCalculator
```

### Issue: Angles contain NaN or infinite values

**Solution**: Verify all joint coordinates are valid (no NaN in original Vicon data).

```python
# In mediapipe_angle_calculator.py, check:
assert not np.isnan(p1).any() and not np.isnan(p2).any() and not np.isnan(p3).any()
```

---

## Expected Improvements

After retraining with MediaPipe 33-landmark representation:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Video Inference Confidence | 99% (Incorrect) | 70-85% (Correct/Incorrect) |
| Feature Space Alignment | Poor (39→17 mismatch) | Good (33→33 exact) |
| Calibration | Miscalibrated | Calibrated to MediaPipe |

---

## Comparison: Old vs New Pipeline

### Old Pipeline (39-joint Vicon)

```
UI-PRMD Vicon (39 joints)
    ↓
Load as-is (no conversion)
    ↓
Extract angles from 39 joints
    ↓
Train model with in_channels_angle=13
    ↓
Video inference: Pad/truncate YOLO 17 → 39 shape
    → Feature mismatch → Poor calibration
```

### New Pipeline (33-joint MediaPipe)

```
UI-PRMD Vicon (39 joints)
    ↓
Convert to MP 33 landmarks (heuristic mapping)
    ↓
Extract angles from 33 joints
    ↓
Train model with in_channels_angle=12, num_joints=33
    ↓
Video inference: Extract MediaPipe 33 landmarks
    → Exact feature match → Better calibration
```

---

## Next Steps

1. **Run Validation**: `python -m temporal_pyramid_stgat.validation.validate_mediapipe_pipeline`
2. **Start Training**: `python temporal_pyramid_stgat/training/train_mediapipe.py`
3. **Monitor Progress**: Check `temporal_pyramid_stgat/weights/pyramid_stgat_mediapipe33_metrics.csv`
4. **Test on Video**: `python Backend/temporal_pyramid_stgat/infer_video.py --checkpoint weights/pyramid_stgat_mediapipe33_best.pt --pose-backend mediapipe --video /path/to/video.mp4`

---

## Documentation References

- [MediaPipe Pose](https://mediapipe.dev/solutions/pose)
- [Temporal Pyramid STGAT Model](../models/pyramid_stgat.py)
- [Original Training Script](train_triplet.py)
- [Video Inference](../infer_video.py)
