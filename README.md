# KinetoCheck

AI-assisted physical rehabilitation movement assessment platform.

This repository contains the ongoing bachelor thesis implementation for automatic evaluation of rehabilitation exercises from skeleton motion data. The project combines dataset-driven model training, API-based inference, video analysis, and real-time camera pipelines.

## 1. Project Purpose

KinetoCheck aims to support rehabilitation assessment by:

- Detecting whether an exercise repetition is executed correctly or incorrectly.
- Providing model confidence and explainability-oriented feedback.
- Bridging research datasets (UI-PRMD, IntelliRehab-like formats) to practical video/camera inference.
- Building toward real-time, therapist-friendly, objective movement analysis.

## 2. Current Repository Structure

- `Backend/`: Main FastAPI backend and canonical model-serving project.
	- `app/models`, `app/preprocessing`, `app/services`: production model, preprocessing, and API inference stack.
	- `training/`: baseline ST-GAT family training.
	- `temporal_pyramid_stgat/`: temporal-pyramid STGAT training + inference branch.
	- `inference/`: real-time Orbbec/Azure Kinect inference utilities.
	- `diagnostics/`: model/data diagnostic scripts.
	- `setup/`: environment and sensor setup scripts.
	- `notebooks/`: backend notebooks.
	- `weights/`: trained model checkpoints (preserved).
- `Datasets/`: Local dataset storage (UI-PRMD and skeleton data).
- `ExeChecker/`: Two-stream experimental pipeline (2D joints + clinical angles).
	- `exechecker/models`, `exechecker/data`, `exechecker/training`, `exechecker/inference`: two-stream implementation.
	- `checkpoints/`: two-stream checkpoints (preserved).
	- `diagnostics/`, `notebooks/`: utility scripts and notebooks.
- `frontendWeb/`: Web frontend solution folder.
- `A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises-master/`: Reference research implementation and materials.
- `RUNNING_GUIDE.md`: Practical run instructions.
- `CUDA_SETUP_STATUS.md`: CUDA and environment setup status.

Canonical data policy:
- `Datasets/UIPRMD` is the only canonical UI-PRMD dataset copy.
- Duplicate project-local dataset copies are removed.

Canonical model policy:
- Keep all trained checkpoints under project-owned folders:
	- `Backend/weights/**`
	- `Backend/temporal_pyramid_stgat/weights/**`
	- `ExeChecker/checkpoints/**`

## 3. What Has Been Done So Far

The following work is already implemented and documented in this repository.

### 3.1 Core Backend and API

- FastAPI backend with prediction endpoints, health/model endpoints, and Swagger docs.
- Video and keypoint inference routes.
- Service-oriented architecture (facade-style inference service, model repository, preprocessing adapters).
- Dataset-aware preprocessing pipelines for different skeleton formats.

### 3.2 ST-GAT Training Pipeline (Mainline)

- Per-exercise binary classification (`correct` vs `incorrect`) training flow.
- Sequence normalization to fixed temporal length.
- Callback-based training loop (early stopping, checkpointing, LR scheduling).
- Multiple dataset variants supported via abstract factory approach.

### 3.3 Temporal Pyramid STGAT Expansion

- Temporal-pyramid training and deployment selection workflows.
- Video inference tooling and assessment CSV outputs.
- Leave-One-Subject-Out (LOSO) support for stronger subject-level validation.

### 3.4 MediaPipe 33-Joint Retraining Work

- Implemented conversion pipeline from Vicon-style representation to MediaPipe 33 landmarks.
- Added 12-angle extraction and standardization stream.
- Added dedicated training and validation scripts for MediaPipe-aligned checkpoints.
- Addressed domain mismatch between training and video inference feature spaces.
- Backend pose extraction path now uses MediaPipe as default and keeps 3D coordinates (`x`, `y`, `z`) for inference where the selected dataset/model expects 3D inputs.

### 3.5 Orbbec / Azure Kinect Real-Time Prototype

- Real-time joint extraction module (`Backend/inference/orbbec_joint_extractor.py`) with:
	- 32-joint to UI-PRMD-style mapping.
	- Pelvis-centered, torso-scaled normalization.
	- Streaming-friendly threaded frame capture design.
- Setup scripts and troubleshooting guides for Windows sensor stack and `pyk4a`.
- Visualizer and real-time inference integration scripts.

### 3.6 Diagnostics and Validation Utilities

- Label-swap hypothesis testing scripts for Vicon data issues.
- Dataset-specific diagnostics and import tests.
- Documentation for architecture patterns and pipeline behavior.

### 3.7 ExeChecker Two-Stream Branch

- Separate two-stream model branch combining:
	- STGAT over 2D joint trajectories.
	- Temporal CNN over angle signals.
- Triplet-based embedding training and LOSO evaluation.
- Joint-attention interpretability artifacts.

## 4. Current Status (March 2026)

### Completed

- End-to-end backend infrastructure for offline inference.
- Multiple trainable model pipelines and dataset adapters.
- MediaPipe retraining infrastructure for better train/infer alignment.
- Initial real-time camera ingestion prototype (Orbbec path).

### In Progress

- Consolidation of model variants into one stable production path.
- Robust cross-exercise benchmarking and metrics standardization.
- Real-time reliability validation in practical usage conditions.

### Known Constraints

- Local NVIDIA GPU is available, but CUDA-enabled PyTorch installation is currently blocked by network/package access constraints in one environment.
- Some scripts remain experimental/research-oriented and require careful configuration per dataset and checkpoint.
- Multi-repetition video segmentation is still an open improvement area.

## 5. Intended Direction (What Will Be Done Next)

This section captures project intent for upcoming thesis and engineering milestones.

### Short Term

- Finalize one canonical training + inference pipeline for thesis experiments.
- Run structured LOSO and cross-subject evaluation across targeted exercises.
- Improve reproducibility (fixed configs, saved experiment manifests, clearer checkpoints).

### Mid Term

- Improve explainability output quality (joint-level and temporal feedback consistency).
- Add repetition-level segmentation for long clips with multiple repetitions.
- Strengthen real-time camera pipeline robustness and fallback behavior.

### Long Term

- Integrate clinician-facing reporting/feedback views.
- Expand to broader exercise coverage and larger test cohorts.
- Package as a stable application workflow suitable for practical pilot testing.

## 6. Bachelor Thesis Contribution Focus

The thesis contribution in this project is centered on:

- Designing a practical end-to-end architecture from capture/video to assessment.
- Solving representation mismatch between training data and real inference data.
- Prototyping real-time skeleton extraction and normalization for deployment readiness.
- Evaluating model behavior with subject-aware protocols and diagnostic tools.

In short: the originality is the integration effort across heterogeneous data formats, model families, and runtime conditions into one rehabilitation-focused assessment system.

## 7. Quick Start

For complete run instructions, see `RUNNING_GUIDE.md`.

Basic backend start:

```powershell
cd Backend
pip install -r requirements.txt
python run.py
```

Pose backend selection (optional):

```powershell
# Default is now MediaPipe 3D
$env:POSE_EXTRACTOR = "mediapipe"

# Fallback to YOLO 2D if needed
$env:POSE_EXTRACTOR = "yolo"
```

API docs:

- `http://localhost:8000/docs`

Temporal Pyramid STGAT training/inference examples are documented in `RUNNING_GUIDE.md` and backend markdown guides.

## 8. Key Tech Stack

- Python, FastAPI, Uvicorn
- PyTorch, Torch Geometric
- OpenCV, MediaPipe, Ultralytics (pose extraction)
- NumPy, SciPy, scikit-learn
- Optional `pyk4a` for Orbbec/Azure Kinect-compatible pipelines

## 9. Important Notes for Future Work

- Keep dataset split protocols and labels strictly versioned.
- Prioritize model calibration and repeatability over raw single-run accuracy.
- Treat real-time camera support as deployment engineering, not only model training.
- Maintain one source of truth for checkpoint selection and experiment tracking.

## 10. Document References

- `RUNNING_GUIDE.md`
- `CUDA_SETUP_STATUS.md`
- `Backend/STGAT_TRAINING_EXPLANATION.md`
- `Backend/MEDIAPIPE_IMPLEMENTATION_SUMMARY.md`
- `Backend/MEDIAPIPE_QUICKSTART.md`
- `Backend/INTELLIREHAB_2D_TRAINING_AND_3D_COMPARISON.md`
- `Backend/LABEL_SWAP_TEST_GUIDE.md`
- `Backend/DESIGN_PATTERNS.md`
- `Backend/WINDOWS_PYK4A_FIX.md`

---

If you are using this repository as your thesis master project log, keep this README updated each time a milestone is validated (new experiment result, architecture change, or deployment step).
