# KinetoCheck Project Map

This map groups the repository by model implementation, preprocessing, datasets, inference, and checkpoints.

## 1. Production Backend (Canonical)

Root: `Backend/`

- API and serving
  - `app/api`
  - `app/services`
- Model definitions
  - `app/models`
- Preprocessors
  - `app/preprocessing`
- Training (baseline ST-GAT family)
  - `training`
- Temporal Pyramid STGAT branch
  - `temporal_pyramid_stgat/models`
  - `temporal_pyramid_stgat/preprocessing`
  - `temporal_pyramid_stgat/training`
  - `temporal_pyramid_stgat/inference.py`
  - `temporal_pyramid_stgat/infer_video.py`
  - `temporal_pyramid_stgat/live_infer_webcam.py`
- Diagnostics
  - `diagnostics/*`
- Setup
  - `setup/*`
- Notebooks
  - `notebooks/*`

## 2. ExeChecker (Experimental Two-Stream)

Root: `ExeChecker/`

- Data pipeline
  - `exechecker/data`
- Models
  - `exechecker/models/stgat.py`
  - `exechecker/models/two_stream.py`
- Training
  - `exechecker/training`
- Inference
  - `exechecker/inference/two_stream_pipeline.py`
- CLI scripts
  - `scripts/train_two_stream.py`
  - `scripts/infer_two_stream_demo.py`
- Diagnostics and notebooks
  - `diagnostics/*`
  - `notebooks/*`

## 3. Datasets (Canonical)

- UI-PRMD: `Datasets/UIPRMD`
- SkeletonData / IntelliRehab-like assets: `Datasets/SkeletonData`

Policy:
- Keep one canonical UI-PRMD copy at `Datasets/UIPRMD`.
- Do not keep duplicated dataset trees under model-project folders.

## 4. Checkpoints and Weights (Preserved)

- Backend baseline: `Backend/weights`
- Backend temporal pyramid: `Backend/temporal_pyramid_stgat/weights`
- ExeChecker two-stream: `ExeChecker/checkpoints`

Policy:
- Keep checkpoints in project-owned folders.
- Never move or overwrite checkpoints without an explicit migration note.

## 5. Compatibility Notes

- After refactor, compatibility wrapper scripts remain in `Backend/` and `ExeChecker/` for old entrypoint paths.
- Prefer running the new canonical script locations for all new automation.
