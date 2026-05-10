# Phase-Aware Exercise Evaluator — Upgrade Guide

## What changed and why

### Problem with the original design

The original `ExerciseEvaluator` used global mean-pooling over both time and
joints before projecting to an embedding.  This produced a single
similarity score but threw away all frame-level information, making it
impossible to:

- show what the correct pose looks like *right now* (mid-squat vs. top of squat);
- predict *which direction* a joint needs to move to fix the error;
- distinguish a wrong angle from a wrong phase (user doing the right move too early).

---

## Architecture overview

```
template_seq (B, 9, T_t, J)        user_seq (B, 9, T_u, J)
        │                                    │
        └──────────┐           ┌─────────────┘
                   ▼           ▼
              ┌────────────────────┐
              │  Shared ST-GAT     │  ← unchanged encoder
              │  encoder stack     │    (GraphAttentionLayer
              │  (N blocks)        │     + TemporalPyramid)
              └────────┬───────────┘
               (B,C,T,J) frame features
                   │           │
          template_feat    user_feat
                   │           │
                   └─────┬─────┘
                         │
                  ┌──────▼──────┐
                  │ PhaseAligner │  NEW — soft attention warp
                  │ (T_u × T_t) │       user frames → template time
                  └──────┬──────┘
                         │ warped_template (B, C, T_u, J)
                         │
                  ┌──────▼──────────┐
                  │  FrameDecoder   │  NEW — per-frame per-joint MLP
                  │  (concat → MLP) │       predicts Δxyz + confidence
                  └──────┬──────────┘
                         │
              ┌──────────┴────────────┐
              │                       │
       correction_delta (B,T,J,3)  joint_conf (B,T,J)
              │                       │
              └─────────┬─────────────┘
                        │
                 ┌──────▼──────┐
                 │ JointScorer  │  UPDATED — fuses attention entropy,
                 │              │  delta magnitude, inverse confidence
                 └──────┬───────┘
                        │
               joint_importance (B, J)

 — Global path (unchanged) —
  user_feat / template_feat  →  GlobalAvgPool  →  proj_head  →  L2-norm
  cosine_similarity → similarity_score
```

---

## New modules

### `PhaseAligner`

Soft DTW-style warp implemented as cross-attention:

- Query = user frame features averaged over joints → `(B, H, T_u)`
- Key   = template frame features averaged over joints → `(B, H, T_t)`
- Output = `warp_weights (B, T_u, T_t)` + `warped_template (B, C, T_u, J)`

Every user frame gets a soft weighted combination of template frames.
This removes the need for a separate DTW pre-processing step and the
alignment is jointly learned with the rest of the model.

### `FrameDecoder`

Lightweight MLP head (`Conv2d(2C → C → C/2 → 4)`) that takes the
concatenation of user features and warped template features and predicts:

- `delta_xyz (B, T, J, 3)` — how far and in which direction each joint
  needs to move to match the template phase.
- `joint_conf (B, T, J)` — sigmoid confidence; high = the model is
  certain about its delta prediction.

### `JointScorer` (updated)

Now fuses four signals instead of one:

| Signal             | Weight | Meaning |
|--------------------|--------|---------|
| Attention outgoing mean | 0.30 | Joint was globally important |
| Attention entropy        | 0.20 | Model was uncertain about neighbours |
| Delta magnitude          | 0.35 | Large correction needed |
| Inverse confidence       | 0.15 | Decoder unsure → potential problem |

### `DeltaRegressionLoss` (new)

Auxiliary Huber loss training the FrameDecoder to predict ground-truth
correction deltas (warped template XYZ − user XYZ).  Applied only on
correctly-labelled samples to avoid teaching the model to "correct"
stylistic differences.

Weight controlled by `--delta-weight` (default 0.1 of contrastive loss).

---

## Output dict changes

The `forward()` return dict is a **superset** of the original.  All keys
that existed before are still present at the same tensor shapes.

| Key | Shape | Status |
|-----|-------|--------|
| `similarity_score` | (B,) | **Unchanged** |
| `template_embedding` | (B, D) | **Unchanged** |
| `user_embedding` | (B, D) | **Unchanged** |
| `user_attention_weights` | list[(B,T,J,J)] | **Unchanged** |
| `joint_importance` | (B, J) | Updated (richer score) |
| `warped_template_xyz` | (B, T_u, J, 3) | **New** |
| `correction_delta` | (B, T_u, J, 3) | **New** |
| `joint_confidence` | (B, T_u, J) | **New** |
| `warp_weights` | (B, T_u, T_t) | **New** |
| `joint_error_magnitude` | (B, J) | **New** |
| `joint_confidence_mean` | (B, J) | **New** |

---

## Checkpoint changes

New keys stored in `best_checkpoint.pt`:

| Key | Type | Purpose |
|-----|------|---------|
| `template_xyz_tensor` | Tensor (3, T, J) | Raw XYZ canonical pose (for overlay) |
| `in_channels` | int | Reproducible model rebuild |
| `hidden_channels` | list[int] | idem |
| `embedding_dim` | int | idem |
| `use_phase_decoder` | bool | Whether phase head was trained |
| `feature_channels` | list[str] | `["x","y","z","vx","vy","vz","ax","ay","az"]` |
| `preprocessor_config` | dict | Alignment method used |

All **old keys are preserved** (`template_tensor`, `val_threshold`, etc.).

---

## Backward compatibility

### Loading old checkpoints in inference

```python
model = ExerciseEvaluator(
    in_channels=9,
    hidden_channels=(64, 128),
    embedding_dim=128,
    use_phase_decoder=True,   # new head will be randomly initialised
)
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
# missing = [phase_aligner.*, frame_decoder.*]  — fine, randomly init
```

The similarity score path is **identical**, so old checkpoints give
identical correctness predictions.  The phase decoder will produce noisy
deltas until fine-tuned.

### Loading new checkpoints in old code

Old inference code that only reads `model_state_dict`, `template_tensor`,
and `val_threshold` is unaffected — those fields are unchanged.

### Factory / multi-exercise training

`train_uiprmd_exercises_factory.py` needs the same small additions as
`train_uiprmd_exercises.py`:

1. Import `DeltaRegressionLoss`.
2. Pass `use_phase_decoder=cfg.use_phase_decoder` when constructing `ExerciseEvaluator`.
3. Store `template_xyz_tensor` and the new metadata in each checkpoint.
4. Add the delta loss to the training step the same way shown in
   `train_uiprmd_exercises.py`.

---

## Migration steps

### Re-training from scratch (recommended)

```bash
python Train/train_uiprmd_exercises.py \
    --data-root Datasets/UIPRMD \
    --output-dir checkpoints/uiprmd_v2 \
    --epochs 40 \
    --delta-weight 0.1
```

### Fine-tuning the phase heads only on existing checkpoints

```python
# Freeze encoder, train only phase_aligner + frame_decoder
for name, param in model.named_parameters():
    if "phase_aligner" not in name and "frame_decoder" not in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4
)
```

Recommended: fine-tune for 10–15 epochs with `delta_weight=0.3`.

---

## Video annotation: what's new

The annotated video now shows three layers:

1. **Blue ghost skeleton** — the phase-matched canonical pose for the current
   frame (drawn semi-transparently using `cv2.addWeighted`).
2. **Yellow arrows** on red joints — direction and rough magnitude of the
   predicted correction delta.
3. **Coloured joints** — red = top-3 error joints, amber = attention-heavy,
   white = fine.  Circle radius encodes normalised joint importance.

The HUD adds per-joint deviation magnitude: `Fix: left_knee (0.042)`.

---

## Residual limitations

1. **PhaseAligner is global per frame**, not local per joint.  It cannot
   handle cases where different body parts are at different phases
   simultaneously (e.g., arms and legs desynchronised).  A per-joint
   attention head would address this at higher compute cost.

2. **DeltaRegressionLoss trains only on correct samples**.  With a small
   dataset and class imbalance, the decoder may underfit.  Consider adding
   a small-weight incorrect-sample term with a larger Huber delta.

3. **Template is a mean over all correct-form subjects**.  Subject-specific
   anthropometry differences (arm length, torso ratio) will inflate delta
   magnitudes for body-proportion reasons unrelated to form errors.
   A height-normalised or bone-length-normalised alignment in
   `UIPRMDPreprocessor` would reduce this noise.

4. **Overlay uses normalised [0,1] coordinates**.  The XY overlay is in
   MediaPipe's image-fraction space, which is correct.  The Z (depth)
   channel is not visualised because the camera projection is unknown
   at annotation time.

5. **Warp is soft and bidirectional**.  Unlike hard DTW, the soft warp can
   assign a user frame to multiple template phases simultaneously when the
   model is uncertain.  This is correct probabilistically but can make
   the overlay "smear" during fast transitions.  A sharpness temperature
   (`softmax(logits / τ)` with τ < 1) can be tuned post-training.
