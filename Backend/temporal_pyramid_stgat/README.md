"""
Documentation: 

# Temporal Pyramid STGAT with Triplet Loss & Distance-Based Scoring

## Overview

This is a improved version of your STGAT model designed for **interpretable clinical feedback** on exercise quality. It provides three levels of information:

1. **Binary Result** (Correct/Incorrect) - Tells user if movement is valid
2. **Quality Score (0-1)** - Tells user how far from correct ("how wrong")
3. **Attention Maps** (planned) - Shows user where the errors occur

## Architecture

### Core Components

```
Input: Skeleton coordinates (17 joints) + Calculated angles
        ↓
Stream 1: Spatio-Temporal Graph Attention (Pyramid)
        ↓
Stream 2: Joint Angles (CNN + LSTM)
        ↓
Fusion: Weighted combination of both streams
        ↓
Embedding Projection: D → 128-dim embedding space
        ↓
Triplet Loss: Learn to cluster correct/incorrect movements
        ↓
Binary Classifier: 0=Correct, 1=Incorrect
        ↓
Distance-to-Score Mapper: Embedding norm → 0-1 quality score
        ↓
Output: {prediction, quality_score, embeddings, attention_maps}
```

### Temporal Pyramid

Multi-scale temporal processing at scales [1×, 2×, 4×, 8×]:
- **Scale 1×**: Full resolution (all temporal details)
- **Scale 2×**: Coarse motion patterns
- **Scale 4×**: Very coarse overall trajectory
- **Scale 8×**: Skeletal coarsest representation

Each scale processed independently through STGAT blocks, then features fused.

## Loss Function Strategy

### Triplet Loss (60% weight)
```
L_triplet = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```

- **Anchor**: Current sample
- **Positive**: Another sample with same label (correct or incorrect)
- **Negative**: Sample with different label

Goal: Correct exercises cluster in embedding space, incorrect ones pushed away.

### Cross-Entropy Loss (40% weight)
```
L_ce = -log(P(correct class))
```

Ensures accurate binary classification.

### Combined Loss
```
L_total = 0.6 * L_triplet + 0.4 * L_ce
```

## Quality Score Computation

Quality score derived from embedding distance:

$$\text{quality} = \sigma(-\lambda \cdot (||e|| - \mu))$$

Where:
- $e$ = embedding vector
- $||e||$ = L2 norm (distance from origin)
- $\sigma$ = sigmoid function
- $\lambda$, $\mu$ = learnable parameters

**Interpretation:**
- Score close to 1.0: Correct form (embedding near correct cluster center)
- Score close to 0.0: Incorrect form (embedding far from correct cluster)
- Score ~0.5: Borderline/ambiguous form

## Two-Stream Architecture

### Stream 1: Skeletal Coordinates + Temporal Pyramid
- Input: 17 joint positions (X, Y, Z)
- Processing: Normalized by centering on pelvis
- Model: Spatial-Temporal Graph Attention
- Captures: Body position and motion dynamics

### Stream 2: Joint Angles  
- Input: 13 angles from anatomical joint triplets
- Processing: Standardized (z-score)
- Model: 1D CNN + LSTM
- Captures: Joint flexion angles and angular velocities (clinical quality)

### Fusion
```
fused = w_coord * coord_features + w_angle * angle_features
```

Weighted combination learns importance of spatial vs. angular information.

## Training Configuration

```python
config = PyramidSTGATConfig.for_ui_prmd()

config.batch_size = 16                # Batch size (triplet loss needs varied labels)
config.learning_rate = 0.001          # Initial LR
config.epochs = 100                   # Max epochs
config.early_stopping_patience = 20   # Patience for early stopping
config.validation_split = 0.2         # 80% train, 20% val

# Model params
config.embedding_dim = 128            # Embedding space dimensionality
config.num_scales = 4                 # 4 temporal scales
config.temporal_scales = [1, 2, 4, 8] # Downsample factors
```

## Usage

### Training with Triplet Loss

```bash
# Train on all exercises
python temporal_pyramid_stgat/training/train_triplet.py

# Train on specific exercise
python temporal_pyramid_stgat/training/train_triplet.py --exercise 0 --epochs 150

# With custom learning rate
python temporal_pyramid_stgat/training/train_triplet.py --lr 0.0005 --batch-size 32
```

### Inference with Interpretable Feedback

```python
from temporal_pyramid_stgat.inference import PyramidSTGATInference

# Load model
inference = PyramidSTGATInference("path/to/checkpoint.pt", device="cuda")

# Predict on single sequence
result = inference.predict_from_sequence(skeleton_seq)  # (T, 17, 3)

print(f"Prediction: {result['prediction_label']}")
print(f"Quality Score: {result['quality_score']:.3f}")
print(f"Interpretation: {inference.interpret_quality_score(result['quality_score'])}")

# Predict batch
batch_results = inference.predict_batch(sequences)  # (N, T, 17, 3)
```

### Demo

```bash
python temporal_pyramid_stgat/inference.py \
    --checkpoint temporal_pyramid_stgat/weights/pyramid_stgat_best.pt \
    --dataset Datasets/UIPRMD \
    --exercise 0
```

## Output Structure

When running inference, model returns a dictionary:

```python
{
    'prediction': 0 or 1,                    # 0=Correct, 1=Incorrect
    'prediction_label': 'Correct',           # Human-readable
    'quality_score': 0.87,                   # 0-1 (closer to 1 = more correct)
    'confidence': 0.95,                      # Softmax confidence of prediction
    'class_probabilities': [0.95, 0.05],     # P(correct), P(incorrect)
    'embeddings': array(128,),               # 128-dim embedding
    'logits': array([2.3, -1.8]),            # Raw classification scores
}
```

## Interpretable Feedback Quality Levels

- **0.9-1.0**: "Excellent form! Very close to reference movement."
- **0.75-0.9**: "Good form. Minor adjustments needed."
- **0.6-0.75**: "Fair form. Notable differences from reference."
- **0.4-0.6**: "Poor form. Significant improvement needed."
- **0.0-0.4**: "Very poor form. Major corrections required."

## Advantages Over Binary Classification

| Aspect | Binary Only | With Distance Score |
|--------|-------------|-------------------|
| User Feedback | Pass/Fail | Quantified quality (0-1) |
| Confidence Signal | Logit value (unbounded) | Normalized 0-1 |
| Learning Signal | Only class labels | Metric distance (triplet) |
| Interpretability | Limited | Clear 0-1 scale |
| Clinical Feedback | "You're wrong" | "You're 30% off from correct" |

## Limitations & Future Work

1. **Attention Maps**: Currently None - can be extracted from GAT layers
2. **Subject Generalization**: Test with Leave-One-Subject-Out (LOSO) validation
3. **Fine-tuning**: May need adjustment for MediaPipe 2D data (currently 3D Vicon)
4. **Temporal Scope**: Currently fixed 240 frames - can add variable length support

## Hyperparameter Tuning

If accuracy is poor:

```python
# Increase triplet margin
criterion = CombinedLoss(margin=2.0)  # Higher margin = harder negatives

# Adjust loss weights
criterion = CombinedLoss(lambda_triplet=0.7, lambda_ce=0.3)  # More emphasis on embedding space

# Larger embedding dimension
embedding_dim = 256  # More capacity

# Different scales
temporal_scales = [1, 2, 4, 8, 16]  # More scales
```

## References

- Triplet Loss: [FaceNet paper](https://arxiv.org/abs/1503.03832)
- Graph Attention Networks: [GAT paper](https://arxiv.org/abs/1710.10903)
- Spatio-Temporal Graphs: [ST-GCN paper](https://arxiv.org/abs/1801.07455)
- UI-PRMD: [Original paper](https://arxiv.org/abs/1901.10435)
"""

__all__ = ['README']
