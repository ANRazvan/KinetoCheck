# KinetoCheck ST-GAT Model: Complete Explanation

A detailed guide to understanding how the ST-GAT (Spatial-Temporal Graph Attention Network) model works, what it learns, how data is preprocessed, and what every hyperparameter does.

---

## Table of Contents

1. [What is ST-GAT?](#what-is-st-gat)
2. [Model Architecture](#model-architecture)
3. [How Data Flows Through the Model](#how-data-flows-through-the-model)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [What the Model Learns](#what-the-model-learns)
6. [Hyperparameters: Complete Reference](#hyperparameters-complete-reference)
7. [Training Process](#training-process)
8. [Inference and Feedback](#inference-and-feedback)
9. [The Ghost Skeleton Overlay](#the-ghost-skeleton-overlay)

---

## What is ST-GAT?

### The Simple Explanation

Imagine you're teaching someone how to do an exercise. You:
1. Look at their **skeleton** (joint positions over time)
2. Compare it frame-by-frame with a **perfect template** (correct form)
3. Notice which **joints are wrong** and by **how much**
4. Tell them what to fix

The ST-GAT model does exactly this using neural networks. It learns to:
- Understand **spatial relationships** (how joints connect in the skeleton)
- Understand **temporal patterns** (how the exercise flows over time)
- Compare the user's pose to the template
- Generate **per-joint feedback** about what's wrong

### The Boring Technical Definition

**ST-GAT = Spatial-Temporal Graph Attention Network**

- **Graph**: Skeleton joints connected by edges (shoulders connected to elbows, etc.)
- **Attention**: The model learns to focus on important joints/frames
- **Spatial**: How joints relate to each other in 3D space
- **Temporal**: How movement evolves over time
- **Network**: A deep learning model with many layers

---

## Model Architecture

### Overview: Two Parallel Brains

The model is **Siamese** — it has two identical copies:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Two Sequences                      │
├────────────────────────────────────────────────────────────┤
│  1. Template (Correct Form)    2. User Attempt              │
│         (B, 12, T_t, 17)            (B, 12, T_u, 17)        │
└────────────────────┬──────────────────────────────┬──────────┘
                     │                              │
        ┌────────────▼─────────────┐   ┌──────────▼─────────────┐
        │  Encoder #1 (ST-GAT)     │   │  Encoder #1 (ST-GAT)   │
        │  - Graph Attention       │   │  - Graph Attention     │
        │  - Temporal Pyramid      │   │  - Temporal Pyramid    │
        └────────────┬─────────────┘   └──────────┬─────────────┘
                     │                            │
        ┌────────────▼──────────────┐  ┌─────────▼──────────────┐
        │  Embedding (128-dim)      │  │  Embedding (128-dim)   │
        │  Template Representation  │  │  User Representation   │
        └────────────┬──────────────┘  └─────────┬──────────────┘
                     │                           │
                     └─────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Similarity Score (0-1)    │
                    │   How correct is the user?  │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │            Phase Aligner (Advanced)                 │
        │  Soft-DTW alignment: match user frames to template  │
        └────────────────┬─────────────────────────────────────┘
                         │
        ┌────────────────▼──────────────────┐
        │   Frame Decoder (Advanced)        │
        │   Per-joint correction deltas     │
        │   (Δx, Δy, Δz for each joint)    │
        └────────────────┬──────────────────┘
                         │
        ┌────────────────▼──────────────────┐
        │    Output: Feedback to User       │
        │  - Where joints are wrong         │
        │  - How much by                    │
        │  - Confidence in prediction       │
        └───────────────────────────────────┘
```

### Component 1: Graph Attention Layer

**What it does:** Looks at relationships between joints.

```python
class GraphAttentionLayer(nn.Module):
    """
    Spatial graph attention over body joints.

    Input shape:  (B, C_in, T, J)
    Output shape: (B, C_out, T, J)
    Attention:    (B, T, J, J)
    
    B = batch size (number of videos)
    C = channels (features per joint)
    T = time frames
    J = joints (17 for COCO skeleton)
    """
```

**How it works:**

```python
def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x shape: (B, C, T, J) - features for each joint at each frame
    # adjacency shape: (J, J) - which joints are connected
    
    # Step 1: Compute attention weights between all pairs of joints
    # Each joint asks: "which other joints should I pay attention to?"
    attention_weights = self.attention_mechanism(x, adjacency)
    
    # Step 2: Use attention to combine information from connected joints
    # Each joint receives weighted information from its neighbors
    output = self.aggregate_with_attention(x, attention_weights)
    
    return output, attention_weights
```

**In Human Terms:**

Imagine joints as people in a team:
- The **elbow** pays attention to the **shoulder** (its parent) and **wrist** (its child)
- The **elbow** learns an **attention weight** for each: "I care 80% about shoulder, 60% about wrist"
- The **elbow** combines information from both: `0.8 * shoulder_info + 0.6 * wrist_info`
- The network learns these weights automatically

**The COCO-17 Skeleton:**

```
The 17 joints are (in order):

0: nose          5: left_wrist     11: left_ankle
1: left_shoulder 6: right_wrist    12: right_ankle
2: right_shoulder 7: left_hip      13: left_heel
3: left_elbow     8: right_hip     14: right_heel
4: right_elbow    9: left_knee     15: left_foot_index
               10: right_knee     16: right_foot_index

The skeleton connections (edges) are:
- Face: nose → eyes
- Arms: shoulder → elbow → wrist
- Torso: shoulders ↔ hips
- Legs: hip → knee → ankle
```

### Component 2: Temporal Pyramid

**What it does:** Looks at movement patterns across time at multiple scales.

```python
class TemporalPyramid(nn.Module):
    """
    Multi-scale temporal modelling with dilated 1-D convolutions per joint.

    Input shape:  (B, C_in, T, J)
    Output shape: (B, C_out, T, J)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Iterable[int] = (3, 5, 7),      # Window sizes
        dilations: Iterable[int] = (1, 2, 3),         # Skip amounts
        dropout: float = 0.1,
    ):
```

**How it works:**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x shape: (B, C, T, J)
    
    # We run 3 parallel convolutions with different receptive fields:
    
    # Conv 1: kernel_size=3, dilation=1
    #   Look at: [t-1, t, t+1]  →  Short-term motion (1 frame window)
    out1 = conv_3x1(x)
    
    # Conv 2: kernel_size=5, dilation=2
    #   Look at: [t-4, t-2, t, t+2, t+4]  →  Medium-term patterns (5 frames)
    out2 = conv_5x2(x)
    
    # Conv 3: kernel_size=7, dilation=3
    #   Look at: [t-9, t-6, t-3, t, t+3, t+6, t+9]  →  Long-term patterns (10 frames)
    out3 = conv_7x3(x)
    
    # Combine all three scales
    output = out1 + out2 + out3
    return output
```

**In Human Terms:**

Imagine analyzing a tennis swing:
- **Short-term** (kernel=3): "The elbow is moving fast right now"
- **Medium-term** (kernel=5): "The elbow is accelerating"
- **Long-term** (kernel=7): "This is the power-generation phase"

All three perspectives matter! The model learns to combine them.

### Component 3: ST-GAT Block

**What it does:** Combines spatial and temporal attention in one block.

```python
class STGATBlock(nn.Module):
    """
    One spatial-temporal block:
    1) Spatial graph attention over joints
    2) Temporal pyramid convolution over frames
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        self.graph_attention = GraphAttentionLayer(in_channels, out_channels)
        self.temporal_pyramid = TemporalPyramid(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Spatial attention (compare joints to each other)
        x_spatial, attention = self.graph_attention(x, adjacency)
        x_spatial = self.dropout(x_spatial)
        
        # Step 2: Temporal convolution (look at motion over time)
        x_temporal = self.temporal_pyramid(x_spatial)
        x_temporal = self.dropout(x_temporal)
        
        # Residual connection (add original + processed)
        output = x + x_temporal
        
        return output, attention
```

**Process:**
```
Input (B, C, T, J)
    ↓
[Graph Attention] → learns joint relationships
    ↓
[Dropout] → prevent overfitting
    ↓
[Temporal Pyramid] → learns motion patterns
    ↓
[Dropout]
    ↓
[Add residual] → input + output (helps gradient flow)
    ↓
Output (B, C, T, J)
```

The model stacks multiple ST-GAT blocks to build deeper representations.

### Component 4: Phase Aligner (Advanced Feature)

**What it does:** Aligns the user's video frames to the template using soft-DTW.

**The Problem:**

```
Template:  Frame 1  →  Frame 2  →  Frame 3  →  Frame 4
           (Squat   (Going     (Bottom    (Rising
            start)   down)      position) back)

User:      Frame 1  →  Frame 2  →  Frame 3  →  Frame 4  →  Frame 5
           (Squat   (Going     (Almost   (Going    (Rising
            start)   down)      bottom)   up)       back)
```

The user's video is slightly slower. Frame 3 (user) doesn't match Frame 3 (template).

**The Solution: Soft-DTW Warping**

```python
class PhaseAligner(nn.Module):
    """
    For every user frame t, it computes a soft weighted sum over template frames.
    
    Example:
    User Frame 3 might align as:
      0.6 × Template Frame 2  +  0.4 × Template Frame 3
    
    Because the user is between two template poses!
    """

    def forward(self, user_feat: torch.Tensor, template_feat: torch.Tensor):
        # Compute similarity between each user frame and each template frame
        # shape: (B, T_u, T_t)
        warp_weights = self.compute_warp_weights(user_feat, template_feat)
        
        # For each user frame, blend multiple template frames
        warped_template = warp_weights @ template_feat
        
        # warp_weights shape: (B, T_u, T_t)
        # template_feat shape: (B, C, T_t, J)
        # warped_template shape: (B, C, T_u, J)
        
        return warped_template, warp_weights
```

### Component 5: Frame Decoder (Advanced Feature)

**What it does:** Predicts joint corrections per frame.

```python
class FrameDecoder(nn.Module):
    """
    Given user and template features, predicts per-joint corrections.
    
    Outputs:
    - delta_xyz  : (B, T, J, 3)  — predicted correction offset (Δx, Δy, Δz)
    - joint_conf : (B, T, J)     — confidence in the prediction [0, 1]
    """

    def forward(self, user_feat: torch.Tensor, warped_tmpl: torch.Tensor):
        # Concatenate user and template features
        combined = torch.cat([user_feat, warped_tmpl], dim=1)
        
        # Compute per-frame corrections
        delta_xyz = self.decoder_network(combined)    # (B, T, J, 3)
        confidence = self.confidence_head(combined)   # (B, T, J)
        
        return delta_xyz, confidence
```

**In Human Terms:**

"The user's right elbow is 5cm too low and 3cm too far back."

The decoder predicts: **Δx = +3cm, Δy = 0cm, Δz = +5cm**

---

## How Data Flows Through the Model

### Training Forward Pass

```python
def forward(
    self,
    template_seq: torch.Tensor,      # Shape: (B, 12, T_t, 17)
    user_seq: torch.Tensor,           # Shape: (B, 12, T_u, 17)
    template_xyz_raw: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    
    # ===== ENCODING PHASE =====
    # Both sequences go through the same encoder
    
    template_embedding, template_all_features, template_attentions = \
        self._encode_full(template_seq, return_attentions=True)
    
    user_embedding, user_all_features, user_attentions = \
        self._encode_full(user_seq, return_attentions=True)
    
    # template_embedding shape: (B, 128) — compressed representation
    # template_all_features shape: (B, C_final, T_t, 17)
    # template_attentions: list of attention maps from each block
    
    # ===== SIMILARITY COMPUTATION =====
    # Compare the two embeddings with contrastive loss
    
    # Normalize embeddings to unit length
    template_embedding_norm = F.normalize(template_embedding, p=2, dim=1)
    user_embedding_norm = F.normalize(user_embedding, p=2, dim=1)
    
    # Compute cosine similarity
    similarity_score = (template_embedding_norm * user_embedding_norm).sum(dim=1)
    # similarity_score shape: (B,) with values in [-1, 1]
    # 1.0 = perfect match, 0.0 = neutral, -1.0 = opposite
    
    # ===== PHASE ALIGNMENT (if enabled) =====
    if self.use_phase_decoder:
        # Align user frames to template using soft-DTW
        warped_template, warp_weights = self.phase_aligner(
            user_all_features, template_all_features
        )
        # warp_weights shape: (B, T_u, T_t)
        # warped_template shape: (B, C, T_u, 17)
        
        # ===== FRAME DECODER =====
        # Predict per-joint corrections
        correction_delta, joint_confidence = self.frame_decoder(
            user_all_features, warped_template
        )
        # correction_delta shape: (B, T_u, 17, 3)
        # joint_confidence shape: (B, T_u, 17)
        
        # ===== JOINT SCORING =====
        # Fuse attention entropy, delta magnitude, confidence
        joint_importance, joint_error_magnitude, joint_confidence_mean = \
            JointScorer.compute_scores(
                user_attentions,
                correction_delta,
                joint_confidence,
            )
```

### Output Dictionary

```python
return {
    # Main similarity score
    'similarity_score': similarity_score,           # (B,)
    
    # Embeddings for contrastive loss
    'template_embedding': template_embedding,       # (B, 128)
    'user_embedding': user_embedding,               # (B, 128)
    
    # Phase-aware feedback (if enabled)
    'warped_template_xyz': warped_template_xyz,     # (B, T_u, 17, 3)
    'correction_delta': correction_delta,           # (B, T_u, 17, 3)
    'joint_confidence': joint_confidence,           # (B, T_u, 17)
    'warp_weights': warp_weights,                   # (B, T_u, T_t)
    
    # Attention maps for interpretability
    'user_attention_weights': user_attentions,      # list of (B, T, J, J)
    
    # Joint diagnostics
    'joint_importance': joint_importance,           # (B, 17)
    'joint_error_magnitude': joint_error_magnitude, # (B, 17)
    'joint_confidence_mean': joint_confidence_mean, # (B, 17)
}
```

---

## Data Preprocessing Pipeline

### Step 1: Load Raw Vicon Data

```python
from Preprocessing.UIPRMD_loader import UIPRMDLoader

loader = UIPRMDLoader(data_root=Path("Datasets/UIPRMD"))
records = loader.load_vicon_data(exercise_id=0)  # Load Exercise 1

# Each record looks like:
# {
#     'file': '/path/to/m01_s02_e01_positions.txt',
#     'sequence': np.array of shape (180, 39, 3),  # 180 frames, 39 Vicon markers, 3D coords
#     'label': 0,                                   # 0=correct, 1=incorrect
#     'subset': 'segmented_correct',
#     'exercise_id': 0,                             # 0-based
#     'subject_id': 2,
#     'rep': 1,
#     'is_incorrect_file': False,
# }
```

**Raw Vicon Data:** 39 markers (full body capture), 3D coordinates (X, Y, Z in mm).

### Step 2: Align Vicon to MediaPipe COCO-17

```python
class UIPRMDPreprocessor:
    def align_vicon_to_mediapipe(self, vicon_data: np.ndarray) -> np.ndarray:
        """
        Convert from 39 Vicon markers to 17 MediaPipe/COCO landmarks.
        
        Input:  (T, 39, 3)  — 39 Vicon markers
        Output: (T, 17, 3)  — 17 MediaPipe joints
        
        Mapping (Vicon → MediaPipe):
        - Vicon marker 0 → MediaPipe joint 0 (nose)
        - Vicon marker 14/15 → MediaPipe joint 1/2 (shoulders)
        - ... and so on
        """
        
        # Extract individual markers
        nose      = vicon_data[:, 0, :]              # Vicon marker 0
        l_shoulder = vicon_data[:, 14, :]            # Vicon marker 14
        r_shoulder = vicon_data[:, 15, :]            # Vicon marker 15
        # ... and 14 more joints
        
        # Stack into COCO-17 order
        aligned = np.stack([
            nose,          # 0
            l_shoulder,    # 1
            r_shoulder,    # 2
            l_elbow,       # 3
            r_elbow,       # 4
            # ... 12 more
        ], axis=1)  # Shape: (T, 17, 3)
        
        # Hip-centered translation
        # Move the origin to the midpoint between left and right hip
        hip_center = (aligned[:, 7, :] + aligned[:, 8, :]) * 0.5
        aligned = aligned - hip_center[:, np.newaxis, :]
        
        return aligned.astype(np.float32)
```

### Step 3: Normalize Coordinates

```python
def normalize(self, keypoints: np.ndarray) -> np.ndarray:
    """
    Z-score normalization: convert raw coordinates to mean=0, std=1.
    
    This centers the data and scales it, making training more stable.
    """
    mean = float(np.mean(keypoints))
    std = float(np.std(keypoints))
    
    if std > 0.0:
        keypoints = (keypoints - mean) / std
    
    return keypoints
```

**Why normalize?**

Raw coordinates are in mm: nose might be at (1200, 800, 500) mm
After normalization: (-0.5, 1.2, -0.3)

This helps the neural network learn better because:
- Values are in a standard range
- Network gradients are more stable
- Different exercises with different scales are comparable

### Step 4: Resample to Fixed Length

```python
def pad_or_truncate(self, keypoints: np.ndarray) -> np.ndarray:
    """
    Resample sequence to target length (default: 150 frames).
    
    Input:  (T, 17, 3)  where T can be any value
    Output: (150, 17, 3)
    
    Method: Linear interpolation between frames.
    """
    
    # If video has 180 frames and we want 150:
    # Resample by linear interpolation
    
    if num_frames == 150:
        return keypoints  # Already correct size
    elif num_frames > 150:
        # Interpolate: take every 1.2th frame
        indices = np.linspace(0, num_frames - 1, 150)
        resampled = np.interp(indices, np.arange(num_frames), keypoints, axis=0)
    else:
        # Extrapolate: fill in between existing frames
        resampled = np.interp(indices, np.arange(num_frames), keypoints, axis=0)
    
    return resampled
```

### Step 5: Compute Features

```python
def build_features_from_aligned(processed: np.ndarray) -> np.ndarray:
    """
    Convert (T, 17, 3) preprocessed sequence into (12, T, 17) model features.
    
    The 12 channels are:
    ┌─ Channels 0-2:   XYZ position (preprocessed)
    ├─ Channels 3-5:   Velocity (Δposition per frame)
    ├─ Channels 6-8:   Acceleration (Δvelocity per frame)
    ├─ Channel 9:      Joint angle (radians)
    ├─ Channel 10:     Angular velocity
    └─ Channel 11:     Bone length normalized by torso
    
    Why 12 channels?
    
    Position alone is not enough. The model needs to understand:
    - How fast the joint is moving (velocity)
    - Whether it's speeding up or slowing down (acceleration)
    - The angle between connected joints (bending, rotation)
    - Relative proportions (bone ratio)
    """
    
    # Step 1: Position (given)
    position = processed  # (T, 17, 3)
    
    # Step 2: Velocity - how position changes
    velocity = np.diff(position, axis=0, prepend=position[:1])
    # For each frame, compute: current_position - previous_position
    # First frame: velocity = 0 (no previous frame)
    
    # Step 3: Acceleration - how velocity changes
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    # Similar to velocity, but for velocities
    
    # Step 4: Joint angles
    angles = compute_joint_angles(processed)  # (T, 17)
    
    # For each joint, compute the angle formed by:
    #   (parent_joint → center_joint → child_joint)
    # Example: angle at elbow = angle formed by (shoulder → elbow → wrist)
    
    # Step 5: Angular velocity
    ang_vel = np.diff(angles, axis=0, prepend=angles[:1])
    
    # Step 6: Bone ratio
    bone = compute_bone_lengths_normalized(processed)  # (T, 16)
    # For each pair of joints, compute distance / torso_length
    # Torso = left_hip to left_shoulder distance
    # This makes the model scale-invariant
    
    # Combine all 12 channels
    features = np.concatenate(
        [position, velocity, acceleration, angles, ang_vel, bone], 
        axis=-1
    )  # (T, 17, 12)
    
    # Transpose to (12, T, 17)
    features = np.transpose(features, (2, 0, 1))
    
    return features
```

### Full Preprocessing Pipeline Diagram

```
Raw Vicon Data (180, 39, 3)
    ↓
[align_vicon_to_mediapipe]  → Map 39 markers to 17 joints
    ↓
Aligned (180, 17, 3)
    ↓
[Hip centering]  → Move origin to hip midpoint
    ↓
Centered (180, 17, 3)
    ↓
[normalize]  → Z-score: (x - mean) / std
    ↓
Normalized (180, 17, 3)
    ↓
[pad_or_truncate]  → Resample to 150 frames
    ↓
Fixed-length (150, 17, 3)
    ↓
[build_features_from_aligned]  → Compute 12 channels
    ↓
Features (12, 150, 17)
    ↓
[Input to model]
```

---

## What the Model Learns

### Loss Functions

The model learns by minimizing three losses:

#### 1. **Contrastive Loss** (Main)

```python
class ContrastiveLoss(nn.Module):
    """
    Makes correct/incorrect pairs have different embedding distances.
    
    If label = 1.0 (correct):
        Pull embeddings together
        loss = distance between embeddings
    
    If label = 0.0 (incorrect):
        Push embeddings apart
        loss = max(0, margin - distance)
    
    Margin: minimum distance required between correct and incorrect
    Default: margin = 1.0
    """
    
    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor):
        # Compute distance between embeddings
        distance = torch.norm(emb_a - emb_b, p=2, dim=1)
        
        if label == 1.0:
            # Correct pair: minimize distance
            loss = distance
        else:
            # Incorrect pair: push apart
            loss = torch.relu(margin - distance)
        
        return loss.mean()
```

**In Human Terms:**

```
Embedding space (2D visualization):

Correct pairs:        Incorrect pairs:
T1 • — • U1           T2 •              • U2  (far apart)
    margin ≈ 1.0

After training:
✓ Correct pairs cluster together
✓ Incorrect pairs are pushed away
```

#### 2. **Delta Regression Loss** (Correction Feedback)

```python
class DeltaRegressionLoss(nn.Module):
    """
    Trains the FrameDecoder to predict accurate joint corrections.
    
    Ground truth delta = warped_template_xyz - user_xyz
    Predicted delta = model output
    
    Loss = Huber loss (robust to outliers)
    
    Only applied when label = 1.0 (correct exercises)
    """
    
    def forward(self, predicted_delta, ground_truth_delta):
        # Huber loss: smooth at 0, linear for large errors
        loss = torch.nn.HuberLoss(delta=0.05)
        return loss(predicted_delta, ground_truth_delta)
```

#### 3. **Range of Motion Loss** (Advanced)

```python
class RangeOfMotionLoss(nn.Module):
    """
    Encourages the model to respect the anatomy of human movement.
    
    For each joint, the model should not predict corrections that
    violate realistic joint limits.
    
    Example:
    - Elbow: can only bend 0° to 160°
    - Hip: can only rotate ±45° (simplified)
    
    If model predicts: "bend elbow 200°" → loss increases
    """
```

### What the Model Learns - Step by Step

**Epoch 1:** Random predictions, high loss

```python
model_output = random_vector()  # Random embeddings
loss = high_value  # Maybe 5.0 or 10.0
```

**Epoch 10:** Starting to distinguish

```python
# Correct pairs get close:
correct_pair_distance = 0.8  # Target: < 1.0
loss_correct = 0.8

# Incorrect pairs push apart:
incorrect_pair_distance = 1.5  # Target: > 1.0
loss_incorrect = 0  # Already far enough

total_loss ≈ 0.4 (lower!)
```

**Epoch 30:** Learned representations

The model learns to extract features like:
- "How deep is the squat?" → dimension 1 of embedding
- "Are knees aligned?" → dimension 2
- "Is posture upright?" → dimension 3
- ... 125 more abstract concepts

---

## Hyperparameters: Complete Reference

### Training Configuration

```python
@dataclass
class TrainingConfig:
    # Data
    data_root: Path = Path("Datasets") / "UIPRMD"
    output_dir: Path = Path("checkpoints") / "uiprmd"
    exercise_ids: tuple = tuple(range(10))  # Which exercises to train
    
    # Training procedure
    epochs: int = 30                    # How many passes through data
    batch_size: int = 8                 # Videos per batch
    learning_rate: float = 1e-3         # Step size for optimizer
    weight_decay: float = 1e-4          # L2 regularization
    patience: int = 10                  # Early stopping: stop if no improvement
    seed: int = 42                      # Random seed
    
    # Data split
    train_ratio: float = 0.8            # 80% training
    val_ratio: float = 0.1              # 10% validation
    test_ratio: float = 0.1             # 10% testing
    
    # Model architecture
    hidden_channels: tuple = (64, 128)  # Feature dimensions per ST-GAT block
    embedding_dim: int = 128            # Final embedding size
    use_phase_decoder: bool = True      # Enable phase alignment + decoder
    
    # Loss weights
    margin: float = 1.0                 # Contrastive loss margin (see below)
    delta_weight: float = 0.1           # How much to weight delta regression loss
    rom_weight: float = 2.0             # How much to weight ROM loss
    
    # Hardware
    device: str = "auto"                # "cuda" or "cpu"
    num_workers: int = 0                # Data loading parallelism
```

### Hyperparameter Details

#### **Learning Rate: 1e-3 = 0.001**

```
How big is each step when updating weights?

Learning rate = 0.001:
    weight_new = weight_old - 0.001 * gradient
    Small steps → slow learning, stable
    
Learning rate = 0.1:
    weight_new = weight_old - 0.1 * gradient
    Large steps → fast learning, might overshoot

Too high: Model diverges (loss increases)
Too low: Model trains forever
```

#### **Weight Decay: 1e-4 = 0.0001**

```
Prevents overfitting by penalizing large weights.

loss = main_loss + 0.0001 * sum(weight²)

Interpretation:
    If weight = 10, penalty = 0.0001 * 100 = 0.01
    If weight = 100, penalty = 0.0001 * 10000 = 1.0
    
    Large weights get heavily penalized → model prefers small weights
    Smaller weights = less specific to training data = better generalization
```

#### **Batch Size: 8**

```
Process 8 videos at once before updating weights.

Larger batch (e.g., 32):
    ✓ More stable gradients
    ✗ Slower updates
    ✗ Uses more memory

Smaller batch (e.g., 1):
    ✓ More frequent updates
    ✓ Uses less memory
    ✗ Noisier gradients
```

#### **Margin: 1.0**

```
In contrastive loss:

If label = 1.0 (correct pair):
    loss = distance
    Goal: minimize to near 0

If label = 0.0 (incorrect pair):
    loss = max(0, 1.0 - distance)
    Goal: distance ≥ 1.0, then loss = 0
    
    If distance = 0.8 (too close):
        loss = max(0, 1.0 - 0.8) = 0.2 (penalize)
    
    If distance = 1.5 (far enough):
        loss = max(0, 1.0 - 1.5) = 0 (no penalty)

Margin interpretation:
    Margin = 0.5: Correct and incorrect can be closer
    Margin = 1.0: Clear separation required
    Margin = 2.0: Strong separation required
```

#### **Delta Weight: 0.1**

```
Balances two losses:

total_loss = contrastive_loss + 0.1 * delta_regression_loss

If delta_weight = 0:
    Only learn similarity, no per-joint corrections

If delta_weight = 1.0:
    Equal focus on both losses

If delta_weight = 10.0:
    Focus heavily on predicting correct deltas
    (but might ignore overall similarity)

Default 0.1 means:
    "Mostly focus on similarity, a bit on deltas"
```

#### **ROM Weight: 2.0**

```
Range of Motion loss weight:

total_loss = ... + 2.0 * rom_loss

High ROM weight (e.g., 2.0):
    ✓ Model respects anatomical limits
    ✗ Less freedom to learn
    
Low ROM weight (e.g., 0.1):
    ✓ More freedom to learn
    ✗ Might predict unrealistic corrections

Default 2.0:
    "Strongly enforce anatomically plausible predictions"
```

#### **Hidden Channels: (64, 128)**

```
Architecture depth:

Layer 1: input (12 channels) → 64 channels
         ST-GAT Block #1
         
Layer 2: 64 channels → 128 channels
         ST-GAT Block #2
         
Layer 3: 128 channels → 128 channels (pooling)
         Final embedding: 128-dim

Deeper (e.g., (128, 256, 256)):
    ✓ More expressive
    ✗ Slower, more memory, more parameters to overfit

Shallower (e.g., (32, 64)):
    ✓ Faster, less memory
    ✗ Less expressive
```

#### **Embedding Dim: 128**

```
Final representation size:

Each video is compressed to a 128-dimensional vector.

128 values capture:
    ~64 concepts (assuming sparse coding)
    
Examples of concepts (learned automatically):
    - "Squat depth" (value: 0.8)
    - "Knee alignment" (value: 0.2)
    - "Upright posture" (value: 0.95)
    - ... 125 more abstract features

Larger (e.g., 512):
    ✓ More capacity
    ✗ Overkill for this problem, more params, slower
    
Smaller (e.g., 32):
    ✓ Less computation
    ✗ Loses information
```

#### **Epochs: 30**

```
How many times to pass through the entire dataset.

Epoch 1: Process all training videos once
Epoch 2: Process all training videos again (but model is different)
...
Epoch 30: Final pass

Typically:
    - Loss decreases each epoch
    - After ~10 epochs: diminishing returns
    - After ~30 epochs: might start overfitting

With patience=10:
    - If validation loss doesn't improve for 10 epochs → stop early
    - Prevents wasting time / overfitting
```

#### **Patience: 10**

```
Early stopping mechanism:

best_val_f1 = 0.70
patience_counter = 0

Epoch 10: val_f1 = 0.72 → new best! reset counter
Epoch 11: val_f1 = 0.71 → no improvement, counter = 1
Epoch 12: val_f1 = 0.70 → no improvement, counter = 2
...
Epoch 20: counter = 10 → STOP training

Saves time and prevents overfitting.
```

---

## Training Process

### 1. Data Loading

```python
train_records = loader.load_vicon_data(exercise_id=0)
# Returns ~200 records of correct exercises

val_records = sample(train_records, 0.1)  # 10% for validation
test_records = sample(train_records, 0.1) # 10% for testing
train_records = remaining 80%

# Example: 200 records:
#   - 160 for training
#   -  20 for validation
#   -  20 for testing
```

### 2. Build Template

```python
template_tensor = build_template_tensor(train_records, preprocessor)

# Process all 160 correct exercises:
# 1. Preprocess each (align, normalize, resample, features)
# 2. Average them: template = mean of all (12, 150, 17) tensors
# 3. Result: (12, 150, 17) — the "perfect" exercise

template_xyz_tensor = build_raw_xyz_template(train_records, preprocessor)
# Same but for raw coordinates (for overlay visualization)
```

### 3. Create Pair Dataset

```python
class ExercisePairDataset:
    def __init__(self, records, template_tensor, preprocessor):
        self.records = records
        self.template = template_tensor  # (12, 150, 17)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        seq = record['sequence']  # (T, 39, 3) raw Vicon
        label = record['label']   # 0 or 1
        
        # Preprocess
        aligned = preprocessor.align_vicon_to_mediapipe(seq)
        processed = preprocessor.process(aligned)  # (12, 150, 17)
        
        return {
            'template': self.template,  # Always the same
            'user_seq': processed,      # Different each time
            'label': float(label),      # 0.0 or 1.0
        }
```

### 4. Training Loop

```python
for epoch in range(1, cfg.epochs + 1):
    for batch in train_loader:
        template_batch = batch['template']      # (B, 12, 150, 17)
        user_batch = batch['user_seq']          # (B, 12, 150, 17)
        labels_batch = batch['label']           # (B,)
        
        # Forward pass
        output = model(template_batch, user_batch)
        
        # Compute losses
        similarity = output['similarity_score']  # (B,)
        embeddings_tmpl = output['template_embedding']
        embeddings_user = output['user_embedding']
        
        loss_contrastive = criterion(
            embeddings_tmpl, embeddings_user, labels_batch
        )
        
        if use_phase_decoder:
            delta_pred = output['correction_delta']
            delta_gt = compute_ground_truth_delta(...)
            loss_delta = delta_criterion(delta_pred, delta_gt)
            loss_rom = rom_criterion(...)
            
            total_loss = loss_contrastive + 0.1*loss_delta + 2.0*loss_rom
        else:
            total_loss = loss_contrastive
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Validation
    val_metrics = evaluate(model, val_loader, ...)
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        torch.save(model.state_dict(), 'best_checkpoint.pt')
    
    print(f"Epoch {epoch} - train_loss: {total_loss:.4f}, val_f1: {val_metrics['f1']:.4f}")
```

### 5. Evaluation Metrics

```python
def classification_metrics(scores, targets, threshold=0.5):
    """
    Given similarity scores and ground-truth labels, compute metrics.
    
    scores: (N,) — model's similarity predictions
    targets: (N,) — ground-truth labels (0 or 1)
    threshold: 0.5 — threshold for binary classification
    
    If score >= 0.5 → predict "correct"
    If score < 0.5 → predict "incorrect"
    """
    
    predictions = (scores >= threshold).float()
    
    # True Positives: predicted 1, actual 1
    tp = ((predictions == 1) & (targets == 1)).sum()
    
    # False Positives: predicted 1, actual 0
    fp = ((predictions == 1) & (targets == 0)).sum()
    
    # True Negatives: predicted 0, actual 0
    tn = ((predictions == 0) & (targets == 0)).sum()
    
    # False Negatives: predicted 0, actual 1
    fn = ((predictions == 0) & (targets == 1)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
```

---

## Inference and Feedback

### Inference Process

```python
def run_prediction(input_tensor, raw_sequence, models, device):
    """
    Given a video, produce exercise feedback.
    
    input_tensor: (1, 12, 150, 17) — preprocessed sequence
    raw_sequence: (150, 17, 3) — original coordinates
    models: list of 10 ExerciseEvaluator (one per exercise)
    """
    
    best_score = -1
    best_model_idx = -1
    best_output = None
    
    # Evaluate against all 10 exercise templates
    for i, model in enumerate(models):
        output = model(template, input_tensor)
        score = output['similarity_score'].item()
        
        if score > best_score:
            best_score = score
            best_model_idx = i
            best_output = output
    
    # Best match:
    exercise_name = f"Exercise {best_model_idx + 1}"
    similarity = best_score  # [0, 1] where 1 = perfect
    
    return {
        'exercise_name': exercise_name,
        'similarity_score': similarity,
        'correction_delta': best_output['correction_delta'],
        'joint_importance': best_output['joint_importance'],
        'worst_joints': top_3_most_important_joints,
    }
```

### Generate Feedback

```python
# From output:
correction_delta = output['correction_delta']  # (1, 150, 17, 3)
joint_importance = output['joint_importance']  # (1, 17)

worst_joints = []
for joint_idx in range(17):
    importance = joint_importance[0, joint_idx].item()
    if importance > threshold:
        avg_delta = correction_delta[0, :, joint_idx, :].abs().mean()
        worst_joints.append({
            'joint': COCO17_NAMES[joint_idx],
            'joint_index': joint_idx,
            'importance': float(importance),
            'avg_correction': float(avg_delta),  # pixels/units
            'problem_score': float(importance * avg_delta),
        })

worst_joints.sort(key=lambda x: x['problem_score'], reverse=True)

# User sees:
# 1. Right knee (problem_score: 0.85) — "Too much inward bend"
# 2. Left ankle (problem_score: 0.71) — "Roll to the outside"
# 3. Lower back (problem_score: 0.68) — "Lean forward too much"
```

---

## The Ghost Skeleton Overlay

### What Is It?

The **ghost skeleton** (displayed in blue on the annotated video) is a semi-transparent animated skeleton showing the **perfect form** overlaid on the user's body, frame-by-frame.

Think of it like a **training partner's skeleton** that appears on top of the user's skeleton:
- User skeleton: **White** (what the user is actually doing)
- Ghost skeleton: **Blue** (what they should be doing at that moment)
- User can visually compare their joints to the ghost joints in real-time

### Visual Feedback

```
Frame-by-frame comparison:

        User Video                    Annotated Video
┌─────────────────────────┐      ┌─────────────────────────┐
│    White Skeleton       │      │    White Skeleton       │
│  (user's actual pose)   │  →   │  (user's actual pose)   │
│                         │      │                         │
│                         │      │  Blue Ghost Skeleton    │
│                         │      │ (correct form for now)  │
│     RGB Frame           │      │                         │
└─────────────────────────┘      └─────────────────────────┘

User can see:
- "My elbow is inside the ghost's elbow" → "too bent inward"
- "My knee is ahead of the ghost's knee" → "knees too far forward"
- "My back is tilted vs ghost" → "posture wrong"
```

### How the Ghost Is Computed

The ghost skeleton computation has **5 critical fixes** baked in:

#### **Fix A: Auto-Detect Camera Orientation**

```python
def _detect_lateral_axis(template_xyz: np.ndarray) -> int:
    """
    Auto-detect which axis is the camera's lateral direction.
    
    Problem:
    - If camera faces the subject front-on: X=depth, Y=lateral
    - If camera faces from the side: X=lateral, Y=depth
    - Hardcoding X is wrong if camera angle changes
    
    Solution: Try both axes, pick the one with wider shoulder separation.
    Wider separation → shoulders are spread side-to-side → that's the lateral axis
    """
    
    # Compute shoulder distance on each axis
    shoulder_sep_x = np.abs(template_xyz[0, :, SHOULDER_L] - template_xyz[0, :, SHOULDER_R])
    shoulder_sep_y = np.abs(template_xyz[1, :, SHOULDER_L] - template_xyz[1, :, SHOULDER_R])
    
    # Which axis has bigger separation?
    if shoulder_sep_x.mean() > shoulder_sep_y.mean():
        return 0  # Use X as lateral axis
    else:
        return 1  # Use Y as lateral axis
```

**Why this matters:** Without this, the ghost could appear mirrored or twisted on the video.

#### **Fix B: Single-Pass Proportional Scaling**

```python
def compute_ghost_scale(template_xyz, user_sequence):
    """
    OLD (BROKEN): Scale height globally, THEN scale X separately → distorts proportions
    NEW (FIXED): Single uniform scale using nose-to-foot distance
    
    Result: Ghost is a rigid, proportionally-correct miniature/magnified version of template
    """
    
    # Measure nose-to-foot distance (full body height)
    tmpl_nose_y = template_xyz[2, :, NOSE_IDX]      # Z axis (height)
    tmpl_foot_y = template_xyz[2, :, [FOOT_L, FOOT_R]].mean(axis=1)
    tmpl_height = np.abs(tmpl_nose_y - tmpl_foot_y).clip(min=1e-3)
    tmpl_height_median = np.median(tmpl_height)
    
    user_nose_y = user_sequence[:, NOSE_IDX, 1]     # Screen Y
    user_foot_y = user_sequence[:, [FOOT_L, FOOT_R], 1].mean(axis=1)
    user_height = np.abs(user_nose_y - user_foot_y).clip(min=1e-3)
    user_height_median = np.median(user_height)
    
    # Single global scale factor
    global_scale = user_height_median / tmpl_height_median
    # Example: user is 200px tall, template is 150px → scale = 1.33
    # All joints scaled by 1.33x (shoulders, elbows, everything)
    
    return global_scale
```

**Why this matters:** Prevents the ghost from looking stretched or squashed.

#### **Fix C: Phase-Locked Frame Timing**

```python
def compute_ghost_timing(user_sequence, template_sequence):
    """
    OLD (BROKEN): User video has 180 frames, template has 150 frames.
    Map frame 1→1, 2→1.67, 3→2.33, etc.
    Problem: Template frame might be "freeze" if model is poorly trained
    
    NEW (FIXED): Linear deterministic mapping of template timeline to video timeline.
    For each video frame, precompute which template frame to display.
    """
    
    T_seq = user_sequence.shape[0]      # Video frames (e.g., 180)
    T_t = template_sequence.shape[0]    # Template frames (e.g., 150)
    
    if T_t == T_seq:
        # Same length: 1-to-1 mapping
        template_frame_per_video = np.arange(T_seq)
    else:
        # Different lengths: linearly resample
        # Map video frames [0, 180) to template frames [0, 150)
        template_frame_per_video = np.linspace(0, T_t - 1, num=T_seq)
    
    # Interpolate between frames for smooth animation
    idx0 = np.floor(template_frame_per_video).astype(int)
    idx1 = np.minimum(idx0 + 1, T_t - 1)
    alpha = (template_frame_per_video - idx0).astype(float)
    
    # For video frame i, blend template[idx0[i]] and template[idx1[i]]
    # Example: video frame 10 maps to template frame 8.3
    #   interpolate: 0.7 * template[8] + 0.3 * template[9]
    
    interpolated_template_poses = (1 - alpha)[:, None, None] * template_seq[idx0] \
                                  + alpha[:, None, None] * template_seq[idx1]
    
    return interpolated_template_poses  # (T_seq, J, 2) smooth over time
```

**Why this matters:** Ghost moves smoothly through every frame, not jumping around or freezing.

#### **Fix D: Ghost From Phase-Aligned Warp**

```python
def compute_perfect_ghost(warp_weights, template_xyz, raw_sequence, ghost_anchor="hips"):
    """
    Ghost is built from phase-aligned template, not raw template.
    
    Process:
    1. Model computes warp_weights: soft alignment between user and template
    2. Use these weights to blend template poses
    3. Result: Ghost reflects what model thinks is phase-matching
    """
    
    # For each user frame, compute weighted blend of template poses
    if warp_weights is not None:
        warped_template = warp_weights @ template_xyz  # (T_u, J, 3)
        # Each user frame gets a blend of template frames based on model's alignment
    else:
        warped_template = template_xyz  # Fallback: use raw template
    
    # Example with concrete numbers:
    # User frame 50 might have warp_weights = [0.1, 0.3, 0.4, 0.2, 0.0, ...]
    # This means: blend 10% frame 0 + 30% frame 1 + 40% frame 2 + ...
    # Result: Ghost pose for user frame 50 is the weighted sum
    
    return warped_template
```

**Why this matters:** Ghost doesn't just copy the template; it warps intelligently based on the model's learned alignment.

### Building the Ghost Frame-by-Frame

```python
for video_frame_idx in range(num_video_frames):
    # Step 1: Get the template pose for this frame
    # (already interpolated if lengths don't match)
    template_pose = interpolated_template_poses[video_frame_idx]  # (J, 2)
    
    # Step 2: Center on user's anchor point
    # Anchor can be hips, ankles, heels, or foot-index
    user_anchor = (raw_sequence[video_frame_idx, ANCHOR_LEFT] + \
                   raw_sequence[video_frame_idx, ANCHOR_RIGHT]) / 2.0
    
    template_centered = template_pose - template_pose[ANCHOR_LEFT+ANCHOR_RIGHT]//2
    
    # Step 3: Scale to match user's current body size
    # (computed once from all frames — robust median)
    ghost_joints = template_centered * global_scale
    
    # Step 4: Position at user's anchor
    ghost_frame = ghost_joints + user_anchor
    
    # Step 5: Clip to screen bounds [0, 1] (normalized coordinates)
    ghost_frame = np.clip(ghost_frame, -0.05, 1.05)
    
    # Store for rendering
    ghost_screen[video_frame_idx] = ghost_frame
```

### Visualization in Video

```python
def draw_ghost_skeleton(frame, ghost_xyz, width, height, alpha=0.55):
    """
    Render the ghost skeleton on the annotated video.
    
    ghost_xyz: (J, 3)  — joint coordinates (some may be out of frame)
    width, height: video resolution
    alpha: transparency (0.55 = semi-transparent)
    
    Process:
    1. Convert normalized coordinates [0,1] to pixels
    2. Draw circles at joints (blue)
    3. Draw lines between connected joints (blue, semi-transparent)
    4. Semi-transparency allows user to see through to their own skeleton
    """
    
    COLOR_OVERLAY = (220, 130, 30)  # Blue-ish overlay color (BGR in OpenCV)
    
    # Convert from normalized [0,1] to pixels [0,width] / [0,height]
    ghost_pixel = ghost_xyz[:, :2].copy()
    ghost_pixel[:, 0] *= width
    ghost_pixel[:, 1] *= height
    
    # Draw joints
    for joint_idx, (x, y) in enumerate(ghost_pixel):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (int(x), int(y)), radius=4, 
                      color=COLOR_OVERLAY, thickness=-1)  # Filled circle
    
    # Draw skeleton edges
    for parent_idx, child_idx in COCO17_EDGES:
        p_x, p_y = ghost_pixel[parent_idx]
        c_x, c_y = ghost_pixel[child_idx]
        
        if 0 <= p_x < width and 0 <= p_y < height and \
           0 <= c_x < width and 0 <= c_y < height:
            # Blend with semi-transparency
            overlay = frame.copy()
            cv2.line(overlay, (int(p_x), int(p_y)), (int(c_x), int(c_y)),
                    color=COLOR_OVERLAY, thickness=2)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
```

### Anchor Modes

The ghost can be anchored to different body parts:

```python
ANCHOR_OPTIONS = {
    "hips": (HIP_L, HIP_R),           # Center hips
    "ankles": (ANKLE_L, ANKLE_R),     # Center ankles (feet on ground)
    "heels": (HEEL_L, HEEL_R),        # Center heels (best for squats)
    "foot_index": (FOOT_L, FOOT_R),   # Center toe area
}

# Choice depends on exercise:
# - Squat: "heels" (feet stay grounded)
# - Running: "hips" (follow center of mass)
# - Jumping: "ankles" (focus on takeoff)
```

### Chirality Fix

```python
def fix_mirror_symmetry(template_xy_seq, raw_sequence):
    """
    If user is mirrored relative to template, flip template X-axis.
    
    Example:
    - Template: left shoulder at X=0.3, right shoulder at X=0.7
    - User: left shoulder at X=0.7, right shoulder at X=0.3
    - User is mirrored! Flip template X: X_new = 1 - X
    """
    
    tmpl_lr = template_xy_seq[:, SHOULDER_L, 0] - template_xy_seq[:, SHOULDER_R, 0]
    user_lr = raw_sequence[:, SHOULDER_L, 0] - raw_sequence[:, SHOULDER_R, 0]
    
    if (tmpl_lr.mean() * user_lr.mean()) < 0:  # Opposite signs
        # Mirror template
        template_xy_seq[:, :, 0] = 1.0 - template_xy_seq[:, :, 0]
        print("Template X flipped to match user chirality")
```

### Diagnostic Information

The ghost computation returns debug info:

```python
ghost_debug = {
    "warp_shape": (150, 150),           # Soft alignment matrix shape
    "tmpl_shape": (150, 17, 2),         # Template coordinates shape
    "entropy": [0.5, 2.1, 3.8],         # Warp entropy (how peaked are weights?)
    "argmax_unique_count": 42,          # How many unique template frames used?
    "anchor_mode": "hips",              # Which anchor was used?
    "anchor_joints": [7, 8],            # Joint indices (hip_l, hip_r)
}

# Interpretation:
# - Low entropy → very peaked weights → model is confident about alignment
# - High entropy → diffuse weights → model is uncertain
# - Few unique frames → ghost reuses template frames (not smooth)
# - Many unique frames → ghost flows smoothly through template
```

---

## Summary: The Big Picture

```python
INPUT: Video of user doing exercise
    ↓
PREPROCESSING: Skeleton extraction, alignment, normalization, feature computation
    ↓
SIAMESE ENCODING: Compare user skeleton to template skeleton
    ↓
SIMILARITY SCORE: "How close to perfect form?" [0, 1]
    ↓
PHASE ALIGNMENT: Match frames between user and template
    ↓
FRAME DECODER: Predict joint corrections per frame
    ↓
JOINT SCORING: Identify problematic joints
    ↓
OUTPUT: Structured feedback showing what's wrong and how to fix it
```

### The Model Learns Through Contrastive Learning

1. See correct exercise → embedding close to template
2. See incorrect exercise → embedding far from template
3. Repeat millions of times → model learns robust representations

### Why This Architecture?

**Graph Attention:**
- Joints are not independent; shoulders affect elbows
- Attention learns which connections matter

**Temporal Pyramid:**
- Squats have different timescales
- Short-term: joint shaking
- Medium-term: acceleration patterns
- Long-term: overall movement flow

**Phase Aligner:**
- User video length ≠ template length
- Align frames dynamically before comparing

**Frame Decoder:**
- Don't just say "wrong"
- Say "elbow 5cm too low, 3cm too forward"

---

## References

- **ST-GAT Paper**: Spatial-Temporal Graph Attention Networks
- **Contrastive Learning**: Siamese networks for metric learning
- **DTW**: Dynamic Time Warping for sequence alignment
- **MediaPipe COCO-17**: Standard pose skeleton with 17 joints
- **UIPRMD Dataset**: UI-PRMD exercise dataset with Vicon motion capture
