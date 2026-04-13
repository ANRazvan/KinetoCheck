"""
Temporal Pyramid STGAT: Interpretable Exercise Quality Assessment

A Spatio-Temporal Graph Attention Network with Temporal Pyramid and Triplet Loss
for assessing physical rehabilitation exercise quality on the UI-PRMD dataset.

Provides three levels of interpretable feedback:
1. Binary classification (Correct/Incorrect)
2. Quality score (0-1) via embedding distance
3. Attention maps (future enhancement)

Key components:
- Temporal Pyramid: Multi-scale temporal processing [1×, 2×, 4×, 8×]
- STGAT: Spatial-Temporal Graph Attention for skeleton joints
- Two-Stream: Coordinates + Calculated angles
- Triplet Loss: Learns embedding space where correct movements cluster
- Distance mapping: Embedding distances → 0-1 quality scores

Usage:
    from temporal_pyramid_stgat import PyramidSTGATInference
    
    inference = PyramidSTGATInference("model.pt")
    result = inference.predict_from_sequence(skeleton_seq)
    
    print(f"Prediction: {result['prediction_label']}")
    print(f"Quality: {result['quality_score']:.3f}")
"""

from . import preprocessing
from . import models
from . import training
from .config import PyramidSTGATConfig

__version__ = "0.1.0"
__all__ = [
    'preprocessing',
    'models',
    'training',
    'PyramidSTGATConfig',
]
