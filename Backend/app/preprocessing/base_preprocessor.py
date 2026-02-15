"""
Base preprocessor utilities for skeleton data.
Contains z-score normalization and sequence length equalization
based on the reference Preprocessor implementation.
"""

import numpy as np


def z_score_normalization(data):
    """
    Apply z-score normalization to the entire dataset.
    
    Args:
        data: list of exercises, each is (seq_length, num_joints, joint_dim)
              or np.ndarray of shape (num_samples, seq_length, num_joints, joint_dim)
    Returns:
        Normalized data in the same structure.
    """
    data = np.array(data, dtype=np.float32)
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        data = (data - mean) / std
    return data.tolist() if isinstance(data, np.ndarray) else data


def make_equal_length(timeseries, target_length):
    """
    Resize a single time series to a target length using linear interpolation.
    
    Args:
        timeseries: list or array of shape (current_length, num_joints, joint_dim)
        target_length: desired number of frames
    Returns:
        Resized time series of shape (target_length, num_joints, joint_dim)
    """
    timeseries = np.array(timeseries, dtype=np.float32)
    current_length = timeseries.shape[0]
    
    if current_length == target_length:
        return timeseries.tolist()
    
    # Flatten spatial dimensions for interpolation
    original_shape = timeseries.shape  # (frames, joints, dim)
    flat = timeseries.reshape(current_length, -1)  # (frames, joints*dim)
    
    original_indices = np.linspace(0, 1, current_length)
    target_indices = np.linspace(0, 1, target_length)
    
    interpolated = np.zeros((target_length, flat.shape[1]), dtype=np.float32)
    for feat_idx in range(flat.shape[1]):
        interpolated[:, feat_idx] = np.interp(target_indices, original_indices, flat[:, feat_idx])
    
    result = interpolated.reshape(target_length, *original_shape[1:])
    return result.tolist()
