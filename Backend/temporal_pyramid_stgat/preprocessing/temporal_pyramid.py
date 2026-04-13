"""
Temporal Pyramid Utilities

Creates multi-scale temporal representations for temporal pyramid networks.
Handles downsampling/upsampling at different scales: 1x, 1/2x, 1/4x, 1/8x.
"""

import numpy as np
from typing import List, Tuple, Dict
import torch


class TemporalPyramidSampler:
    """
    Create temporal pyramid by downsampling at different rates.
    """
    
    # Standard pyramid scales
    SCALES = [1, 2, 4, 8]  # Downsample by 1x, 2x, 4x, 8x
    
    @staticmethod
    def create_pyramid(sequence: np.ndarray, 
                      scales: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Create multi-scale temporal pyramid from a single sequence.
        
        Args:
            sequence: (T, ...) base sequence
            scales: Downsampling factors [1, 2, 4, 8] or custom
            
        Returns:
            Dict mapping scale -> downsampled sequence
            Example: {1: (T, ...), 2: (T/2, ...), 4: (T/4, ...), 8: (T/8, ...)}
        """
        if scales is None:
            scales = TemporalPyramidSampler.SCALES
        
        pyramid = {}
        for scale in scales:
            if scale == 1:
                pyramid[scale] = sequence
            else:
                pyramid[scale] = sequence[::scale]
        
        return pyramid
    
    @staticmethod
    def create_pyramid_batch(sequences: np.ndarray,
                            scales: List[int] = None,
                            pad_mode: str = 'constant') -> Dict[int, np.ndarray]:
        """
        Create pyramid for a whole batch and pad to same length per scale.
        
        Args:
            sequences: (N, T, ...) batch of sequences
            scales: Downsampling factors
            pad_mode: 'constant' or 'replicate'
            
        Returns:
            Dict mapping scale -> padded batch (N, T_padded, ...)
        """
        if scales is None:
            scales = TemporalPyramidSampler.SCALES
        
        pyramid_batch = {}
        
        for scale in scales:
            # Downsample batch
            if scale == 1:
                downsampled = sequences
            else:
                downsampled = sequences[:, ::scale]
            
            # Pad to max length in batch
            max_len = downsampled.shape[1]
            padded = TemporalPyramidSampler._pad_batch(downsampled, max_len, pad_mode)
            pyramid_batch[scale] = padded
        
        return pyramid_batch
    
    @staticmethod
    def _pad_batch(batch: np.ndarray, target_len: int, mode: str = 'constant') -> np.ndarray:
        """
        Pad batch sequences to target length.
        
        Args:
            batch: (N, T, ...) 
            target_len: Target temporal dimension
            mode: 'constant' (zero) or 'replicate' (repeat last frame)
            
        Returns:
            Padded batch (N, target_len, ...)
        """
        N, T = batch.shape[0], batch.shape[1]
        
        if T == target_len:
            return batch
        
        if T > target_len:
            return batch[:, :target_len]  # Truncate
        
        # Pad
        pad_len = target_len - T
        shape = list(batch.shape)
        shape[1] = pad_len
        
        if mode == 'constant':
            pad = np.zeros(shape, dtype=batch.dtype)
        elif mode == 'replicate':
            # Repeat last frame
            pad = np.tile(batch[:, -1:], (1, pad_len) + (1,) * (batch.ndim - 2))
        else:
            raise ValueError(f"Unknown pad mode: {mode}")
        
        return np.concatenate([batch, pad], axis=1)


class TemporalPyramidGraph:
    """
    Graph structure for temporal pyramid (pyramid with skip connections).
    """
    
    def __init__(self, scales: List[int] = None):
        """
        Args:
            scales: Pyramid scales [1, 2, 4, 8]
        """
        self.scales = scales or TemporalPyramidSampler.SCALES
        self.connections = self._build_connections()
    
    def _build_connections(self) -> Dict[int, List[int]]:
        """
        Define upsampling connections between scales.
        E.g., scale 8 upsamples to 4, then 4 to 2, then 2 to 1.
        
        Returns:
            Dict mapping scale -> [scales it can fuse with]
        """
        connections = {}
        sorted_scales = sorted(self.scales, reverse=True)
        
        for i, scale in enumerate(sorted_scales):
            if i == 0:
                connections[scale] = []  # Coarsest has no input
            else:
                # Can receive from next finer scale and pass to next coarser
                finer = sorted_scales[i - 1]
                connections[scale] = [finer]
        
        return connections
    
    def get_upsampling_factor(self, from_scale: int, to_scale: int) -> int:
        """
        Get upsampling factor from one scale to another.
        
        Args:
            from_scale: Source scale (coarser)
            to_scale: Target scale (finer)
            
        Returns:
            Upsampling factor
        """
        if from_scale == to_scale:
            return 1
        return from_scale // to_scale


class TemporalAttentionWeights:
    """
    Compute attention weights for multi-scale features.
    """
    
    @staticmethod
    def compute_scale_importance(scales: List[int]) -> Dict[int, float]:
        """
        Compute importance weights for each scale.
        Finer scales (more detail) get higher weight.
        
        Args:
            scales: List of scales [1, 2, 4, 8]
            
        Returns:
            Dict mapping scale -> weight (sums to 1)
        """
        # Weights inversely proportional to scale
        weights_raw = {s: 1.0 / s for s in scales}
        total = sum(weights_raw.values())
        return {s: w / total for s, w in weights_raw.items()}
    
    @staticmethod
    def apply_time_decay(features: torch.Tensor, 
                        decay_factor: float = 0.95) -> torch.Tensor:
        """
        Apply exponential time decay to features over time.
        More recent frames weighted more heavily.
        
        Args:
            features: (T, D)
            decay_factor: Decay per timestep
            
        Returns:
            Weighted features (T, D)
        """
        T = features.shape[0]
        weights = decay_factor ** torch.arange(T - 1, -1, -1, 
                                              dtype=torch.float32,
                                              device=features.device)
        weights = weights / weights.sum()
        
        return features * weights.unsqueeze(1)


class PyramidPreprocessor:
    """
    Combined preprocessing with pyramid creation.
    """
    
    def __init__(self, scales: List[int] = None, normalize: bool = True):
        """
        Args:
            scales: Pyramid scales
            normalize: Whether to normalize before pyramid
        """
        self.scales = scales or TemporalPyramidSampler.SCALES
        self.normalize = normalize
    
    def process_sequence(self, sequence: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Process a single sequence: normalize + create pyramid.
        
        Args:
            sequence: (T, D) or (T, J, C)
            
        Returns:
            Dict mapping scale -> processed sequence
        """
        # Normalize
        if self.normalize:
            seq = self._normalize(sequence)
        else:
            seq = sequence
        
        # Create pyramid
        pyramid = TemporalPyramidSampler.create_pyramid(seq, self.scales)
        return pyramid
    
    def process_batch(self, batch: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Process batch: normalize + create pyramid + pad.
        
        Args:
            batch: (N, T, D) or (N, T, J, C)
            
        Returns:
            Dict mapping scale -> batch pyramid (N, T_scale, ...)
        """
        # Normalize
        if self.normalize:
            batch = self._normalize_batch(batch)
        
        # Create pyramid with padding
        pyramid = TemporalPyramidSampler.create_pyramid_batch(batch, self.scales)
        return pyramid
    
    @staticmethod
    def _normalize(sequence: np.ndarray) -> np.ndarray:
        """Normalize single sequence (z-score)."""
        mean = np.mean(sequence)
        std = np.std(sequence) + 1e-6
        return ((sequence - mean) / std).astype(np.float32)
    
    @staticmethod
    def _normalize_batch(batch: np.ndarray) -> np.ndarray:
        """Normalize batch (per-sample z-score)."""
        N = batch.shape[0]
        normalized = []
        
        for i in range(N):
            seq = batch[i]
            mean = np.mean(seq)
            std = np.std(seq) + 1e-6
            normalized.append((seq - mean) / std)
        
        return np.stack(normalized).astype(np.float32)


def create_pyramid_collate_fn(scales: List[int] = None):
    """
    Custom collate function for DataLoader to create pyramids in batch.
    
    Args:
        scales: Pyramid scales
        
    Returns:
        Collate function
    """
    if scales is None:
        scales = TemporalPyramidSampler.SCALES
    
    def collate_fn(batch):
        # batch: List of tuples (seq, label) where seq is (T, D)
        sequences = np.array([item[0] for item in batch])  # (N, T, D)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        
        # Create pyramid batch
        pyramid = TemporalPyramidSampler.create_pyramid_batch(sequences, scales)
        
        # Convert to tensors
        pyramid_tensors = {
            scale: torch.from_numpy(downsampled).float()
            for scale, downsampled in pyramid.items()
        }
        
        return pyramid_tensors, labels
    
    return collate_fn
