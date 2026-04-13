"""
Spatio-Temporal Graph Attention Network with Temporal Pyramid

Combines:
1. Graph Attention for spatial skeleton topology (17 joints)
2. Temporal Pyramid for multi-scale temporal modeling
3. Two-stream architecture (coordinates + angles)
4. Hierarchical temporal attention across scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============= UI-PRMD Skeleton Graph =============

UIPRMD_EDGES = [
    # Spine (0-6)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    # Shoulders (7, 8) connected to spine
    (4, 7), (4, 8),
    # Right arm (7, 9, 10, 11)
    (7, 9), (9, 10), (10, 11),
    # Left arm (8, 12, 13, 14)
    (8, 12), (12, 13), (13, 14),
    # Legs (15, 16) connected to pelvis
    (0, 15), (0, 16),
]


def build_edge_index(num_nodes: int = 17, edges: list = None) -> torch.Tensor:
    """
    Build bidirectional edge index with self-loops for skeleton graph.
    
    Args:
        num_nodes: Number of skeleton joints
        edges: List of (src, dst) edge tuples
        
    Returns:
        edge_index: (2, E) tensor
    """
    if edges is None:
        edges = UIPRMD_EDGES
    
    # Filter valid edges
    valid_edges = [(s, d) for s, d in edges 
                   if s < num_nodes and d < num_nodes]
    
    # Bidirectional
    src = [e[0] for e in valid_edges] + [e[1] for e in valid_edges]
    dst = [e[1] for e in valid_edges] + [e[0] for e in valid_edges]
    
    # Add self-loops
    src += list(range(num_nodes))
    dst += list(range(num_nodes))
    
    return torch.tensor([src, dst], dtype=torch.long)


# ============= Spatial Modules =============

class SpatialGATBlock(nn.Module):
    """
    Spatial Graph Attention over skeleton nodes.
    Processes all T timesteps independently.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.gat = GATConv(
            in_channels, 
            out_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T*J, C_in) node features
            edge_index: (2, E) edge indices
            
        Returns:
            out: (B*T*J, C_out) features
        """
        out = self.gat(x, edge_index)
        out = self.bn(out)
        return F.relu(out)


class TemporalConvBlock(nn.Module):
    """
    Temporal 1D convolution on joint features.
    Processes temporal dynamics for each joint independently.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*J, C_in, T) temporal sequence per joint
            
        Returns:
            out: (B*J, C_out, T)
        """
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)


# ============= Temporal Pyramid Modules =============

class PyramidBranch(nn.Module):
    """
    Single pyramid branch for one temporal scale.
    Combines spatial attention + temporal convolution.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_heads: int = 4, num_joints: int = 17):
        super().__init__()

        self.num_joints = num_joints
        self.edge_index = build_edge_index(num_joints)

        # Spatial-temporal layers
        self.spatial1 = SpatialGATBlock(in_channels, hidden_channels,
                                       num_heads=num_heads)
        self.temporal1 = TemporalConvBlock(hidden_channels, hidden_channels,
                                          kernel_size=3)

        self.spatial2 = SpatialGATBlock(hidden_channels, hidden_channels,
                                       num_heads=num_heads)
        self.temporal2 = TemporalConvBlock(hidden_channels, hidden_channels,
                                          kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, C) batch of sequences

        Returns:
            out: (B, T, J, C) processed features
        """
        B, T, J, C = x.shape

        # Reshape for spatial processing: (B*T*J, C)
        x_flat = x.reshape(B * T * J, C)

        # Repeat edge index for batch
        edge_index = self.edge_index.to(x.device)

        # Spatial attention
        x_spatial = self.spatial1(x_flat, edge_index)  # (B*T*J, C)

        # Reshape for temporal processing: (B*J, C, T)
        x_temp = x_spatial.reshape(B, T, J, -1)
        x_temp = x_temp.permute(0, 2, 3, 1).reshape(B * J, -1, T)

        # Temporal convolution
        x_temp = self.temporal1(x_temp)  # (B*J, C, T)

        # Reshape back: (B*T*J, C)
        x_temp = x_temp.reshape(B, J, -1, T).permute(0, 3, 1, 2)
        x_temp = x_temp.reshape(B * T * J, -1)

        # Second spatial layer
        x_spatial = self.spatial2(x_temp, edge_index)

        # Second temporal layer
        x_temp = x_spatial.reshape(B, T, J, -1)
        x_temp = x_temp.permute(0, 2, 3, 1).reshape(B * J, -1, T)
        x_temp = self.temporal2(x_temp)

        # Back to (B, T, J, C)
        x_out = x_temp.reshape(B, J, -1, T).permute(0, 3, 1, 2)

        return x_out


class MultiScaleFusion(nn.Module):
    """
    Fuses features from different temporal scales using attention.
    """
    
    def __init__(self, channels: int, num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales
        
        # Cross-scale attention
        self.attention = nn.MultiheadAttention(
            channels, 
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # Fusion weights (learnable)
        self.scale_weights = nn.Parameter(
            torch.ones(num_scales) / num_scales
        )
    
    def forward(self, pyramid: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pyramid: Dict mapping scale -> (B, T_scale, J, C) features
            
        Returns:
            fused: (B, T_max, J, C) fused features at finest scale
        """
        scales = sorted(pyramid.keys())
        features_list = []
        
        # Upsample coarser features to finest scale
        finest_t = pyramid[scales[0]].shape[1]
        
        for scale in scales:
            feat = pyramid[scale]  # (B, T_scale, J, C)
            B, T_scale, J, C = feat.shape
            
            if scale > 1:
                # Upsample: (B, T_scale, J, C) -> (B, T_max, J, C)
                feat = self._upsample_temporal(feat, finest_t)
            
            features_list.append(feat)
        
        # Stack features: (num_scales, B, T_max, J, C)
        stacked = torch.stack(features_list)
        
        # Apply learned weights
        weights = F.softmax(self.scale_weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
        
        return weighted
    
    @staticmethod
    def _upsample_temporal(feat: torch.Tensor, target_t: int) -> torch.Tensor:
        """Upsample temporal dimension."""
        B, T, J, C = feat.shape
        # Simple repeat upsampling
        scale_factor = target_t / T
        feat_reshaped = feat.permute(0, 3, 1, 2)  # (B, C, T, J)
        upsampled = F.interpolate(
            feat_reshaped, 
            size=(target_t, J),
            mode='bilinear',
            align_corners=False
        )
        return upsampled.permute(0, 2, 3, 1)  # (B, T, J, C)


class FrameMLPScorer(nn.Module):
    """Frame-wise MLP scorer that outputs incorrectness logits per frame."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) -> (B, T)
        return self.net(x).squeeze(-1)


class FrameRNNScorer(nn.Module):
    """Frame-wise bidirectional GRU scorer for temporal consistency."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) -> (B, T)
        y, _ = self.gru(x)
        y = self.dropout(y)
        return self.out(y).squeeze(-1)


class FrameTCNScorer(nn.Module):
    """Frame-wise temporal CNN scorer."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) -> (B, H, T)
        y = x.transpose(1, 2)
        y = F.relu(self.bn1(self.conv1(y)))
        y = self.dropout(y)
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.dropout(y)
        # (B, 1, T) -> (B, T)
        return self.out(y).squeeze(1)


# ============= Full Model =============

class TemporalPyramidSTGAT(nn.Module):
    """
    Complete Temporal Pyramid STGAT model with two-stream fusion and triplet loss.
    
    Stream 1: Skeleton coordinates processed by ST-GAT with pyramid
    Stream 2: Joint angles (separate simple CNN-LSTM pipeline)
    
    Outputs:
    - Embeddings: Used for triplet loss training
    - Binary predictions: Correct/Incorrect classification
    - Distance scores: 0-1 mapping of embedding distances for interpretable feedback
    """
    
    def __init__(self, 
                 in_channels_coord: int = 3,
                 in_channels_angle: int = 13,
                 hidden_channels: int = 64,
                 num_heads: int = 4,
                 num_joints: int = 17,
                 num_scales: int = 4,
                 scales: list = None,
                 dropout: float = 0.3,
                 embedding_dim: int = 128,
                 use_triplet_loss: bool = True,
                 frame_head_type: str = "gru",
                 frame_head_hidden: int = 128,
                 frame_aggregation: str = "topk_mean",
                 frame_topk_ratio: float = 0.2):
        super().__init__()
        
        self.num_joints = num_joints
        self.scales = scales or [1, 2, 4, 8]
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.use_triplet_loss = use_triplet_loss
        self.frame_head_type = frame_head_type.lower()
        self.frame_aggregation = frame_aggregation.lower()
        self.frame_topk_ratio = frame_topk_ratio
        
        # ===== Stream 1: Spatial-Temporal Graph Attention (Coordinates) =====
        
        # Input projection
        self.coord_proj = nn.Linear(in_channels_coord, hidden_channels)
        
        # Pyramid branches (one per scale)
        self.pyramid_branches = nn.ModuleDict({
            str(scale): PyramidBranch(
                hidden_channels, 
                hidden_channels,
                num_heads=num_heads,
                num_joints=num_joints
            )
            for scale in self.scales
        })
        
        # Multi-scale fusion
        self.pyramid_fusion = MultiScaleFusion(hidden_channels, 
                                               num_scales=len(self.scales))
        
        # ===== Stream 2: Angle Features (CNN + LSTM) =====
        
        self.angle_proj = nn.Linear(in_channels_angle, hidden_channels)
        
        # Temporal CNN for angles
        self.angle_conv1 = TemporalConvBlock(hidden_channels, hidden_channels, 
                                             kernel_size=3)
        self.angle_conv2 = TemporalConvBlock(hidden_channels, hidden_channels, 
                                             kernel_size=3)
        
        # LSTM for angle dynamics
        self.angle_lstm = nn.LSTM(
            hidden_channels,
            hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # ===== Stream Fusion & Output Heads =====
        
        # Fuse both streams
        self.fusion_weight_coord = nn.Parameter(torch.tensor(0.6))
        self.fusion_weight_angle = nn.Parameter(torch.tensor(0.4))
        
        # Embedding projection (for triplet loss)
        self.embedding_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Frame-wise correctness head (colleague proposal).
        if self.frame_head_type == "mlp":
            self.frame_scorer = FrameMLPScorer(hidden_channels, frame_head_hidden, dropout)
        elif self.frame_head_type in {"rnn", "gru"}:
            self.frame_scorer = FrameRNNScorer(hidden_channels, frame_head_hidden, dropout)
        elif self.frame_head_type in {"1dcnn", "cnn1d", "tcn", "conv1d"}:
            self.frame_scorer = FrameTCNScorer(hidden_channels, frame_head_hidden, dropout)
        else:
            raise ValueError(
                f"Unknown frame_head_type '{frame_head_type}'. "
                "Expected one of: mlp | gru | 1dcnn"
            )
        
        # Binary classification head (correct/incorrect)
        self.binary_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)  # 2 classes
        )
        
        # Distance-to-score mapping (learns to map embedding distances to 0-1)
        # Parameterized as a sigmoid with learnable scale and shift
        self.distance_scale = nn.Parameter(torch.tensor(10.0))
        self.distance_shift = nn.Parameter(torch.tensor(0.5))

    def _aggregate_frame_probs(self, frame_probs: torch.Tensor) -> torch.Tensor:
        """Aggregate frame-level incorrectness probabilities to clip-level probability."""
        if self.frame_aggregation == "mean":
            return frame_probs.mean(dim=1)
        if self.frame_aggregation == "max":
            return frame_probs.max(dim=1).values
        if self.frame_aggregation == "noisy_or":
            return 1.0 - torch.prod(1.0 - frame_probs.clamp(1e-6, 1.0 - 1e-6), dim=1)
        if self.frame_aggregation == "topk_mean":
            t = frame_probs.shape[1]
            k = max(1, int(round(self.frame_topk_ratio * t)))
            topk_vals, _ = torch.topk(frame_probs, k=k, dim=1)
            return topk_vals.mean(dim=1)
        raise ValueError(
            f"Unknown frame_aggregation '{self.frame_aggregation}'. "
            "Expected one of: mean | max | topk_mean | noisy_or"
        )
    
    def forward(self, pyramid_coords: Dict[int, torch.Tensor],
                angles: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            pyramid_coords: Dict mapping scale -> (B, T_scale, J, 3) coordinates
            angles: (B, T, num_angles) joint angles
            return_attention: Whether to extract and return attention maps
            
        Returns:
            Dict with keys:
            - 'embeddings': (B, embedding_dim) learned embeddings
            - 'logits': (B, 2) classification logits
            - 'predictions': (B,) binary predictions (0=correct, 1=incorrect)
            - 'distance_scores': (B,) 0-1 quality scores based on embedding distances
            - 'attention_maps': (optional) attention weights from spatial GAT layers
        """
        # ===== Stream 1: Process pyramid through STGAT =====
        
        processed_pyramid = {}
        for scale, coord_feats in pyramid_coords.items():
            # Project coordinates
            B, T, J, C = coord_feats.shape
            coord_proj = self.coord_proj(coord_feats)  # (B, T, J, hidden)
            
            # Process through pyramid branch
            branch_key = str(scale)
            if branch_key not in self.pyramid_branches:
                # If scale not in model, skip or downsample
                continue
            
            processed = self.pyramid_branches[branch_key](coord_proj)
            processed_pyramid[scale] = processed
        
        # Fuse multi-scale features
        if processed_pyramid:
            coord_features = self.pyramid_fusion(processed_pyramid)  # (B, T, J, hidden)
        else:
            coord_features = self.coord_proj(pyramid_coords[1])
        
        # ===== Stream 2: Process angles =====
        
        # Project angles
        angle_feats = self.angle_proj(angles)  # (B, T, hidden)
        
        # Temporal CNN
        angle_feats = angle_feats.permute(0, 2, 1)  # (B, hidden, T)
        angle_feats = self.angle_conv1(angle_feats)
        angle_feats = self.angle_conv2(angle_feats)
        angle_feats = angle_feats.permute(0, 2, 1)  # (B, T, hidden)
        
        # LSTM
        angle_feats, _ = self.angle_lstm(angle_feats)  # (B, T, hidden)
        
        # ===== Fusion =====
        # Build frame embeddings by fusing both streams per timestep.
        coord_frame = coord_features.mean(dim=2)  # (B, T, hidden)
        if angle_feats.shape[1] != coord_frame.shape[1]:
            angle_feats = F.interpolate(
                angle_feats.permute(0, 2, 1),
                size=coord_frame.shape[1],
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        frame_fused = (
            torch.sigmoid(self.fusion_weight_coord) * coord_frame
            + torch.sigmoid(self.fusion_weight_angle) * angle_feats
        )

        # Clip embedding for metric learning branch.
        fused = frame_fused.mean(dim=1)  # (B, hidden)
        
        # ===== Embedding Generation =====
        
        embeddings = self.embedding_proj(fused)  # (B, embedding_dim)
        
        # ===== Binary Classification =====
        
        # Predict incorrectness per frame and aggregate to clip level.
        frame_logits_incorrect = self.frame_scorer(frame_fused)  # (B, T)
        frame_probs_incorrect = torch.sigmoid(frame_logits_incorrect)  # (B, T)
        clip_prob_incorrect = self._aggregate_frame_probs(frame_probs_incorrect)  # (B,)
        clip_logit_incorrect = torch.logit(clip_prob_incorrect.clamp(1e-6, 1.0 - 1e-6))

        logits = torch.stack([-clip_logit_incorrect, clip_logit_incorrect], dim=1)  # (B, 2)
        predictions = logits.argmax(dim=1)  # (B,) -> 0 or 1
        
        # ===== Distance-to-Score Mapping =====
        
        # Keep 0-1 quality score semantics: higher means more correct.
        distance_scores = 1.0 - clip_prob_incorrect
        
        # Build output dict
        output = {
            'embeddings': embeddings,
            'logits': logits,
            'predictions': predictions,
            'distance_scores': distance_scores,
            'frame_logits_incorrect': frame_logits_incorrect,
            'frame_probs_incorrect': frame_probs_incorrect,
            'clip_prob_incorrect': clip_prob_incorrect,
        }
        
        if return_attention:
            # Future: extract attention from GAT layers here
            output['attention_maps'] = None
        
        return output
