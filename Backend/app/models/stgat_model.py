import os
import logging
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

from app.models.base_model import BaseMovementModel
from app.preprocessing.graph_builder import KINECT_EDGES
from config import settings


# 17-joint order must match UIPRMDPreprocessor.align_vicon_to_mediapipe output:
# [MP0, MP11, MP12, MP13, MP14, MP15, MP16, MP23, MP24,
#  MP25, MP26, MP27, MP28, MP29, MP30, MP31, MP32]
UIPRMD_VICON_17_JOINT_NAMES = [
    "Nose",            # 0  (MP0)
    "LShoulder",       # 1  (MP11)
    "RShoulder",       # 2  (MP12)
    "LElbow",          # 3  (MP13)
    "RElbow",          # 4  (MP14)
    "LWrist",          # 5  (MP15)
    "RWrist",          # 6  (MP16)
    "LHip",            # 7  (MP23)
    "RHip",            # 8  (MP24)
    "LKnee",           # 9  (MP25)
    "RKnee",           # 10 (MP26)
    "LAnkle",          # 11 (MP27)
    "RAnkle",          # 12 (MP28)
    "LHeel",           # 13 (MP29)
    "RHeel",           # 14 (MP30)
    "LFootIndex",      # 15 (MP31)
    "RFootIndex",      # 16 (MP32)
]

# Spatial graph for the aligned 17-joint Vicon/MediaPipe representation.
UIPRMD_VICON_17_EDGES = [
    (0, 1), (0, 2),                     # head to shoulders
    (1, 2),                             # shoulder girdle
    (1, 3), (3, 5),                     # left arm
    (2, 4), (4, 6),                     # right arm
    (1, 7), (2, 8), (7, 8),             # torso
    (7, 9), (9, 11),                    # left upper/lower leg
    (8, 10), (10, 12),                  # right upper/lower leg
    (11, 13), (11, 15), (13, 15),       # left foot chain
    (12, 14), (12, 16), (14, 16),       # right foot chain
]


def get_edges_for_num_keypoints(num_keypoints: int):
    """Return the skeleton graph edges for the active joint layout."""
    if num_keypoints == len(UIPRMD_VICON_17_JOINT_NAMES):
        return UIPRMD_VICON_17_EDGES
    if num_keypoints == settings.NUM_KEYPOINTS:
        return KINECT_EDGES

    logger.warning(
        "Unknown num_keypoints=%s. Falling back to Kinect graph and filtering out-of-range edges.",
        num_keypoints,
    )
    return KINECT_EDGES


def build_edge_index(edges, num_nodes: int) -> torch.Tensor:
    """Build a bidirectional edge_index tensor from an edge list."""
    # Filter edges to ensure they are within num_nodes range
    valid_edges = [e for e in edges if e[0] < num_nodes and e[1] < num_nodes]
    
    if len(valid_edges) < len(edges):
        logger.warning(f"Filtered {len(edges) - len(valid_edges)} edges out of bounds for num_nodes={num_nodes}")
        
    src = [e[0] for e in valid_edges] + [e[1] for e in valid_edges]
    dst = [e[1] for e in valid_edges] + [e[0] for e in valid_edges]
    # Add self-loops
    src += list(range(num_nodes))
    dst += list(range(num_nodes))
    return torch.tensor([src, dst], dtype=torch.long)


class TemporalConvBlock(nn.Module):
    """1-D temporal convolution across the frame dimension."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch * num_keypoints, channels, seq_len)
        return F.relu(self.bn(self.conv(x)))


class SpatialGATBlock(nn.Module):
    """Graph Attention over the skeleton at each time step."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout, concat=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (num_nodes_in_batch, channels)
        out = self.gat(x, edge_index)
        return F.relu(self.bn(out))


class STGATNetwork(nn.Module):
    """
    Spatial-Temporal Graph Attention Network.
    
    Pipeline per ST-GAT block:
        1. Spatial GAT across skeleton graph (per frame)
        2. Temporal Conv across time (per keypoint)
    """

    def __init__(
        self,
        num_keypoints: int,
        keypoint_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        num_heads: int,
        seq_length: int,
        dropout: float,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(keypoint_dim, hidden_dim)

        # Stacked ST-GAT blocks
        self.spatial_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.spatial_blocks.append(
                SpatialGATBlock(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
            )
            self.temporal_blocks.append(
                TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3)
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Pre-build edge index with dataset-consistent skeleton topology.
        edges = get_edges_for_num_keypoints(num_keypoints)
             
        self.register_buffer(
            "edge_index",
            build_edge_index(edges, num_keypoints),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, num_keypoints, keypoint_dim)
        Returns:
            logits: (batch, num_classes)
        """
        B, T, K, D = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, K, H)

        for spatial_blk, temporal_blk in zip(self.spatial_blocks, self.temporal_blocks):
            # ── Spatial GAT (process each frame independently) ──
            x_flat = x.reshape(B * T, K, self.hidden_dim)            # (B*T, K, H)
            # Build batched graph
            graphs = []
            for i in range(B * T):
                g = Data(x=x_flat[i], edge_index=self.edge_index)
                graphs.append(g)
            batch = Batch.from_data_list(graphs)
            spatial_out = spatial_blk(batch.x, batch.edge_index)       # (B*T*K, H)
            x = spatial_out.reshape(B, T, K, self.hidden_dim)

            # ── Temporal Conv (process each keypoint independently) ──
            x_perm = x.permute(0, 2, 3, 1)                            # (B, K, H, T)
            x_perm = x_perm.reshape(B * K, self.hidden_dim, T)        # (B*K, H, T)
            temporal_out = temporal_blk(x_perm)                        # (B*K, H, T)
            x = temporal_out.reshape(B, K, self.hidden_dim, T).permute(0, 3, 1, 2)  # (B, T, K, H)

        # Global pooling: mean over time and keypoints
        x = x.mean(dim=[1, 2])   # (B, H)
        return self.classifier(x)


# ── Strategy wrapper ─────────────────────────────────────────────────

class STGATModel(BaseMovementModel):
    """Concrete Strategy: Spatial-Temporal GAT model."""

    def __init__(self):
        self.model: STGATNetwork | None = None
        # Device selection from settings
        if settings.DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(settings.DEVICE)
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.grad_clip_norm: float = settings.GRAD_CLIP_NORM
        # AMP (only meaningful on CUDA)
        self.use_amp: bool = settings.USE_AMP and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def build(self, **kwargs) -> None:
        self.model = STGATNetwork(
            num_keypoints=kwargs.get("num_keypoints", settings.NUM_KEYPOINTS),
            keypoint_dim=kwargs.get("keypoint_dim", settings.KEYPOINT_DIM),
            hidden_dim=kwargs.get("hidden_dim", settings.GAT_HIDDEN_DIM),
            num_classes=kwargs.get("num_classes", settings.NUM_CLASSES),
            num_layers=kwargs.get("num_layers", settings.GAT_NUM_LAYERS),
            num_heads=kwargs.get("num_heads", settings.GAT_NUM_HEADS),
            seq_length=kwargs.get("seq_length", settings.SEQUENCE_LENGTH),
            dropout=kwargs.get("dropout", settings.GAT_DROPOUT),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=kwargs.get("lr", settings.LEARNING_RATE),
            weight_decay=kwargs.get("weight_decay", settings.WEIGHT_DECAY),
        )

    def load_weights(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Call build() before loading weights.")
        
        state = torch.load(path, map_location=self.device)
        
        # Filter out keys with shape mismatches (e.g., edge_index due to different num_keypoints)
        model_state = self.model.state_dict()
        filtered_state = {}
        for k, v in state.items():
            if k in model_state:
                if v.shape != model_state[k].shape:
                    logger.warning(f"Skipping layer {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state[k].shape}")
                    continue
            filtered_state[k] = v
            
        # Use strict=False to handle missing/unexpected keys after filtering
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.eval()

    def save_weights(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        torch.save(self.model.state_dict(), path)

    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            input_data: (seq_length, num_keypoints, keypoint_dim) or
                        (batch, seq_length, num_keypoints, keypoint_dim)
        """
        if self.model is None:
            raise RuntimeError("Call build() and load_weights() first.")

        self.model.eval()
        tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()

        labels_map = {0: "correct", 1: "incorrect"}
        return {
            "label": labels_map.get(pred_class, str(pred_class)),
            "confidence": round(confidence, 4),
            "details": {"raw_probs": probs[0].cpu().tolist()},
        }

    def train_step(self, batch: Any) -> Dict[str, float]:
        self.model.train()
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
            logits = self.model(x)
            loss = self.criterion(logits, y)

        self.scaler.scale(loss).backward()

        if self.grad_clip_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}

    def eval_step(self, batch: Any) -> Dict[str, float]:
        self.model.eval()
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.use_amp):
            logits = self.model(x)
            loss = self.criterion(logits, y)
            acc = (logits.argmax(dim=-1) == y).float().mean().item()

        return {"loss": loss.item(), "accuracy": acc}

    def get_model_info(self) -> Dict[str, Any]:
        param_count = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "name": "ST-GAT",
            "type": "Spatial-Temporal Graph Attention Network",
            "parameters": param_count,
            "device": str(self.device),
        }