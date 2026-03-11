import os
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

from app.models.base_model import BaseMovementModel
from config import settings


# ── Kinect skeleton adjacency (25 joints) ───────────────────────────
# Joint indices: 0=SpineBase, 1=SpineMid, 2=Neck, 3=Head,
# 4=ShoulderLeft, 5=ElbowLeft, 6=WristLeft, 7=HandLeft,
# 8=ShoulderRight, 9=ElbowRight, 10=WristRight, 11=HandRight,
# 12=HipLeft, 13=KneeLeft, 14=AnkleLeft, 15=FootLeft,
# 16=HipRight, 17=KneeRight, 18=AnkleRight, 19=FootRight,
# 20=SpineShoulder, 21=HandTipLeft, 22=ThumbLeft,
# 23=HandTipRight, 24=ThumbRight
KINECT_EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3),                        # spine + head
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),      # left arm + hand
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # right arm + hand
    (0, 12), (12, 13), (13, 14), (14, 15),                    # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),                    # right leg
]


def build_edge_index(edges, num_nodes: int) -> torch.Tensor:
    """Build a bidirectional edge_index tensor from an edge list."""
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
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

        # Pre-build edge index
        self.register_buffer(
            "edge_index",
            build_edge_index(KINECT_EDGES, num_keypoints),
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
        self.model.load_state_dict(state)
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