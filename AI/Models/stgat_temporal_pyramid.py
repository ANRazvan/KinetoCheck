from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_coco17_adjacency(include_self: bool = True) -> torch.Tensor:
    """
    Build the undirected adjacency matrix for the 17-joint COCO skeleton.

    Joint order:
    0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear,
    5:l_shoulder, 6:r_shoulder, 7:l_elbow, 8:r_elbow,
    9:l_wrist, 10:r_wrist, 11:l_hip, 12:r_hip,
    13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle
    """
    num_joints = 17
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    if include_self:
        adjacency.fill_diagonal_(1.0)

    return adjacency


class GraphAttentionLayer(nn.Module):
    """
    Spatial graph attention over body joints.

    Input shape:  (B, C_in, T, J)
    Output shape: (B, C_out, T, J)
    Attention:    (B, T, J, J)
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.attn_src = nn.Parameter(torch.empty(out_channels))
        self.attn_dst = nn.Parameter(torch.empty(out_channels))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.residual.weight) if isinstance(self.residual, nn.Conv2d) else None
        nn.init.zeros_(self.residual.bias) if isinstance(self.residual, nn.Conv2d) and self.residual.bias is not None else None
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C_in, T, J)
        adjacency: (J, J) with 1 for valid edges and 0 for non-edges
        """
        bsz, _, frames, joints = x.shape
        if adjacency.shape != (joints, joints):
            raise ValueError(f"Expected adjacency of shape {(joints, joints)}, got {tuple(adjacency.shape)}")

        x_btjc = x.permute(0, 2, 3, 1).contiguous()  # (B, T, J, C_in)
        h = self.proj(x_btjc)  # (B, T, J, C_out)

        src_scores = (h * self.attn_src).sum(dim=-1)  # (B, T, J)
        dst_scores = (h * self.attn_dst).sum(dim=-1)  # (B, T, J)
        logits = self.leaky_relu(src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2))  # (B, T, J, J)

        edge_mask = adjacency.to(device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        logits = logits.masked_fill(~edge_mask, float("-inf"))
        attention = F.softmax(logits, dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("btij,btjf->btif", attention, h)  # (B, T, J, C_out)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C_out, T, J)
        out = out + self.residual(x)

        return out, attention


class TemporalPyramid(nn.Module):
    """
    Multi-scale temporal modeling with dilated 1D convolutions applied per joint.

    Input shape:  (B, C_in, T, J)
    Output shape: (B, C_out, T, J)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Iterable[int] = (3, 5, 7),
        dilations: Iterable[int] = (1, 2, 3),
        dropout: float = 0.1,
    ):
        super().__init__()

        kernel_sizes = list(kernel_sizes)
        dilations = list(dilations)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same length")

        self.branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            if k % 2 == 0:
                raise ValueError("Temporal kernel sizes should be odd to preserve sequence length")
            padding = (d * (k // 2), 0)
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), dilation=(d, 1), padding=padding, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        merged_channels = out_channels * len(self.branches)
        self.fuse = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = [branch(x) for branch in self.branches]
        stacked = torch.cat(multi_scale, dim=1)
        return self.fuse(stacked)


class STGATBlock(nn.Module):
    """
    One spatial-temporal block:
    1) Spatial graph attention over joints
    2) Temporal pyramid convolution over frames
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = GraphAttentionLayer(in_channels, out_channels, dropout=dropout)
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        self.temporal_pyramid = TemporalPyramid(out_channels, out_channels, dropout=dropout)
        self.out_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_out, attn = self.spatial_attn(x, adjacency)
        spatial_out = self.activation(self.spatial_bn(spatial_out))

        temporal_out = self.temporal_pyramid(spatial_out)
        out = self.activation(self.out_bn(temporal_out) + spatial_out)
        return out, attn


class ExerciseEvaluator(nn.Module):
    """
    Siamese Temporal Pyramid ST-GAT for movement comparison.

    Inputs:
    - template_seq: (B, C, T, J) from correct template (for example Vicon-aligned)
    - user_seq:     (B, C, T, J) from user performance (for example MediaPipe)

    Outputs:
    - similarity_score: cosine similarity in [-1, 1] between template and user embeddings
    - user_attention_weights: list of per-block attention tensors, each (B, T, J, J)
    - joint_importance: (B, J), averaged from last input-branch attention map
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: tuple[int, ...] = (64, 128),
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        if len(hidden_channels) == 0:
            raise ValueError("hidden_channels must contain at least one stage")

        adjacency = build_coco17_adjacency(include_self=True)
        self.register_buffer("adjacency", adjacency, persistent=False)

        blocks = []
        c_in = in_channels
        for c_out in hidden_channels:
            blocks.append(STGATBlock(c_in, c_out, dropout=dropout))
            c_in = c_out
        self.encoder = nn.ModuleList(blocks)

        self.proj_head = nn.Sequential(
            nn.Linear(c_in, c_in),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c_in, embedding_dim),
        )

    def encode(self, x: torch.Tensor, return_attentions: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attentions = []
        out = x
        for block in self.encoder:
            out, attn = block(out, self.adjacency)
            if return_attentions:
                attentions.append(attn)

        pooled = out.mean(dim=2).mean(dim=2)  # global average over T and J => (B, C)
        embedding = self.proj_head(pooled)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding, attentions

    @staticmethod
    def _joint_importance_from_attention(attention: torch.Tensor) -> torch.Tensor:
        """
        Convert attention map (B, T, J, J) to per-joint importance (B, J).
        Uses mean outgoing attention over time and neighbors.
        """
        return attention.mean(dim=1).mean(dim=-1)

    def forward(self, template_seq: torch.Tensor, user_seq: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        template_embedding, _ = self.encode(template_seq, return_attentions=False)
        user_embedding, user_attentions = self.encode(user_seq, return_attentions=True)

        similarity_score = F.cosine_similarity(template_embedding, user_embedding, dim=-1)

        last_attention = user_attentions[-1]
        joint_importance = self._joint_importance_from_attention(last_attention)

        return {
            "similarity_score": similarity_score,
            "template_embedding": template_embedding,
            "user_embedding": user_embedding,
            "user_attention_weights": user_attentions,
            "joint_importance": joint_importance,
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss on siamese embeddings.

    Label convention:
    - label = 1.0 -> similar pair (correct phase matches template)
    - label = 0.0 -> dissimilar pair (incorrect phase)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float().view(-1)

        distances = 1.0 - F.cosine_similarity(emb_a, emb_b, dim=-1)
        positive = labels * distances.pow(2)
        negative = (1.0 - labels) * F.relu(self.margin - distances).pow(2)
        loss = (positive + negative).mean()
        return loss
