# """
# ST-GAT with Temporal Pyramid and Phase-Aware Feedback.

# Upgrade summary
# ---------------
# - PhaseAligner: soft-DTW-based temporal alignment that warps user frames
#   onto the template time axis so every frame gets a matching template pose.
# - FrameDecoder: a lightweight per-frame head that predicts correction deltas
#   (Δx, Δy, Δz per joint) relative to the phase-matched template.
# - JointScorer: per-joint error magnitude and confidence derived from
#   attention entropy and decoder deltas.
# - ExerciseEvaluator keeps the same encode() / forward() signature.
#   The new output dict is a superset of the old one, so existing training
#   code that only reads similarity_score is unaffected.
# - ContrastiveLoss is unchanged.

# Backward compatibility
# ----------------------
# - Old checkpoints (without the decoder / phase head) can still be loaded:
#   call model.load_state_dict(ckpt, strict=False).  The decoder and phase
#   head will be freshly initialised and will produce noisy deltas until the
#   model is fine-tuned.  The similarity_score pathway is identical.

# FIX (overlay bug)
# -----------------
# - forward() now accepts an optional `template_xyz_raw` argument
#   (shape: (B, 3, T_t, J) in raw image-fraction space).
# - When provided, warped_template_xyz is computed by warping this raw tensor
#   through warp_weights instead of pulling from template_seq[:, :3].
# - template_seq[:, :3] contains preprocessed (hip-centred, z-scored) XYZ —
#   far outside [0,1] — so it was useless for pixel-space overlay.
# - Training code does NOT pass template_xyz_raw → backward compatible.
# - Inference code DOES pass it → overlay lands in frame.
# """

# from __future__ import annotations

# from typing import Iterable

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ---------------------------------------------------------------------------
# # Adjacency matrix
# # ---------------------------------------------------------------------------

# def build_coco17_adjacency(include_self: bool = True) -> torch.Tensor:
#     """
#     Undirected adjacency matrix for the 17-joint COCO skeleton.

#     Joint order:
#     0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear,
#     5:l_shoulder, 6:r_shoulder, 7:l_elbow, 8:r_elbow,
#     9:l_wrist, 10:r_wrist, 11:l_hip, 12:r_hip,
#     13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle
#     """
#     num_joints = 17
#     edges = [
#         (0, 1), (0, 2), (1, 3), (2, 4),
#         (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
#         (5, 11), (6, 12), (11, 12),
#         (11, 13), (13, 15), (12, 14), (14, 16),
#     ]

#     adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)
#     for i, j in edges:
#         adjacency[i, j] = 1.0
#         adjacency[j, i] = 1.0

#     if include_self:
#         adjacency.fill_diagonal_(1.0)

#     return adjacency


# # ---------------------------------------------------------------------------
# # Spatial graph attention
# # ---------------------------------------------------------------------------

# class GraphAttentionLayer(nn.Module):
#     """
#     Spatial graph attention over body joints.

#     Input shape:  (B, C_in, T, J)
#     Output shape: (B, C_out, T, J)
#     Attention:    (B, T, J, J)
#     """

#     def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1, alpha: float = 0.2):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.proj = nn.Linear(in_channels, out_channels, bias=False)
#         self.attn_src = nn.Parameter(torch.empty(out_channels))
#         self.attn_dst = nn.Parameter(torch.empty(out_channels))
#         self.leaky_relu = nn.LeakyReLU(alpha)
#         self.dropout = nn.Dropout(dropout)

#         self.residual: nn.Module
#         if in_channels == out_channels:
#             self.residual = nn.Identity()
#         else:
#             self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         nn.init.xavier_uniform_(self.proj.weight)
#         if isinstance(self.residual, nn.Conv2d):
#             nn.init.xavier_uniform_(self.residual.weight)
#             if self.residual.bias is not None:
#                 nn.init.zeros_(self.residual.bias)
#         nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
#         nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

#     def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         x:         (B, C_in, T, J)
#         adjacency: (J, J) — 1 for valid edges, 0 otherwise

#         Returns
#         -------
#         out:       (B, C_out, T, J)
#         attention: (B, T, J, J)
#         """
#         bsz, _, frames, joints = x.shape
#         if adjacency.shape != (joints, joints):
#             raise ValueError(
#                 f"Expected adjacency of shape {(joints, joints)}, got {tuple(adjacency.shape)}"
#             )

#         x_btjc = x.permute(0, 2, 3, 1).contiguous()   # (B, T, J, C_in)
#         h = self.proj(x_btjc)                           # (B, T, J, C_out)

#         src_scores = (h * self.attn_src).sum(dim=-1)    # (B, T, J)
#         dst_scores = (h * self.attn_dst).sum(dim=-1)    # (B, T, J)
#         logits = self.leaky_relu(
#             src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2)
#         )                                               # (B, T, J, J)

#         edge_mask = (
#             adjacency.to(device=x.device, dtype=torch.bool)
#             .unsqueeze(0).unsqueeze(0)
#         )
#         logits = logits.masked_fill(~edge_mask, float("-inf"))
#         attention = F.softmax(logits, dim=-1)
#         attention = self.dropout(attention)

#         out = torch.einsum("btij,btjf->btif", attention, h)  # (B, T, J, C_out)
#         out = out.permute(0, 3, 1, 2).contiguous()           # (B, C_out, T, J)
#         out = out + self.residual(x)

#         return out, attention


# # ---------------------------------------------------------------------------
# # Temporal pyramid
# # ---------------------------------------------------------------------------

# class TemporalPyramid(nn.Module):
#     """
#     Multi-scale temporal modelling with dilated 1-D convolutions per joint.

#     Input shape:  (B, C_in, T, J)
#     Output shape: (B, C_out, T, J)
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_sizes: Iterable[int] = (3, 5, 7),
#         dilations: Iterable[int] = (1, 2, 3),
#         dropout: float = 0.1,
#     ):
#         super().__init__()

#         kernel_sizes = list(kernel_sizes)
#         dilations = list(dilations)
#         if len(kernel_sizes) != len(dilations):
#             raise ValueError("kernel_sizes and dilations must have the same length")

#         self.branches = nn.ModuleList()
#         for k, d in zip(kernel_sizes, dilations):
#             if k % 2 == 0:
#                 raise ValueError("Temporal kernel sizes should be odd to preserve sequence length")
#             padding = (d * (k // 2), 0)
#             self.branches.append(
#                 nn.Sequential(
#                     nn.Conv2d(
#                         in_channels, out_channels,
#                         kernel_size=(k, 1), dilation=(d, 1),
#                         padding=padding, bias=False,
#                     ),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True),
#                 )
#             )

#         merged_channels = out_channels * len(self.branches)
#         self.fuse = nn.Sequential(
#             nn.Conv2d(merged_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         multi_scale = [branch(x) for branch in self.branches]
#         stacked = torch.cat(multi_scale, dim=1)
#         return self.fuse(stacked)


# # ---------------------------------------------------------------------------
# # ST-GAT block
# # ---------------------------------------------------------------------------

# class STGATBlock(nn.Module):
#     """
#     One spatial-temporal block:
#     1) Spatial graph attention over joints
#     2) Temporal pyramid convolution over frames
#     """

#     def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
#         super().__init__()
#         self.spatial_attn = GraphAttentionLayer(in_channels, out_channels, dropout=dropout)
#         self.spatial_bn = nn.BatchNorm2d(out_channels)
#         self.temporal_pyramid = TemporalPyramid(out_channels, out_channels, dropout=dropout)
#         self.out_bn = nn.BatchNorm2d(out_channels)
#         self.activation = nn.ReLU(inplace=True)

#     def forward(
#         self, x: torch.Tensor, adjacency: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         spatial_out, attn = self.spatial_attn(x, adjacency)
#         spatial_out = self.activation(self.spatial_bn(spatial_out))

#         temporal_out = self.temporal_pyramid(spatial_out)
#         out = self.activation(self.out_bn(temporal_out) + spatial_out)
#         return out, attn


# # ---------------------------------------------------------------------------
# # Phase aligner
# # ---------------------------------------------------------------------------

# class PhaseAligner(nn.Module):
#     """
#     Soft differentiable temporal alignment between a user sequence and a
#     template sequence via a learnable attention-based warping.

#     For every user frame t, it computes a soft weighted sum over template
#     frames, acting like a differentiable DTW warp.  The warped template
#     carries phase-matched reference poses back into user time.

#     Input
#     -----
#     user_feat     : (B, C, T_u, J)  — per-frame ST-GAT features for the user
#     template_feat : (B, C, T_t, J)  — same for the template

#     Output
#     ------
#     warped_template : (B, C, T_u, J) — template features aligned to user time
#     warp_weights    : (B, T_u, T_t)  — soft assignment matrix (for inspection)
#     """

#     def __init__(self, channels: int, num_joints: int = 17, dropout: float = 0.1):
#         super().__init__()
#         hidden = max(32, channels // 4)
#         self.q_proj = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
#         self.k_proj = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
#         self.scale = hidden ** -0.5
#         self.dropout = nn.Dropout(dropout)

#     def forward(
#         self,
#         user_feat: torch.Tensor,
#         template_feat: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         # Queries from user, keys from template
#         q = self.q_proj(user_feat)       # (B, H, T_u, J)
#         k = self.k_proj(template_feat)   # (B, H, T_t, J)

#         # Pool over joints → (B, H, T)
#         q_t = q.mean(dim=-1)             # (B, H, T_u)
#         k_t = k.mean(dim=-1)             # (B, H, T_t)

#         # Attention: (B, T_u, T_t)
#         logits = torch.einsum("bhu,bht->but", q_t, k_t) * self.scale
#         warp_weights = F.softmax(logits, dim=-1)
#         warp_weights = self.dropout(warp_weights)

#         # Warp template features into user time
#         tf = template_feat.permute(0, 2, 1, 3)          # (B, T_t, C, J)
#         warped = torch.einsum("but,btcj->bucj", warp_weights, tf)  # (B, T_u, C, J)
#         warped = warped.permute(0, 2, 1, 3).contiguous()           # (B, C, T_u, J)

#         return warped, warp_weights


# # ---------------------------------------------------------------------------
# # Frame-level correction decoder
# # ---------------------------------------------------------------------------

# class FrameDecoder(nn.Module):
#     """
#     Per-frame, per-joint correction decoder.

#     Given user ST-GAT features and the phase-warped template features,
#     predicts:
#       - delta_xyz  : (B, T, J, 3)  — predicted correction offset in XYZ
#       - joint_conf : (B, T, J)     — per-joint confidence in [0, 1]

#     The delta is in the same normalised coordinate space as the input
#     features (i.e. after UIPRMDPreprocessor).
#     """

#     def __init__(self, channels: int, num_joints: int = 17, dropout: float = 0.1):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels // 2),
#             nn.ReLU(inplace=True),
#         )
#         self.delta_head = nn.Conv2d(channels // 2, 3, kernel_size=1)   # Δxyz
#         self.conf_head  = nn.Conv2d(channels // 2, 1, kernel_size=1)   # confidence logit

#     def forward(
#         self,
#         user_feat: torch.Tensor,
#         warped_tmpl: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         combined = torch.cat([user_feat, warped_tmpl], dim=1)  # (B, 2C, T, J)
#         h = self.mlp(combined)                                  # (B, C/2, T, J)

#         delta_xyz  = self.delta_head(h)                    # (B, 3, T, J)
#         joint_conf = torch.sigmoid(self.conf_head(h))      # (B, 1, T, J)

#         delta_xyz  = delta_xyz.permute(0, 2, 3, 1).contiguous()   # (B, T, J, 3)
#         joint_conf = joint_conf.squeeze(1)                         # (B, T, J)

#         return delta_xyz, joint_conf


# # ---------------------------------------------------------------------------
# # Joint importance scorer
# # ---------------------------------------------------------------------------

# class JointScorer(nn.Module):
#     """
#     Fuses attention entropy, frame-level delta magnitude, and confidence
#     into a single per-joint importance score.
#     """

#     @staticmethod
#     def from_attention_and_decoder(
#         attention_stack: list[torch.Tensor],
#         delta_xyz: torch.Tensor,
#         joint_conf: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         attention_stack : list of (B, T, J, J)
#         delta_xyz       : (B, T, J, 3)
#         joint_conf      : (B, T, J)

#         Returns
#         -------
#         joint_importance : (B, J)
#         """
#         attn_imp = torch.stack(
#             [a.mean(dim=1).mean(dim=-1) for a in attention_stack], dim=0
#         ).mean(dim=0)  # (B, J)

#         last_attn = attention_stack[-1]
#         safe_attn = last_attn.clamp(min=1e-9)
#         entropy = -(safe_attn * safe_attn.log()).sum(dim=-1).mean(dim=1)  # (B, J)

#         delta_mag = delta_xyz.norm(dim=-1).mean(dim=1)  # (B, J)
#         inv_conf = (1.0 - joint_conf).mean(dim=1)       # (B, J)

#         def _norm(t: torch.Tensor) -> torch.Tensor:
#             mn = t.min(dim=-1, keepdim=True).values
#             mx = t.max(dim=-1, keepdim=True).values
#             return (t - mn) / (mx - mn + 1e-6)

#         score = (
#             0.30 * _norm(attn_imp) +
#             0.20 * _norm(entropy)  +
#             0.35 * _norm(delta_mag) +
#             0.15 * _norm(inv_conf)
#         )
#         return score   # (B, J)


# # ---------------------------------------------------------------------------
# # Main model
# # ---------------------------------------------------------------------------

# class ExerciseEvaluator(nn.Module):
#     """
#     Siamese Temporal Pyramid ST-GAT with Phase-Aware Feedback.

#     Inputs
#     ------
#     template_seq     : (B, C, T_t, J)  — canonical correct sequence (preprocessed)
#     user_seq         : (B, C, T_u, J)  — user's attempt (preprocessed)
#     template_xyz_raw : (B, 3, T_t, J)  — OPTIONAL raw image-fraction XYZ of
#                        the template (NOT preprocessed).  When supplied,
#                        warped_template_xyz is produced in image-fraction space
#                        so the inference overlay skeleton lands inside the frame.
#                        When None (default / training), the old behaviour is used
#                        (warp template_seq[:,:3] — preprocessed coords, useful
#                        only for the delta loss, not for visualisation).

#     Outputs  (dict)
#     -------
#     similarity_score   : (B,)
#     template_embedding : (B, embed_dim)
#     user_embedding     : (B, embed_dim)

#     # Phase-aware feedback
#     warped_template_xyz : (B, T_u, J, 3)
#         Image-fraction XYZ when template_xyz_raw is provided;
#         preprocessed XYZ otherwise (backward compat).
#     correction_delta    : (B, T_u, J, 3)  — predicted Δxyz (preprocessed space)
#     joint_confidence    : (B, T_u, J)
#     warp_weights        : (B, T_u, T_t)

#     # Joint-level diagnostics
#     user_attention_weights : list[(B, T, J, J)]
#     joint_importance       : (B, J)
#     joint_error_magnitude  : (B, J)
#     joint_confidence_mean  : (B, J)
#     """

#     def __init__(
#         self,
#         in_channels: int = 9,
#         hidden_channels: tuple[int, ...] = (64, 128),
#         embedding_dim: int = 128,
#         num_joints: int = 17,
#         dropout: float = 0.1,
#         use_phase_decoder: bool = True,
#     ):
#         super().__init__()

#         if len(hidden_channels) == 0:
#             raise ValueError("hidden_channels must contain at least one stage")

#         self.use_phase_decoder = use_phase_decoder
#         self.num_joints = num_joints

#         adjacency = build_coco17_adjacency(include_self=True)
#         self.register_buffer("adjacency", adjacency, persistent=False)

#         # --- Shared ST-GAT encoder ---
#         blocks: list[STGATBlock] = []
#         c_in = in_channels
#         for c_out in hidden_channels:
#             blocks.append(STGATBlock(c_in, c_out, dropout=dropout))
#             c_in = c_out
#         self.encoder = nn.ModuleList(blocks)
#         final_channels = c_in

#         # --- Global embedding head ---
#         self.proj_head = nn.Sequential(
#             nn.Linear(final_channels, final_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(final_channels, embedding_dim),
#         )

#         # --- Phase-aware heads ---
#         if use_phase_decoder:
#             self.phase_aligner = PhaseAligner(
#                 channels=final_channels,
#                 num_joints=num_joints,
#                 dropout=dropout,
#             )
#             self.frame_decoder = FrameDecoder(
#                 channels=final_channels,
#                 num_joints=num_joints,
#                 dropout=dropout,
#             )

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def _encode_full(
#         self, x: torch.Tensor, return_attentions: bool = False
#     ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
#         attentions: list[torch.Tensor] = []
#         out = x
#         for block in self.encoder:
#             out, attn = block(out, self.adjacency)
#             if return_attentions:
#                 attentions.append(attn)
#         frame_features = out

#         pooled = out.mean(dim=2).mean(dim=2)
#         embedding = self.proj_head(pooled)
#         embedding = F.normalize(embedding, p=2, dim=-1)
#         return frame_features, embedding, attentions

#     def encode(
#         self, x: torch.Tensor, return_attentions: bool = False
#     ) -> tuple[torch.Tensor, list[torch.Tensor]]:
#         _, embedding, attentions = self._encode_full(x, return_attentions)
#         return embedding, attentions

#     @staticmethod
#     def _joint_importance_from_attention(attention: torch.Tensor) -> torch.Tensor:
#         return attention.mean(dim=1).mean(dim=-1)

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------

#     def forward(
#         self,
#         template_seq: torch.Tensor,
#         user_seq: torch.Tensor,
#         template_xyz_raw: torch.Tensor | None = None,
#     ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
#         """
#         Parameters
#         ----------
#         template_seq     : (B, C, T_t, J)  preprocessed features
#         user_seq         : (B, C, T_u, J)  preprocessed features
#         template_xyz_raw : (B, 3, T_t, J)  raw image-fraction XYZ, optional.
#             Pass this at inference time so warped_template_xyz comes back in
#             pixel-mappable [0,1] coordinates.  At training time leave as None.
#         """
#         template_feat, template_emb, _ = self._encode_full(
#             template_seq, return_attentions=False
#         )
#         user_feat, user_emb, user_attentions = self._encode_full(
#             user_seq, return_attentions=True
#         )

#         similarity_score = F.cosine_similarity(template_emb, user_emb, dim=-1)

#         result: dict[str, torch.Tensor | list[torch.Tensor]] = {
#             "similarity_score": similarity_score,
#             "template_embedding": template_emb,
#             "user_embedding": user_emb,
#             "user_attention_weights": user_attentions,
#         }

#         if self.use_phase_decoder:
#             # Phase alignment (in feature space — drives the loss)
#             warped_tmpl, warp_weights = self.phase_aligner(user_feat, template_feat)

#             # Per-frame correction deltas (preprocessed space — for delta loss)
#             delta_xyz, joint_conf = self.frame_decoder(user_feat, warped_tmpl)

#             # ----------------------------------------------------------------
#             # Warp the XYZ template into user time.
#             #
#             # If the caller supplied raw image-fraction XYZ (inference path),
#             # warp that — the result is in [0,1] and can be used directly as
#             # pixel fractions for the overlay skeleton.
#             #
#             # Otherwise fall back to template_seq[:, :3] (preprocessed XYZ,
#             # hip-centred & z-scored).  This is only useful for the delta loss;
#             # it is NOT in pixel space and will NOT render correctly as an
#             # overlay.  Training code takes this path and that's fine because
#             # training never draws skeletons.
#             # ----------------------------------------------------------------
#             if template_xyz_raw is not None:
#                 # template_xyz_raw : (B, 3, T_t_raw, J)
#                 # warp_weights     : (B, T_u, T_t_feat)  where T_t_feat comes
#                 # from the preprocessed template_seq (fixed by cfg.seq_length).
#                 # T_t_raw (median raw length) != T_t_feat in general, so we
#                 # must resample template_xyz_raw along the time axis to match.
#                 T_t_feat = template_feat.shape[2]   # what warp_weights indexes
#                 T_t_raw  = template_xyz_raw.shape[2]
#                 if T_t_raw != T_t_feat:
#                     # F.interpolate expects (B, C, T) for mode='linear'
#                     # template_xyz_raw is (B, 3, T_t_raw, J) — fold J into batch
#                     B, C, _, J = template_xyz_raw.shape
#                     tmp = template_xyz_raw.permute(0, 3, 1, 2).reshape(B * J, C, T_t_raw)
#                     tmp = F.interpolate(tmp, size=T_t_feat, mode="linear", align_corners=False)
#                     template_xyz_raw = tmp.reshape(B, J, C, T_t_feat).permute(0, 2, 3, 1)
#                     # shape is now (B, 3, T_t_feat, J)
#                 xyz_source = template_xyz_raw        # (B, 3, T_t_feat, J)
#             else:
#                 # Backward-compat training path
#                 xyz_source = template_seq[:, :3, :, :]  # (B, 3, T_t_feat, J)

#             xyz_t = xyz_source.permute(0, 2, 1, 3)      # (B, T_t_feat, 3, J)
#             warped_xyz = torch.einsum(
#                 "but,btcj->bucj", warp_weights, xyz_t
#             ).permute(0, 1, 3, 2).contiguous()           # (B, T_u, J, 3)

#             joint_importance = JointScorer.from_attention_and_decoder(
#                 user_attentions, delta_xyz.detach(), joint_conf.detach()
#             )

#             result.update({
#                 "warped_template_xyz": warped_xyz,             # (B, T_u, J, 3)
#                 "correction_delta": delta_xyz,                 # (B, T_u, J, 3) — preprocessed space
#                 "joint_confidence": joint_conf,                # (B, T_u, J)
#                 "warp_weights": warp_weights,                  # (B, T_u, T_t)
#                 "joint_importance": joint_importance,          # (B, J)
#                 "joint_error_magnitude": delta_xyz.norm(dim=-1).mean(dim=1),
#                 "joint_confidence_mean": joint_conf.mean(dim=1),
#             })
#         else:
#             last_attention = user_attentions[-1]
#             result["joint_importance"] = self._joint_importance_from_attention(last_attention)

#         return result


# # ---------------------------------------------------------------------------
# # Losses
# # ---------------------------------------------------------------------------

# class ContrastiveLoss(nn.Module):
#     """
#     Contrastive loss on siamese embeddings.

#     Label convention
#     ----------------
#     label = 1.0  →  similar pair  (correct phase matches template)
#     label = 0.0  →  dissimilar pair (incorrect phase)
#     """

#     def __init__(self, margin: float = 1.0):
#         super().__init__()
#         self.margin = margin

#     def forward(
#         self,
#         emb_a: torch.Tensor,
#         emb_b: torch.Tensor,
#         labels: torch.Tensor,
#     ) -> torch.Tensor:
#         labels = labels.float().view(-1)
#         distances = 1.0 - F.cosine_similarity(emb_a, emb_b, dim=-1)
#         positive = labels * distances.pow(2)
#         negative = (1.0 - labels) * F.relu(self.margin - distances).pow(2)
#         return (positive + negative).mean()


# class DeltaRegressionLoss(nn.Module):
#     """
#     Auxiliary loss to train the FrameDecoder.

#     Computes the Huber loss between the predicted correction delta and the
#     ground-truth delta (warped_template_xyz - user_xyz).

#     Only applied where the label is correct (label == 1).
#     """

#     def __init__(self, delta_weight: float = 0.1, huber_delta: float = 0.05):
#         super().__init__()
#         self.delta_weight = delta_weight
#         self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")

#     def forward(
#         self,
#         pred_delta: torch.Tensor,     # (B, T, J, 3)
#         warped_xyz: torch.Tensor,     # (B, T, J, 3)
#         user_seq_raw: torch.Tensor,   # (B, C, T, J)
#         labels: torch.Tensor,         # (B,)
#     ) -> torch.Tensor:
#         user_xyz = user_seq_raw[:, :3, :, :].permute(0, 2, 3, 1)  # (B, T, J, 3)
#         gt_delta = warped_xyz - user_xyz

#         loss_all = self.huber(pred_delta, gt_delta)  # (B, T, J, 3)
#         loss_all = loss_all.mean(dim=(1, 2, 3))      # (B,)

#         mask = labels.float().view(-1)
#         if mask.sum() < 1:
#             return torch.tensor(0.0, device=pred_delta.device, requires_grad=True)

#         return (loss_all * mask).sum() / mask.sum()

"""
ST-GAT with Temporal Pyramid and Phase-Aware Feedback.

Upgrade summary
---------------
- PhaseAligner: soft-DTW-based temporal alignment that warps user frames
  onto the template time axis so every frame gets a matching template pose.
- FrameDecoder: a lightweight per-frame head that predicts correction deltas
  (Δx, Δy, Δz per joint) relative to the phase-matched template.
- JointScorer: per-joint error magnitude and confidence derived from
  attention entropy and decoder deltas.
- ExerciseEvaluator keeps the same encode() / forward() signature.
  The new output dict is a superset of the old one, so existing training
  code that only reads similarity_score is unaffected.
- ContrastiveLoss is unchanged.

Backward compatibility
----------------------
- Old checkpoints (without the decoder / phase head) can still be loaded:
  call model.load_state_dict(ckpt, strict=False).  The decoder and phase
  head will be freshly initialised and will produce noisy deltas until the
  model is fine-tuned.  The similarity_score pathway is identical.

FIX (overlay bug)
-----------------
- forward() now accepts an optional `template_xyz_raw` argument
  (shape: (B, 3, T_t, J) in raw image-fraction space).
- When provided, warped_template_xyz is computed by warping this raw tensor
  through warp_weights instead of pulling from template_seq[:, :3].
- template_seq[:, :3] contains preprocessed (hip-centred, z-scored) XYZ —
  far outside [0,1] — so it was useless for pixel-space overlay.
- Training code does NOT pass template_xyz_raw → backward compatible.
- Inference code DOES pass it → overlay lands in frame.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------

def build_coco17_adjacency(include_self: bool = True) -> torch.Tensor:
    """
    Undirected adjacency matrix for the 17-joint COCO skeleton.

    Joint order:
    0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear,
    5:l_shoulder, 6:r_shoulder, 7:l_elbow, 8:r_elbow,
    9:l_wrist, 10:r_wrist, 11:l_hip, 12:r_hip,
    13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle
    """
    num_joints = 17
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    if include_self:
        adjacency.fill_diagonal_(1.0)

    return adjacency


# ---------------------------------------------------------------------------
# Spatial graph attention
# ---------------------------------------------------------------------------

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

        self.residual: nn.Module
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        if isinstance(self.residual, nn.Conv2d):
            nn.init.xavier_uniform_(self.residual.weight)
            if self.residual.bias is not None:
                nn.init.zeros_(self.residual.bias)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:         (B, C_in, T, J)
        adjacency: (J, J) — 1 for valid edges, 0 otherwise

        Returns
        -------
        out:       (B, C_out, T, J)
        attention: (B, T, J, J)
        """
        bsz, _, frames, joints = x.shape
        if adjacency.shape != (joints, joints):
            raise ValueError(
                f"Expected adjacency of shape {(joints, joints)}, got {tuple(adjacency.shape)}"
            )

        x_btjc = x.permute(0, 2, 3, 1).contiguous()   # (B, T, J, C_in)
        h = self.proj(x_btjc)                           # (B, T, J, C_out)

        src_scores = (h * self.attn_src).sum(dim=-1)    # (B, T, J)
        dst_scores = (h * self.attn_dst).sum(dim=-1)    # (B, T, J)
        logits = self.leaky_relu(
            src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2)
        )                                               # (B, T, J, J)

        edge_mask = (
            adjacency.to(device=x.device, dtype=torch.bool)
            .unsqueeze(0).unsqueeze(0)
        )
        logits = logits.masked_fill(~edge_mask, float("-inf"))
        attention = F.softmax(logits, dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("btij,btjf->btif", attention, h)  # (B, T, J, C_out)
        out = out.permute(0, 3, 1, 2).contiguous()           # (B, C_out, T, J)
        out = out + self.residual(x)

        return out, attention


# ---------------------------------------------------------------------------
# Temporal pyramid
# ---------------------------------------------------------------------------

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
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=(k, 1), dilation=(d, 1),
                        padding=padding, bias=False,
                    ),
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


# ---------------------------------------------------------------------------
# ST-GAT block
# ---------------------------------------------------------------------------

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

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_out, attn = self.spatial_attn(x, adjacency)
        spatial_out = self.activation(self.spatial_bn(spatial_out))

        temporal_out = self.temporal_pyramid(spatial_out)
        out = self.activation(self.out_bn(temporal_out) + spatial_out)
        return out, attn


# ---------------------------------------------------------------------------
# Phase aligner
# ---------------------------------------------------------------------------

class PhaseAligner(nn.Module):
    """
    Soft differentiable temporal alignment between a user sequence and a
    template sequence via a learnable attention-based warping.

    For every user frame t, it computes a soft weighted sum over template
    frames, acting like a differentiable DTW warp.  The warped template
    carries phase-matched reference poses back into user time.

    Input
    -----
    user_feat     : (B, C, T_u, J)  — per-frame ST-GAT features for the user
    template_feat : (B, C, T_t, J)  — same for the template

    Output
    ------
    warped_template : (B, C, T_u, J) — template features aligned to user time
    warp_weights    : (B, T_u, T_t)  — soft assignment matrix (for inspection)
    """

    def __init__(self, channels: int, num_joints: int = 17, dropout: float = 0.1):
        super().__init__()
        hidden = max(32, channels // 4)
        self.q_proj = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.scale = hidden ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        user_feat: torch.Tensor,
        template_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Queries from user, keys from template
        q = self.q_proj(user_feat)       # (B, H, T_u, J)
        k = self.k_proj(template_feat)   # (B, H, T_t, J)

        # Pool over joints → (B, H, T)
        q_t = q.mean(dim=-1)             # (B, H, T_u)
        k_t = k.mean(dim=-1)             # (B, H, T_t)

        # Attention: (B, T_u, T_t)
        logits = torch.einsum("bhu,bht->but", q_t, k_t) * self.scale
        warp_weights = F.softmax(logits, dim=-1)
        warp_weights = self.dropout(warp_weights)

        # Warp template features into user time
        tf = template_feat.permute(0, 2, 1, 3)          # (B, T_t, C, J)
        warped = torch.einsum("but,btcj->bucj", warp_weights, tf)  # (B, T_u, C, J)
        warped = warped.permute(0, 2, 1, 3).contiguous()           # (B, C, T_u, J)

        return warped, warp_weights


# ---------------------------------------------------------------------------
# Frame-level correction decoder
# ---------------------------------------------------------------------------

class FrameDecoder(nn.Module):
    """
    Per-frame, per-joint correction decoder.

    Given user ST-GAT features and the phase-warped template features,
    predicts:
      - delta_xyz  : (B, T, J, 3)  — predicted correction offset in XYZ
      - joint_conf : (B, T, J)     — per-joint confidence in [0, 1]

    The delta is in the same normalised coordinate space as the input
    features (i.e. after UIPRMDPreprocessor).
    """

    def __init__(self, channels: int, num_joints: int = 17, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(channels // 2, 3, kernel_size=1)   # Δxyz
        self.conf_head  = nn.Conv2d(channels // 2, 1, kernel_size=1)   # confidence logit

    def forward(
        self,
        user_feat: torch.Tensor,
        warped_tmpl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([user_feat, warped_tmpl], dim=1)  # (B, 2C, T, J)
        h = self.mlp(combined)                                  # (B, C/2, T, J)

        delta_xyz  = self.delta_head(h)                    # (B, 3, T, J)
        joint_conf = torch.sigmoid(self.conf_head(h))      # (B, 1, T, J)

        delta_xyz  = delta_xyz.permute(0, 2, 3, 1).contiguous()   # (B, T, J, 3)
        joint_conf = joint_conf.squeeze(1)                         # (B, T, J)

        return delta_xyz, joint_conf


# ---------------------------------------------------------------------------
# Joint importance scorer
# ---------------------------------------------------------------------------

class JointScorer(nn.Module):
    """
    Fuses attention entropy, frame-level delta magnitude, and confidence
    into a single per-joint importance score.
    """

    @staticmethod
    def from_attention_and_decoder(
        attention_stack: list[torch.Tensor],
        delta_xyz: torch.Tensor,
        joint_conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        attention_stack : list of (B, T, J, J)
        delta_xyz       : (B, T, J, 3)
        joint_conf      : (B, T, J)

        Returns
        -------
        joint_importance : (B, J)
        """
        attn_imp = torch.stack(
            [a.mean(dim=1).mean(dim=-1) for a in attention_stack], dim=0
        ).mean(dim=0)  # (B, J)

        last_attn = attention_stack[-1]
        safe_attn = last_attn.clamp(min=1e-9)
        entropy = -(safe_attn * safe_attn.log()).sum(dim=-1).mean(dim=1)  # (B, J)

        delta_mag = delta_xyz.norm(dim=-1).mean(dim=1)  # (B, J)
        inv_conf = (1.0 - joint_conf).mean(dim=1)       # (B, J)

        def _norm(t: torch.Tensor) -> torch.Tensor:
            mn = t.min(dim=-1, keepdim=True).values
            mx = t.max(dim=-1, keepdim=True).values
            return (t - mn) / (mx - mn + 1e-6)

        score = (
            0.30 * _norm(attn_imp) +
            0.20 * _norm(entropy)  +
            0.35 * _norm(delta_mag) +
            0.15 * _norm(inv_conf)
        )
        return score   # (B, J)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ExerciseEvaluator(nn.Module):
    """
    Siamese Temporal Pyramid ST-GAT with Phase-Aware Feedback.

    Inputs
    ------
    template_seq     : (B, C, T_t, J)  — canonical correct sequence (preprocessed)
    user_seq         : (B, C, T_u, J)  — user's attempt (preprocessed)
    template_xyz_raw : (B, 3, T_t, J)  — OPTIONAL raw image-fraction XYZ of
                       the template (NOT preprocessed).  When supplied,
                       warped_template_xyz is produced in image-fraction space
                       so the inference overlay skeleton lands inside the frame.
                       When None (default / training), the old behaviour is used
                       (warp template_seq[:,:3] — preprocessed coords, useful
                       only for the delta loss, not for visualisation).

    Outputs  (dict)
    -------
    similarity_score   : (B,)
    template_embedding : (B, embed_dim)
    user_embedding     : (B, embed_dim)

    # Phase-aware feedback
    warped_template_xyz : (B, T_u, J, 3)
        Image-fraction XYZ when template_xyz_raw is provided;
        preprocessed XYZ otherwise (backward compat).
    correction_delta    : (B, T_u, J, 3)  — predicted Δxyz (preprocessed space)
    joint_confidence    : (B, T_u, J)
    warp_weights        : (B, T_u, T_t)

    # Joint-level diagnostics
    user_attention_weights : list[(B, T, J, J)]
    joint_importance       : (B, J)
    joint_error_magnitude  : (B, J)
    joint_confidence_mean  : (B, J)
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: tuple[int, ...] = (64, 128),
        embedding_dim: int = 128,
        num_joints: int = 17,
        dropout: float = 0.1,
        use_phase_decoder: bool = True,
    ):
        super().__init__()

        if len(hidden_channels) == 0:
            raise ValueError("hidden_channels must contain at least one stage")

        self.use_phase_decoder = use_phase_decoder
        self.num_joints = num_joints

        adjacency = build_coco17_adjacency(include_self=True)
        self.register_buffer("adjacency", adjacency, persistent=False)

        # --- Shared ST-GAT encoder ---
        blocks: list[STGATBlock] = []
        c_in = in_channels
        for c_out in hidden_channels:
            blocks.append(STGATBlock(c_in, c_out, dropout=dropout))
            c_in = c_out
        self.encoder = nn.ModuleList(blocks)
        final_channels = c_in

        # --- Global embedding head ---
        self.proj_head = nn.Sequential(
            nn.Linear(final_channels, final_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_channels, embedding_dim),
        )

        # --- Phase-aware heads ---
        if use_phase_decoder:
            self.phase_aligner = PhaseAligner(
                channels=final_channels,
                num_joints=num_joints,
                dropout=dropout,
            )
            self.frame_decoder = FrameDecoder(
                channels=final_channels,
                num_joints=num_joints,
                dropout=dropout,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_full(
        self, x: torch.Tensor, return_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        attentions: list[torch.Tensor] = []
        out = x
        for block in self.encoder:
            out, attn = block(out, self.adjacency)
            if return_attentions:
                attentions.append(attn)
        frame_features = out

        pooled = out.mean(dim=2).mean(dim=2)
        embedding = self.proj_head(pooled)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return frame_features, embedding, attentions

    def encode(
        self, x: torch.Tensor, return_attentions: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        _, embedding, attentions = self._encode_full(x, return_attentions)
        return embedding, attentions

    @staticmethod
    def _joint_importance_from_attention(attention: torch.Tensor) -> torch.Tensor:
        return attention.mean(dim=1).mean(dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        template_seq: torch.Tensor,
        user_seq: torch.Tensor,
        template_xyz_raw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Parameters
        ----------
        template_seq     : (B, C, T_t, J)  preprocessed features
        user_seq         : (B, C, T_u, J)  preprocessed features
        template_xyz_raw : (B, 3, T_t, J)  raw image-fraction XYZ, optional.
            Pass this at inference time so warped_template_xyz comes back in
            pixel-mappable [0,1] coordinates.  At training time leave as None.
        """
        template_feat, template_emb, _ = self._encode_full(
            template_seq, return_attentions=False
        )
        user_feat, user_emb, user_attentions = self._encode_full(
            user_seq, return_attentions=True
        )

        similarity_score = F.cosine_similarity(template_emb, user_emb, dim=-1)

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "similarity_score": similarity_score,
            "template_embedding": template_emb,
            "user_embedding": user_emb,
            "user_attention_weights": user_attentions,
        }

        if self.use_phase_decoder:
            # Phase alignment (in feature space — drives the loss)
            warped_tmpl, warp_weights = self.phase_aligner(user_feat, template_feat)

            # Per-frame correction deltas (preprocessed space — for delta loss)
            delta_xyz, joint_conf = self.frame_decoder(user_feat, warped_tmpl)

            # ----------------------------------------------------------------
            # Warp the XYZ template into user time.
            #
            # If the caller supplied raw image-fraction XYZ (inference path),
            # warp that — the result is in [0,1] and can be used directly as
            # pixel fractions for the overlay skeleton.
            #
            # Otherwise fall back to template_seq[:, :3] (preprocessed XYZ,
            # hip-centred & z-scored).  This is only useful for the delta loss;
            # it is NOT in pixel space and will NOT render correctly as an
            # overlay.  Training code takes this path and that's fine because
            # training never draws skeletons.
            # ----------------------------------------------------------------
            if template_xyz_raw is not None:
                # template_xyz_raw : (B, 3, T_t_raw, J)
                # warp_weights     : (B, T_u, T_t_feat)  where T_t_feat comes
                # from the preprocessed template_seq (fixed by cfg.seq_length).
                # T_t_raw (median raw length) != T_t_feat in general, so we
                # must resample template_xyz_raw along the time axis to match.
                T_t_feat = template_feat.shape[2]   # what warp_weights indexes
                T_t_raw  = template_xyz_raw.shape[2]
                if T_t_raw != T_t_feat:
                    # F.interpolate expects (B, C, T) for mode='linear'
                    # template_xyz_raw is (B, 3, T_t_raw, J) — fold J into batch
                    B, C, _, J = template_xyz_raw.shape
                    tmp = template_xyz_raw.permute(0, 3, 1, 2).reshape(B * J, C, T_t_raw)
                    tmp = F.interpolate(tmp, size=T_t_feat, mode="linear", align_corners=False)
                    template_xyz_raw = tmp.reshape(B, J, C, T_t_feat).permute(0, 2, 3, 1)
                    # shape is now (B, 3, T_t_feat, J)
                xyz_source = template_xyz_raw        # (B, 3, T_t_feat, J)
            else:
                # Backward-compat training path
                xyz_source = template_seq[:, :3, :, :]  # (B, 3, T_t_feat, J)

            xyz_t = xyz_source.permute(0, 2, 1, 3)      # (B, T_t_feat, 3, J)
            warped_xyz = torch.einsum(
                "but,btcj->bucj", warp_weights, xyz_t
            ).permute(0, 1, 3, 2).contiguous()           # (B, T_u, J, 3)

            joint_importance = JointScorer.from_attention_and_decoder(
                user_attentions, delta_xyz.detach(), joint_conf.detach()
            )

            result.update({
                "warped_template_xyz": warped_xyz,             # (B, T_u, J, 3)
                "correction_delta": delta_xyz,                 # (B, T_u, J, 3) — preprocessed space
                "joint_confidence": joint_conf,                # (B, T_u, J)
                "warp_weights": warp_weights,                  # (B, T_u, T_t)
                "joint_importance": joint_importance,          # (B, J)
                "joint_error_magnitude": delta_xyz.norm(dim=-1).mean(dim=1),
                "joint_confidence_mean": joint_conf.mean(dim=1),
            })
        else:
            last_attention = user_attentions[-1]
            result["joint_importance"] = self._joint_importance_from_attention(last_attention)

        return result


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss on siamese embeddings.

    Label convention
    ----------------
    label = 1.0  →  similar pair  (correct phase matches template)
    label = 0.0  →  dissimilar pair (incorrect phase)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.float().view(-1)
        distances = 1.0 - F.cosine_similarity(emb_a, emb_b, dim=-1)
        positive = labels * distances.pow(2)
        negative = (1.0 - labels) * F.relu(self.margin - distances).pow(2)
        return (positive + negative).mean()


class DeltaRegressionLoss(nn.Module):
    """
    Auxiliary loss to train the FrameDecoder.

    Computes the Huber loss between the predicted correction delta and the
    ground-truth delta (warped_template_xyz - user_xyz).

    Only applied where the label is correct (label == 1).
    """

    def __init__(self, delta_weight: float = 0.1, huber_delta: float = 0.05):
        super().__init__()
        self.delta_weight = delta_weight
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")

    def forward(
        self,
        pred_delta: torch.Tensor,     # (B, T, J, 3)
        warped_xyz: torch.Tensor,     # (B, T, J, 3)
        user_seq_raw: torch.Tensor,   # (B, C, T, J)
        labels: torch.Tensor,         # (B,)
    ) -> torch.Tensor:
        user_xyz = user_seq_raw[:, :3, :, :].permute(0, 2, 3, 1)  # (B, T, J, 3)
        gt_delta = warped_xyz - user_xyz

        loss_all = self.huber(pred_delta, gt_delta)  # (B, T, J, 3)
        loss_all = loss_all.mean(dim=(1, 2, 3))      # (B,)

        mask = labels.float().view(-1)
        if mask.sum() < 1:
            return torch.tensor(0.0, device=pred_delta.device, requires_grad=True)

        return (loss_all * mask).sum() / mask.sum()