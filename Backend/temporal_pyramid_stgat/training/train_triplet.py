"""
Training script with Triplet Loss for Temporal Pyramid STGAT.

Triplet Loss Strategy:
- Learns embeddings where correct exercises cluster together
- Incorrect exercises pushed away
- Distance between embeddings maps to quality score (0-1)

Interpretable Feedback:
- Binary result: Correct/Incorrect
- Score (0-1): How far from correct (distance-based)
- Attention maps: Where the errors occur (future enhancement)

Usage:
    python train_triplet.py --config pyramid_stgat --exercise 0
"""

import os
import sys
import argparse
import logging
import csv
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Tuple, Dict, Optional
import json

# Add Backend root so `temporal_pyramid_stgat` is importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from temporal_pyramid_stgat.config import PyramidSTGATConfig
from temporal_pyramid_stgat.preprocessing.uiprmd_loader import UIPRMDLoader, DatasetStatistics
from temporal_pyramid_stgat.preprocessing.angle_calculator import TwoStreamFeatures
from temporal_pyramid_stgat.preprocessing.temporal_pyramid import TemporalPyramidSampler
from temporal_pyramid_stgat.preprocessing.mediapipe_uiprmd_loader import MediaPipeUIsprmdLoader
from temporal_pyramid_stgat.preprocessing.mediapipe_angle_calculator import MediaPipeAngleCalculator
from temporal_pyramid_stgat.models.pyramid_stgat import TemporalPyramidSTGAT


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    Minimizes distance between anchor and positive (same class)
    while maximizing distance to negative (different class).
    """
    
    def __init__(self, margin: float = 1.0, distance_type: str = "l2"):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
    
    def forward(self, embeddings: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, embedding_dim) batch of embeddings
            labels: (B,) batch of labels (0=correct, 1=incorrect)
            
        Returns:
            loss: scalar triplet loss
        """
        # Compute pairwise distances
        distances = self._compute_distances(embeddings)  # (B, B)
        
        # For each sample, find hardest positive and negative
        loss = 0.0
        count = 0
        
        for i in range(embeddings.shape[0]):
            # Positives: same label
            pos_mask = (labels == labels[i]) & (torch.arange(len(labels), 
                                                 device=labels.device) != i)
            # Negatives: different label
            neg_mask = labels != labels[i]
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            # Hardest positive (largest distance among positives)
            pos_dist = distances[i][pos_mask].max()
            # Hardest negative (smallest distance among negatives)
            neg_dist = distances[i][neg_mask].min()
            
            # Triplet loss
            triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
            loss += triplet_loss
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return loss / count
    
    def _compute_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances."""
        if self.distance_type == "l2":
            # L2 distance: ||x - y||_2
            diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # (B, B, D)
            distances = torch.norm(diff, dim=2)  # (B, B)
        elif self.distance_type == "cosine":
            # Cosine distance: 1 - cos_similarity
            sim = F.cosine_similarity(embeddings.unsqueeze(1), 
                                     embeddings.unsqueeze(0), dim=2)  # (B, B)
            distances = 1 - sim
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return distances


class CombinedLoss(nn.Module):
    """
    Combined triplet loss + cross-entropy loss.
    
    λ_triplet * L_triplet + λ_ce * L_ce
    """
    
    def __init__(self, lambda_triplet: float = 0.5, 
                 lambda_ce: float = 0.5, margin: float = 1.0):
        super().__init__()
        self.triplet_loss = TripletLoss(margin=margin)
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_triplet = lambda_triplet
        self.lambda_ce = lambda_ce
    
    def forward(self, model_output: Dict[str, torch.Tensor], 
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model_output: Dict with 'embeddings' and 'logits'
            labels: (B,) ground truth labels
            
        Returns:
            loss: combined loss scalar
            loss_dict: Dict of individual losses for logging
        """
        embeddings = model_output['embeddings']
        logits = model_output['logits']
        
        # Triplet loss (learns good embedding space)
        l_triplet = self.triplet_loss(embeddings, labels)
        
        # Cross-entropy loss (ensures correct classification)
        l_ce = self.ce_loss(logits, labels)
        
        # Combined
        total_loss = self.lambda_triplet * l_triplet + self.lambda_ce * l_ce
        
        return total_loss, {
            'triplet': l_triplet.item(),
            'ce': l_ce.item(),
            'total': total_loss.item(),
        }


class PyramidSTGATTripleLossTrainer:
    """Trainer for Temporal Pyramid STGAT with triplet loss."""
    
    def __init__(self, config: PyramidSTGATConfig,
                 val_mode: str = "random",
                 loso_subject: Optional[int] = None):
        self.config = config
        self.val_mode = val_mode
        self.loso_subject = loso_subject
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        os.makedirs(config.model_save_dir, exist_ok=True)

        # Lazy init after inspecting dataset feature dimensions
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Loss
        self.criterion = CombinedLoss(
            lambda_triplet=0.6,
            lambda_ce=0.4,
            margin=1.0
        )
        
        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc_for_loss_ckpt = 0.0
        self.best_epoch_loss = 0
        self.best_val_acc = 0.0
        self.best_epoch_acc = 0

    def _initialize_model(self, num_joints: int, angle_dim: int):
        """Build model and optimizer after dataset dimensions are known."""
        self.config.num_joints = num_joints
        self.config.in_channels_angle = angle_dim

        self.model = TemporalPyramidSTGAT(
            in_channels_coord=self.config.in_channels_coord,
            in_channels_angle=self.config.in_channels_angle,
            hidden_channels=self.config.hidden_channels,
            num_heads=self.config.num_heads,
            num_joints=self.config.num_joints,
            num_scales=self.config.num_scales,
            scales=self.config.temporal_scales,
            dropout=self.config.dropout,
            embedding_dim=128,
            use_triplet_loss=True,
            frame_head_type=self.config.frame_head_type,
            frame_head_hidden=self.config.frame_head_hidden,
            frame_aggregation=self.config.frame_aggregation,
            frame_topk_ratio=self.config.frame_topk_ratio,
        ).to(self.device)

        logger.info(
            f"Model input dims -> joints: {self.config.num_joints}, "
            f"stream2_features: {self.config.in_channels_angle}"
        )
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6,
        )

    def _resolve_dataset_root(self) -> Path:
        """Resolve dataset root for both Backend cwd and repo-root cwd invocations."""
        configured = Path(self.config.dataset_root)
        candidates = [
            configured,
            # Common case when running from Backend and dataset lives at repo-root Datasets/...
            (Path.cwd().parent / configured),
            # Relative to this file: Backend/temporal_pyramid_stgat/training -> repo root
            (Path(__file__).resolve().parents[3] / configured),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        checked = "\n".join(str(c.resolve()) for c in candidates)
        raise FileNotFoundError(
            f"Could not find dataset root '{self.config.dataset_root}'. Checked:\n{checked}"
        )
    
    def load_data(self):
        """Load and preprocess data."""
        dataset_root = self._resolve_dataset_root()
        logger.info(f"Loading UI-PRMD dataset from {dataset_root}")
        
        # Determine if using MediaPipe 33-joint format
        use_mediapipe = self.config.num_joints == 33
        
        if use_mediapipe:
            logger.info("Using MediaPipe 33-joint representation")
            loader = MediaPipeUIsprmdLoader(str(dataset_root))
            coords, angles, labels, metadata = loader.load_with_angles(self.config.exercise_id)
        else:
            logger.info("Using standard UIPRMD format")
            loader = UIPRMDLoader(str(dataset_root))
            coords, labels, metadata = loader.load_all(self.config.exercise_id)
        
        logger.info(f"Loaded {coords.shape[0]} sequences, shape: {coords.shape}")
        logger.info(f"Label distribution: {DatasetStatistics.label_distribution(labels)}")
        
        if use_mediapipe:
            # For MediaPipe: we already have extracted angles from mediapipe_angle_calculator
            # Just normalize coordinates
            logger.info("Normalizing MediaPipe coordinates...")
            all_coords = []
            all_angles = []
            
            for i, coord_seq in enumerate(tqdm(coords, desc="Normalizing", leave=False)):
                # Stream 1: Normalized coordinates
                norm_coords = TwoStreamFeatures._normalize_coordinates(coord_seq)
                all_coords.append(norm_coords)
                all_angles.append(angles[i])  # Already standardized from angle_calculator
            
            coords_array = np.array(all_coords)  # (N, T, 33, 3)
            angles_array = np.array(all_angles)  # (N, T, 12)
        else:
            # Extract two-stream features (original behavior)
            logger.info("Extracting two-stream features (coordinates + full-joint features)...")
            all_coords = []
            all_angles = []
            
            for i, coord_seq in enumerate(tqdm(coords, desc="Extracting features", leave=False)):
                # Stream 1: Normalized coordinates
                norm_coords = TwoStreamFeatures._normalize_coordinates(coord_seq)
                all_coords.append(norm_coords)
                
                # Stream 2: all available joint/features flattened (T, J*3)
                full_features = coord_seq.reshape(coord_seq.shape[0], -1)
                full_features = TwoStreamFeatures.standardize_angles(full_features)
                all_angles.append(full_features)
            
            coords_array = np.array(all_coords)
            angles_array = np.array(all_angles)
        
        logger.info(f"Coordinates shape: {coords_array.shape}")
        logger.info(f"Feature stream shape: {angles_array.shape}")
        
        subject_ids = np.array([m.get('subject_id', -1) for m in metadata], dtype=np.int64)

        return {
            'coords': torch.from_numpy(coords_array).float(),
            'angles': torch.from_numpy(angles_array).float(),
            'labels': torch.from_numpy(labels).long(),
            'subject_ids': torch.from_numpy(subject_ids).long(),
        }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            (avg_loss, accuracy, triplet_loss_avg)
        """
        self.model.train()
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_ce_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for coords, angles, labels in pbar:
            coords = coords.to(self.device)
            angles = angles.to(self.device)
            labels = labels.to(self.device)
            
            # Create pyramid
            pyramid = TemporalPyramidSampler.create_pyramid_batch(
                coords.cpu().numpy(),
                scales=self.config.temporal_scales
            )
            pyramid = {
                scale: torch.from_numpy(feat).float().to(self.device)
                for scale, feat in pyramid.items()
            }
            
            # Forward
            output = self.model(pyramid, angles)
            loss, loss_dict = self.criterion(output, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_triplet_loss += loss_dict['triplet']
            total_ce_loss += loss_dict['ce']
            preds = output['predictions']
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'triplet': f'{loss_dict["triplet"]:.4f}',
                'acc': f'{correct/total:.2%}'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'triplet': total_triplet_loss / len(train_loader),
            'ce': total_ce_loss / len(train_loader),
            'acc': correct / total,
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_ce_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            for coords, angles, labels in pbar:
                coords = coords.to(self.device)
                angles = angles.to(self.device)
                labels = labels.to(self.device)
                
                # Create pyramid
                pyramid = TemporalPyramidSampler.create_pyramid_batch(
                    coords.cpu().numpy(),
                    scales=self.config.temporal_scales
                )
                pyramid = {
                    scale: torch.from_numpy(feat).float().to(self.device)
                    for scale, feat in pyramid.items()
                }
                
                output = self.model(pyramid, angles)
                loss, loss_dict = self.criterion(output, labels)
                
                total_loss += loss.item()
                total_triplet_loss += loss_dict['triplet']
                total_ce_loss += loss_dict['ce']
                preds = output['predictions']
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'triplet': total_triplet_loss / len(val_loader),
            'ce': total_ce_loss / len(val_loader),
            'acc': correct / total,
        }

    @staticmethod
    def _load_existing_metric(ckpt_path: str, metric_key: str) -> Optional[float]:
        """Read metric from existing checkpoint to prevent regressions across runs."""
        if not os.path.exists(ckpt_path):
            return None
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            value = ckpt.get(metric_key)
            return float(value) if value is not None else None
        except Exception:
            return None

    def _save_checkpoint_if_improved(self,
                                     ckpt_path: str,
                                     metric_key: str,
                                     metric_value: float,
                                     better: str,
                                     epoch: int,
                                     train_stats: Dict[str, float],
                                     val_stats: Dict[str, float]) -> bool:
        """Save checkpoint only if better than both current-run and existing checkpoint."""
        existing_metric = self._load_existing_metric(ckpt_path, metric_key)
        improved_vs_existing = (
            existing_metric is None or
            (metric_value < existing_metric if better == 'min' else metric_value > existing_metric)
        )
        if not improved_vs_existing:
            logger.info(
                f"Skip saving {os.path.basename(ckpt_path)}: "
                f"existing {metric_key}={existing_metric:.4f} is better than new {metric_value:.4f}"
            )
            return False

        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'metric_key': metric_key,
            metric_key: metric_value,
            'train_stats': train_stats,
            'val_stats': val_stats,
            # keep compatibility keys
            'best_val_loss': val_stats.get('loss'),
            'best_val_acc': val_stats.get('acc'),
        }, ckpt_path)
        return True
    
    def train(self):
        """Main training loop."""
        # Load data
        data = self.load_data()
        coords = data['coords']
        angles = data['angles']
        labels = data['labels']
        subject_ids = data['subject_ids']

        # Initialize model with inferred dimensions
        inferred_joints = coords.shape[2]
        inferred_feature_dim = angles.shape[2]
        self._initialize_model(inferred_joints, inferred_feature_dim)
        
        # Split
        N = len(labels)
        if self.val_mode == 'loso':
            if self.loso_subject is None:
                raise ValueError("LOSO mode requires --loso-subject")
            val_mask = subject_ids == int(self.loso_subject)
            train_mask = ~val_mask
            val_indices = torch.nonzero(val_mask, as_tuple=False).squeeze(1)
            train_indices = torch.nonzero(train_mask, as_tuple=False).squeeze(1)
            if len(val_indices) == 0:
                raise ValueError(f"No samples found for LOSO subject {self.loso_subject}")
            if len(train_indices) == 0:
                raise ValueError("No training samples remain after LOSO split")
            logger.info(
                f"Validation mode: LOSO (subject={self.loso_subject}) | "
                f"train={len(train_indices)}, val={len(val_indices)}"
            )
        else:
            val_size = int(N * self.config.validation_split)
            train_size = N - val_size
            indices = torch.randperm(N)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            logger.info(
                f"Validation mode: random split ({self.config.validation_split:.0%}) | "
                f"train={len(train_indices)}, val={len(val_indices)}"
            )
        
        train_data = TensorDataset(coords[train_indices], 
                                   angles[train_indices], 
                                   labels[train_indices])
        val_data = TensorDataset(coords[val_indices], 
                                 angles[val_indices], 
                                 labels[val_indices])
        
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, 
                                 shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, 
                               shuffle=False)
        
        logger.info(f"Training set: {len(train_data)}, Validation set: {len(val_data)}")

        base_ckpt = os.path.join(self.config.model_save_dir, self.config.checkpoint_name)
        root, ext = os.path.splitext(base_ckpt)
        best_loss_path = f"{root}_best_loss{ext}"
        best_acc_path = f"{root}_best_acc{ext}"
        
        # Metrics CSV path (one row per epoch)
        metrics_csv_path = f"{root}_metrics.csv"
        epoch_rows = []

        # Training loop
        for epoch in range(self.config.epochs):
            train_stats = self.train_epoch(train_loader)
            val_stats = self.validate(val_loader)
            
            self.scheduler.step()
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_stats['loss']:.4f} "
                f"(CE: {train_stats['ce']:.4f}, Triplet: {train_stats['triplet']:.4f}), "
                f"Acc: {train_stats['acc']:.2%} | "
                f"Val Loss: {val_stats['loss']:.4f} "
                f"(CE: {val_stats['ce']:.4f}, Triplet: {val_stats['triplet']:.4f}), "
                f"Acc: {val_stats['acc']:.2%}"
            )

            epoch_rows.append({
                'epoch': epoch + 1,
                'train_loss': train_stats['loss'],
                'train_ce': train_stats['ce'],
                'train_triplet': train_stats['triplet'],
                'train_acc': train_stats['acc'],
                'val_loss': val_stats['loss'],
                'val_ce': val_stats['ce'],
                'val_triplet': val_stats['triplet'],
                'val_acc': val_stats['acc'],
                'lr': self.optimizer.param_groups[0]['lr'],
            })

            # Best by validation loss
            if val_stats['loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['loss']
                self.best_val_acc_for_loss_ckpt = val_stats['acc']
                self.best_epoch_loss = epoch
                saved = self._save_checkpoint_if_improved(
                    best_loss_path,
                    metric_key='best_val_loss',
                    metric_value=val_stats['loss'],
                    better='min',
                    epoch=epoch + 1,
                    train_stats=train_stats,
                    val_stats=val_stats,
                )
                if saved:
                    logger.info("✓ Saved best-loss checkpoint")

            # Best by validation accuracy
            if val_stats['acc'] > self.best_val_acc:
                self.best_val_acc = val_stats['acc']
                self.best_epoch_acc = epoch
                saved = self._save_checkpoint_if_improved(
                    best_acc_path,
                    metric_key='best_val_acc',
                    metric_value=val_stats['acc'],
                    better='max',
                    epoch=epoch + 1,
                    train_stats=train_stats,
                    val_stats=val_stats,
                )
                if saved:
                    logger.info("✓ Saved best-accuracy checkpoint")
            
            # Early stopping
            if epoch - self.best_epoch_loss >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        logger.info(
            f"Best by loss: epoch {self.best_epoch_loss + 1}, "
            f"val_loss={self.best_val_loss:.4f}, "
            f"val_acc={self.best_val_acc_for_loss_ckpt:.2%}"
        )
        logger.info(
            f"Best by accuracy: epoch {self.best_epoch_acc + 1}, "
            f"val_acc={self.best_val_acc:.2%}"
        )

        # Persist epoch metrics CSV
        with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()) if epoch_rows else [
                'epoch', 'train_loss', 'train_ce', 'train_triplet', 'train_acc',
                'val_loss', 'val_ce', 'val_triplet', 'val_acc', 'lr'
            ])
            writer.writeheader()
            if epoch_rows:
                writer.writerows(epoch_rows)
        logger.info(f"Saved epoch metrics CSV: {metrics_csv_path}")

        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc_for_loss_ckpt': self.best_val_acc_for_loss_ckpt,
            'best_epoch_loss': self.best_epoch_loss + 1,
            'best_val_acc': self.best_val_acc,
            'best_epoch_acc': self.best_epoch_acc + 1,
            'best_loss_ckpt_path': best_loss_path,
            'best_acc_ckpt_path': best_acc_path,
            'metrics_csv_path': metrics_csv_path,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train Temporal Pyramid STGAT with Triplet Loss"
    )
    parser.add_argument('--exercise', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-mode', type=str, default='random', choices=['random', 'loso'])
    parser.add_argument('--loso-subject', type=int, default=None,
                        help='Subject ID for LOSO validation (required when --val-mode loso)')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config = (PyramidSTGATConfig.for_uiprmd_single_exercise(args.exercise)
              if args.exercise is not None
              else PyramidSTGATConfig.for_ui_prmd())
    
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    
    logger.info(f"Config:\n{config}")
    
    trainer = PyramidSTGATTripleLossTrainer(
        config,
        val_mode=args.val_mode,
        loso_subject=args.loso_subject,
    )
    trainer.train()


if __name__ == '__main__':
    main()
