"""
Training script for Temporal Pyramid STGAT on UI-PRMD dataset.

Usage:
    python train.py --config pyramid_stgat --exercise 0
    python train.py --config pyramid_stgat --all-exercises
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_pyramid_stgat.config import PyramidSTGATConfig
from temporal_pyramid_stgat.preprocessing.uiprmd_loader import UIPRMDLoader, DatasetStatistics
from temporal_pyramid_stgat.preprocessing.angle_calculator import TwoStreamFeatures, AngleCalculator
from temporal_pyramid_stgat.preprocessing.temporal_pyramid import PyramidPreprocessor, TemporalPyramidSampler
from temporal_pyramid_stgat.models.pyramid_stgat import TemporalPyramidSTGAT


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PyramidSTGATTrainer:
    """End-to-end trainer for Temporal Pyramid STGAT."""
    
    def __init__(self, config: PyramidSTGATConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create save directory
        os.makedirs(config.model_save_dir, exist_ok=True)
        
        # Initialize model
        self.model = TemporalPyramidSTGAT(
            in_channels_coord=config.in_channels_coord,
            in_channels_angle=config.in_channels_angle,
            hidden_channels=config.hidden_channels,
            num_heads=config.num_heads,
            num_joints=config.num_joints,
            num_scales=config.num_scales,
            scales=config.temporal_scales,
            dropout=config.dropout,
            num_classes=config.num_classes,
            frame_head_type=config.frame_head_type,
            frame_head_hidden=config.frame_head_hidden,
            frame_aggregation=config.frame_aggregation,
            frame_topk_ratio=config.frame_topk_ratio,
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # AMP
        self.use_amp = config.use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def load_data(self):
        """Load and preprocess UI-PRMD data."""
        logger.info(f"Loading UI-PRMD dataset from {self.config.dataset_root}")
        
        loader = UIPRMDLoader(self.config.dataset_root)
        coords, labels, metadata = loader.load_all(self.config.exercise_id)
        
        logger.info(f"Loaded {coords.shape[0]} sequences, shape: {coords.shape}")
        logger.info(f"Label distribution: {DatasetStatistics.label_distribution(labels)}")
        
        # Extract both streams
        logger.info("Extracting two-stream features (coordinates + angles)...")
        all_coords = []
        all_angles = []
        
        for i, coord_seq in enumerate(coords):
            # Stream 1: Normalized coordinates
            norm_coords = TwoStreamFeatures._normalize_coordinates(coord_seq)
            all_coords.append(norm_coords)
            
            # Stream 2: Joint angles
            angles = AngleCalculator.extract_angles(coord_seq)
            angles = TwoStreamFeatures.standardize_angles(angles)
            all_angles.append(angles)
        
        coords_array = np.array(all_coords)  # (N, T, 17, 3)
        angles_array = np.array(all_angles)  # (N, T, num_angles)
        
        logger.info(f"Coordinates shape: {coords_array.shape}")
        logger.info(f"Angles shape: {angles_array.shape}")
        
        # Create pyramid
        logger.info("Creating temporal pyramid...")
        pyramid_preprocessor = PyramidPreprocessor(
            scales=self.config.temporal_scales,
            normalize=self.config.normalize_coords
        )
        
        # We'll handle pyramid in dataloader, so return raw data
        return {
            'coords': torch.from_numpy(coords_array).float(),
            'angles': torch.from_numpy(angles_array).float(),
            'labels': torch.from_numpy(labels).long()
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (coords, angles, labels) in enumerate(pbar):
            coords = coords.to(self.device)  # (B, T, J, 3)
            angles = angles.to(self.device)  # (B, T, num_angles)
            labels = labels.to(self.device)  # (B,)
            
            # Create pyramid
            pyramid = TemporalPyramidSampler.create_pyramid_batch(
                coords.cpu().numpy(),
                scales=self.config.temporal_scales
            )
            pyramid = {
                scale: torch.from_numpy(feat).float().to(self.device)
                for scale, feat in pyramid.items()
            }
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(pyramid, angles)
                    loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(pyramid, angles)
                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.2%}'
            })
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
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
                
                logits = self.model(pyramid, angles)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train(self):
        """Main training loop."""
        # Load data
        data = self.load_data()
        coords = data['coords']
        angles = data['angles']
        labels = data['labels']
        
        # Split train/val
        N = len(labels)
        val_size = int(N * self.config.validation_split)
        train_size = N - val_size
        
        train_indices = torch.randperm(N)[:train_size]
        val_indices = torch.randperm(N)[train_size:train_size + val_size]
        
        train_coords = coords[train_indices]
        train_angles = angles[train_indices]
        train_labels = labels[train_indices]
        
        val_coords = coords[val_indices]
        val_angles = angles[val_indices]
        val_labels = labels[val_indices]
        
        # Dataloaders
        train_data = TensorDataset(train_coords, train_angles, train_labels)
        val_data = TensorDataset(val_coords, val_angles, val_labels)
        
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Training set: {len(train_data)}, Validation set: {len(val_data)}")
        
        # Training loop
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint()
                logger.info(f"✓ Saved best model at epoch {epoch + 1}")
            
            # Early stopping
            if epoch - self.best_epoch >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2%} at epoch {self.best_epoch + 1}")
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        ckpt_path = os.path.join(
            self.config.model_save_dir,
            self.config.checkpoint_name
        )
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }, ckpt_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Temporal Pyramid STGAT on UI-PRMD"
    )
    parser.add_argument('--exercise', type=int, default=None,
                       help='Exercise ID (0-8), or None for all')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Config
    if args.exercise is not None:
        config = PyramidSTGATConfig.for_uiprmd_single_exercise(args.exercise)
    else:
        config = PyramidSTGATConfig.for_ui_prmd()
    
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.use_amp = not args.no_amp
    
    logger.info(f"Config:\n{config}")
    
    # Train
    trainer = PyramidSTGATTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
