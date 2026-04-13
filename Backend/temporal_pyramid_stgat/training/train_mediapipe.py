#!/usr/bin/env python3
"""
Training script for MediaPipe 33-joint retraining.

This script retrains the Temporal Pyramid STGAT model using MediaPipe 33-landmark
representation on the UI-PRMD dataset. The resulting checkpoint is compatible with
video-based inference using MediaPipe pose extraction.

Usage:
    python train_mediapipe.py --exercise 0 --loso-subject 1
    python train_mediapipe.py  # Train on all exercises, no LOSO
"""

import sys
import argparse
import logging
from pathlib import Path

# Add Backend root so absolute imports like `temporal_pyramid_stgat.*` resolve
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from temporal_pyramid_stgat.config import PyramidSTGATConfig
from temporal_pyramid_stgat.training.train_triplet import PyramidSTGATTripleLossTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_mediapipe(exercise_id=None, loso_subject=None, val_mode="random"):
    """
    Train Temporal Pyramid STGAT with MediaPipe 33-joint representation.
    
    Args:
        exercise_id: Optional exercise ID to train single exercise
        loso_subject: Optional subject ID for Leave-One-Subject-Out validation
        val_mode: Validation mode ('random' or 'loso')
    """
    logger.info("=" * 70)
    logger.info("MediaPipe 33-Joint Retraining")
    logger.info("=" * 70)
    
    if loso_subject is not None:
        val_mode = "loso"
        logger.info(f"Using Leave-One-Subject-Out validation with subject {loso_subject}")
    
    # Create config for MediaPipe 33-joint training
    config = PyramidSTGATConfig.for_uiprmd_mediapipe_33joint(exercise_id)
    
    logger.info(f"Configuration:")
    logger.info(f"  - Num joints: {config.num_joints}")
    logger.info(f"  - In channels (coord): {config.in_channels_coord}")
    logger.info(f"  - In channels (angle): {config.in_channels_angle}")
    logger.info(f"  - Exercise ID: {config.exercise_id if config.exercise_id is not None else 'All'}")
    logger.info(f"  - Validation mode: {val_mode}")
    
    # Create trainer
    trainer = PyramidSTGATTripleLossTrainer(
        config,
        val_mode=val_mode,
        loso_subject=loso_subject
    )
    
    # Train
    logger.info("Starting training...")
    try:
        results = trainer.train()
        
        logger.info("=" * 70)
        logger.info("Training completed successfully!")
        logger.info("=" * 70)
        logger.info(f"Best validation loss: {results.get('best_val_loss', 'N/A')}")
        logger.info(f"Best validation accuracy: {results.get('best_val_acc', 'N/A')}")
        logger.info(f"Best-loss checkpoint: {results.get('best_loss_ckpt_path', 'N/A')}")
        logger.info(f"Best-accuracy checkpoint: {results.get('best_acc_ckpt_path', 'N/A')}")
        logger.info(f"Metrics saved: {results.get('metrics_csv_path', 'N/A')}")
        
        return results
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train Temporal Pyramid STGAT with MediaPipe 33-joint representation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on all exercises with random validation split
  python train_mediapipe.py
  
  # Train on single exercise using LOSO with subject 1
  python train_mediapipe.py --exercise 0 --loso-subject 1
  
  # Train on all exercises using LOSO with subject 5
  python train_mediapipe.py --loso-subject 5
        """
    )
    
    parser.add_argument(
        "--exercise",
        type=int,
        default=None,
        help="Exercise ID to train (None = train on all exercises)"
    )
    parser.add_argument(
        "--loso-subject",
        type=int,
        default=None,
        help="Subject ID for Leave-One-Subject-Out validation"
    )
    parser.add_argument(
        "--validation",
        choices=["random", "loso"],
        default="random",
        help="Validation mode (random split or LOSO)"
    )
    
    args = parser.parse_args()
    
    # If loso_subject is specified, override validation mode
    if args.loso_subject:
        args.validation = "loso"
    
    train_mediapipe(
        exercise_id=args.exercise,
        loso_subject=args.loso_subject,
        val_mode=args.validation
    )


if __name__ == "__main__":
    main()
