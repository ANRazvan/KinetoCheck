"""
Inference script for Temporal Pyramid STGAT with interpretable feedback.

Demonstrates the three-level feedback system:
1. Binary Result: Correct/Incorrect classification
2. Quality Score (0-1): How far from correct (embedding distance-based)
3. Attention Maps: Where the errors occur (future enhancement)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

from temporal_pyramid_stgat.config import PyramidSTGATConfig
from temporal_pyramid_stgat.preprocessing.uiprmd_loader import UIPRMDLoader
from temporal_pyramid_stgat.preprocessing.angle_calculator import TwoStreamFeatures
from temporal_pyramid_stgat.preprocessing.mediapipe_angle_calculator import MediaPipeAngleCalculator
from temporal_pyramid_stgat.preprocessing.temporal_pyramid import TemporalPyramidSampler
from temporal_pyramid_stgat.models.pyramid_stgat import TemporalPyramidSTGAT

logger = logging.getLogger(__name__)


class PyramidSTGATInference:
    """Inference engine for Temporal Pyramid STGAT."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Args:
            checkpoint_path: Path to saved model checkpoint
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device)

        # Load checkpoint first so we can restore exact training-time dimensions.
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        ckpt_cfg = ckpt.get('config')
        if isinstance(ckpt_cfg, PyramidSTGATConfig):
            self.config = ckpt_cfg
        elif isinstance(ckpt_cfg, dict):
            self.config = PyramidSTGATConfig(**ckpt_cfg)
        else:
            self.config = PyramidSTGATConfig.for_ui_prmd()
        
        # Load model
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
        
        # Load checkpoint weights
        self.model.load_state_dict(ckpt['model_state'], strict=False)
        self.model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")
    
    def predict_from_sequence(self, 
                             skeleton_seq: np.ndarray) -> Dict[str, any]:
        """
        Get predictions from a single skeleton sequence.
        
        Args:
            skeleton_seq: (T, J, 3) array of joint positions
            
        Returns:
            Dict with:
            - 'prediction': 0=correct, 1=incorrect
            - 'quality_score': 0-1 (0=very wrong, 1=very correct)
            - 'confidence': probability of predicted class
            - 'embeddings': embedding vector
        """
        with torch.no_grad():
            # Extract features
            coords = self._normalize_coordinates(skeleton_seq)  # (T, J, 3)
            # Match angle-feature extraction to training configuration.
            if self.config.num_joints == 33 and self.config.in_channels_angle == MediaPipeAngleCalculator.NUM_ANGLES:
                angles = MediaPipeAngleCalculator.extract_angles(skeleton_seq)  # (T, 12)
                angles = MediaPipeAngleCalculator.standardize_angles(angles)
            else:
                angles = skeleton_seq.reshape(skeleton_seq.shape[0], -1)  # (T, J*3)
                angles = TwoStreamFeatures.standardize_angles(angles)
            
            # Create pyramid
            coords_batch = np.expand_dims(coords, 0)  # (1, T, J, 3)
            pyramid = TemporalPyramidSampler.create_pyramid_batch(
                coords_batch,
                scales=self.config.temporal_scales
            )
            pyramid = {
                scale: torch.from_numpy(feat).float().to(self.device)
                for scale, feat in pyramid.items()
            }
            
            # Inference
            angles_batch = torch.from_numpy(np.expand_dims(angles, 0)).float().to(self.device)
            output = self.model(pyramid, angles_batch, return_attention=False)
            
            # Extract outputs
            prediction = output['predictions'].cpu().item()
            logits = output['logits'].cpu()
            quality_score = output['distance_scores'].cpu().item()
            embeddings = output['embeddings'].cpu()
            
            # Confidence
            probs = torch.softmax(logits, dim=1)[0]
            confidence = probs[prediction].item()
            
            return {
                'prediction': prediction,
                'prediction_label': 'Correct' if prediction == 0 else 'Incorrect',
                'quality_score': quality_score,
                'confidence': confidence,
                'logits': logits.numpy()[0],
                'class_probabilities': probs.numpy(),
                'embeddings': embeddings.numpy()[0],
            }
    
    def predict_batch(self, 
                     sequences: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions for a batch of sequences.
        
        Args:
            sequences: (N, T, J, 3) array
            
        Returns:
            Dict with batch results
        """
        N = sequences.shape[0]
        predictions = []
        quality_scores = []
        confidences = []
        embeddings = []
        
        for i in range(N):
            result = self.predict_from_sequence(sequences[i])
            predictions.append(result['prediction'])
            quality_scores.append(result['quality_score'])
            confidences.append(result['confidence'])
            embeddings.append(result['embeddings'])
        
        return {
            'predictions': np.array(predictions),
            'quality_scores': np.array(quality_scores),
            'confidences': np.array(confidences),
            'embeddings': np.array(embeddings),
        }
    
    @staticmethod
    def _normalize_coordinates(skeleton_seq: np.ndarray, 
                              center_idx: int = 0) -> np.ndarray:
        """Normalize coordinates by centering on pelvis."""
        center = skeleton_seq[:, center_idx:center_idx+1, :]  # (T, 1, 3)
        return (skeleton_seq - center).astype(np.float32)
    
    def interpret_quality_score(self, score: float) -> str:
        """
        Interpret quality score (0-1) for user feedback.
        
        Args:
            score: Quality score from 0 to 1
            
        Returns:
            Interpretable feedback string
        """
        if score >= 0.9:
            return "Excellent form! Very close to reference movement."
        elif score >= 0.75:
            return "Good form. Minor adjustments needed."
        elif score >= 0.6:
            return "Fair form. Notable differences from reference."
        elif score >= 0.4:
            return "Poor form. Significant improvement needed."
        else:
            return "Very poor form. Major corrections required."


def demo_inference(checkpoint_path: str, dataset_root: str, exercise_id: int = 0):
    """
    Demo: Run inference on a few samples from the test set.
    """
    # Load inference model
    inference = PyramidSTGATInference(checkpoint_path, device="cuda")
    
    # Load test data
    loader = UIPRMDLoader(dataset_root)
    coords, labels, metadata = loader.load_all(exercise_id=exercise_id)
    
    # Take first 5 samples
    test_coords = coords[:5]
    test_labels = labels[:5]
    
    # Get predictions
    results = inference.predict_batch(test_coords)
    
    # Print results
    print("\n" + "="*70)
    print(f"Inference Results for Exercise {exercise_id}")
    print("="*70)
    
    for i in range(len(test_coords)):
        gt_label = "Correct" if test_labels[i] == 0 else "Incorrect"
        pred_label = results['predictions'][i]
        pred_label_str = "Correct" if pred_label == 0 else "Incorrect"
        quality = results['quality_scores'][i]
        confidence = results['confidences'][i]
        
        match = "✓" if test_labels[i] == pred_label else "✗"
        
        print(f"\nSample {i + 1} {match}")
        print(f"  Ground Truth:  {gt_label}")
        print(f"  Prediction:    {pred_label_str} (confidence: {confidence:.2%})")
        print(f"  Quality Score: {quality:.3f}")
        print(f"  Interpretation: {inference.interpret_quality_score(quality)}")
    
    # Summary metrics
    accuracy = (results['predictions'] == test_labels).mean()
    print(f"\n{'='*70}")
    print(f"Accuracy on this batch: {accuracy:.2%}")
    print(f"Mean quality score: {results['quality_scores'].mean():.3f}")
    print(f"Mean confidence: {results['confidences'].mean():.2%}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run inference with Temporal Pyramid STGAT"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='Datasets/UIPRMD',
                       help='Path to UI-PRMD dataset')
    parser.add_argument('--exercise', type=int, default=0,
                       help='Exercise ID to evaluate')
    
    args = parser.parse_args()
    
    demo_inference(args.checkpoint, args.dataset, args.exercise)
