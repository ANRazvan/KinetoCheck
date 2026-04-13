"""
Utility functions for Temporal Pyramid STGAT inference and visualization.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import json


class QualityScoreInterpreter:
    """
    Interprets quality scores for different rehabilitation contexts.
    """
    
    # Standard thresholds
    THRESHOLDS = {
        'excellent': (0.90, 1.00),
        'good': (0.75, 0.90),
        'fair': (0.60, 0.75),
        'poor': (0.40, 0.60),
        'very_poor': (0.00, 0.40),
    }
    
    FEEDBACK = {
        'excellent': "Excellent form! Very close to reference movement.",
        'good': "Good form. Minor adjustments needed.",
        'fair': "Fair form. Notable differences from reference.",
        'poor': "Poor form. Significant improvement needed.",
        'very_poor': "Very poor form. Major corrections required.",
    }
    
    @classmethod
    def interpret(cls, score: float) -> Tuple[str, str]:
        """
        Interpret a quality score.
        
        Args:
            score: Quality score (0-1)
            
        Returns:
            (category, feedback_message)
        """
        for category, (min_score, max_score) in cls.THRESHOLDS.items():
            if min_score <= score < max_score:
                return category, cls.FEEDBACK[category]
        
        # Fallback (should not happen)
        if score >= 0.90:
            return 'excellent', cls.FEEDBACK['excellent']
        return 'very_poor', cls.FEEDBACK['very_poor']
    
    @staticmethod
    def get_recommendations(score: float) -> List[str]:
        """
        Get actionable recommendations based on quality score.
        
        Args:
            score: Quality score (0-1)
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if score >= 0.90:
            recommendations = [
                "Continue maintaining this excellent form",
                "Gradually increase exercise difficulty or repetitions",
            ]
        elif score >= 0.75:
            recommendations = [
                "Focus on minor timing adjustments",
                "Ensure smooth transitions between phases",
                "Check your posture and alignment",
            ]
        elif score >= 0.60:
            recommendations = [
                "Reduce exercise speed to improve form",
                "Focus on one problematic joint at a time",
                "Review the reference movement carefully",
                "Check your starting position",
            ]
        elif score >= 0.40:
            recommendations = [
                "Slow down significantly",
                "Practice near a mirror or with video feedback",
                "Review reference movement multiple times",
                "Consider modifying the exercise difficulty",
                "Ask instructor for form correction",
            ]
        else:
            recommendations = [
                "Return to basic form training with instructor",
                "Watch reference video carefully",
                "Practice slowly without added resistance",
                "Focus on understanding movement mechanics",
                "Seek professional guidance before continuing",
            ]
        
        return recommendations


class InferenceResultsLogger:
    """
    Logs inference results for analysis and auditing.
    """
    
    def __init__(self, output_file: str = None):
        """
        Args:
            output_file: Path to save results (JSON format)
        """
        self.results = []
        self.output_file = output_file
    
    def log(self, sample_id: str, result: Dict):
        """
        Log a single inference result.
        
        Args:
            sample_id: Identifier for the sample
            result: Output dict from inference
        """
        log_entry = {
            'sample_id': sample_id,
            'prediction': int(result['prediction']),
            'prediction_label': result['prediction_label'],
            'quality_score': float(result['quality_score']),
            'confidence': float(result['confidence']),
            'class_probabilities': [float(p) for p in result['class_probabilities']],
        }
        
        self.results.append(log_entry)
    
    def save(self):
        """Save logged results to file."""
        if self.output_file:
            with open(self.output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
    
    def summary(self) -> Dict:
        """
        Get summary statistics.
        
        Returns:
            Dict with aggregated metrics
        """
        if not self.results:
            return {}
        
        results_arr = np.array([r['quality_score'] for r in self.results])
        predictions = np.array([r['prediction'] for r in self.results])
        
        return {
            'total_samples': len(self.results),
            'correct_percentage': (predictions == 0).sum() / len(results_arr) * 100,
            'incorrect_percentage': (predictions == 1).sum() / len(results_arr) * 100,
            'mean_quality_score': float(np.mean(results_arr)),
            'std_quality_score': float(np.std(results_arr)),
            'min_quality_score': float(np.min(results_arr)),
            'max_quality_score': float(np.max(results_arr)),
        }


class EmbeddingVisualizer:
    """
    Tools for visualizing learned embeddings in 2D/3D.
    """
    
    @staticmethod
    def reduce_to_2d(embeddings: np.ndarray, method: str = "pca") -> np.ndarray:
        """
        Reduce embeddings from high-dim to 2D.
        
        Args:
            embeddings: (N, embedding_dim) array
            method: "pca" or "tsne"
            
        Returns:
            (N, 2) reduced embeddings
        """
        if method == "pca":
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                return pca.fit_transform(embeddings)
            except ImportError:
                raise ImportError("scikit-learn required for PCA")
        
        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42)
                return tsne.fit_transform(embeddings)
            except ImportError:
                raise ImportError("scikit-learn required for t-SNE")
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def compute_embedding_stats(embeddings: np.ndarray, 
                               labels: np.ndarray) -> Dict:
        """
        Compute statistics on embedding quality.
        
        Args:
            embeddings: (N, D) embeddings
            labels: (N,) class labels
            
        Returns:
            Dict with intra-class and inter-class distances
        """
        correct_embs = embeddings[labels == 0]
        incorrect_embs = embeddings[labels == 1]
        
        # Intra-class distances (should be small)
        intra_correct = np.mean([
            np.linalg.norm(correct_embs[i] - correct_embs[j])
            for i in range(len(correct_embs))
            for j in range(i+1, len(correct_embs))
        ]) if len(correct_embs) > 1 else 0.0
        
        intra_incorrect = np.mean([
            np.linalg.norm(incorrect_embs[i] - incorrect_embs[j])
            for i in range(len(incorrect_embs))
            for j in range(i+1, len(incorrect_embs))
        ]) if len(incorrect_embs) > 1 else 0.0
        
        # Inter-class distance (should be large)
        inter_distance = np.mean([
            np.linalg.norm(correct_embs[i] - incorrect_embs[j])
            for i in range(len(correct_embs))
            for j in range(len(incorrect_embs))
        ]) if len(correct_embs) > 0 and len(incorrect_embs) > 0 else 0.0
        
        return {
            'intra_correct_distance': float(intra_correct),
            'intra_incorrect_distance': float(intra_incorrect),
            'inter_class_distance': float(inter_distance),
            'separation_ratio': inter_distance / (intra_correct + intra_incorrect + 1e-6),
        }


class PerformanceAnalyzer:
    """
    Detailed performance analysis tools.
    """
    
    @staticmethod
    def compute_metrics(predictions: np.ndarray,
                       ground_truth: np.ndarray,
                       quality_scores: np.ndarray = None) -> Dict:
        """
        Compute classification and quality metrics.
        
        Args:
            predictions: (N,) predicted labels
            ground_truth: (N,) true labels
            quality_scores: (N,) quality scores (optional)
            
        Returns:
            Dict with metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': float(accuracy_score(ground_truth, predictions)),
            'precision': float(precision_score(ground_truth, predictions, zero_division=0)),
            'recall': float(recall_score(ground_truth, predictions, zero_division=0)),
            'f1': float(f1_score(ground_truth, predictions, zero_division=0)),
        }
        
        if quality_scores is not None:
            # Quality scores should be higher for correct predictions
            correct_mask = predictions == ground_truth
            incorrect_mask = predictions != ground_truth
            
            if correct_mask.sum() > 0:
                metrics['mean_quality_correct'] = float(quality_scores[correct_mask].mean())
            if incorrect_mask.sum() > 0:
                metrics['mean_quality_incorrect'] = float(quality_scores[incorrect_mask].mean())
        
        return metrics
