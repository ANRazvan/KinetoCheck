"""
Compute reference (average) keypoint sequences for "correct" samples.

This script loads all "correct" samples (label=0) for each exercise
and computes the mean keypoint trajectory. These references are used
for deviation analysis during inference.

Usage:
    python -m tools.compute_reference_keypoints

Output:
    weights/reference_exercise_0.npy
    weights/reference_exercise_1.npy
    ...
    weights/reference_exercise_8.npy
"""

import os
import sys
import numpy as np

# Ensure Backend is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import SkeletonDataset
from config import settings


def compute_reference_for_exercise(exercise_id: int) -> np.ndarray | None:
    """
    Compute average keypoint sequence for correct samples of one exercise.
    
    Returns:
        Array of shape (seq_length, num_keypoints, keypoint_dim) or None if no data.
    """
    print(f"\nExercise {exercise_id}: {settings.exercise_name(exercise_id)}")
    
    # Load all samples for this exercise
    dataset = SkeletonDataset(settings.DATA_DIR, exercise_id=exercise_id)
    if len(dataset) == 0:
        print("  ⚠ No samples found — skipping.")
        return None
    
    # Filter only correct samples (label=0)
    correct_samples = []
    for i in range(len(dataset)):
        x, y = dataset[i]  # x: (seq_len, num_kp, kp_dim), y: scalar
        if y.item() == 0:  # correct label
            correct_samples.append(x.numpy())
    
    if len(correct_samples) == 0:
        print("  ⚠ No correct samples found — skipping.")
        return None
    
    print(f"  Found {len(correct_samples)} correct samples (out of {len(dataset)} total)")
    
    # Compute mean across all correct samples
    reference = np.mean(correct_samples, axis=0)  # (seq_len, num_kp, kp_dim)
    print(f"  Reference shape: {reference.shape}")
    
    return reference


def main():
    print(f"Computing reference keypoints from: {settings.DATA_DIR}")
    print(f"Output directory: {settings.WEIGHTS_DIR}")
    print("=" * 60)
    
    os.makedirs(settings.WEIGHTS_DIR, exist_ok=True)
    
    for exercise_id in sorted(settings.EXERCISES.keys()):
        reference = compute_reference_for_exercise(exercise_id)
        
        if reference is not None:
            output_path = os.path.join(
                settings.WEIGHTS_DIR,
                f"reference_exercise_{exercise_id}.npy"
            )
            np.save(output_path, reference)
            print(f"  ✓ Saved → {output_path}")
    
    print("\n" + "=" * 60)
    print("Reference computation complete!")
    print("\nYou can now use deviation analysis in inference.")


if __name__ == "__main__":
    main()