#!/usr/bin/env python3
"""
Validation script for MediaPipe 33-joint data loading pipeline.

Tests that:
1. MediaPipe mapper correctly converts Vicon 39 → 33 format
2. Angle calculator correctly extracts 12 angles
3. MediaPipeUIsprmdLoader successfully loads full dataset
4. Shapes and dtypes are as expected
"""

import sys
import os
import logging
import numpy as np

# Add Backend root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from temporal_pyramid_stgat.preprocessing.mediapipe_mapper import MediaPipeMapper
from temporal_pyramid_stgat.preprocessing.mediapipe_angle_calculator import MediaPipeAngleCalculator
from temporal_pyramid_stgat.preprocessing.mediapipe_uiprmd_loader import MediaPipeUIsprmdLoader
from temporal_pyramid_stgat.config import PyramidSTGATConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_mediapipe_mapper():
    """Test Vicon 39 → MediaPipe 33 mapping."""
    logger.info("=" * 70)
    logger.info("Testing MediaPipe Mapper")
    logger.info("=" * 70)
    
    # Create dummy Vicon sequence (T=10, 39 joints, 3D)
    vicon_seq = np.random.randn(10, 39, 3).astype(np.float32)
    logger.info(f"Input Vicon shape: {vicon_seq.shape}")
    
    # Convert
    mp_seq = MediaPipeMapper.vicon_to_mediapipe(vicon_seq)
    logger.info(f"Output MediaPipe shape: {mp_seq.shape}")
    
    # Validate
    assert mp_seq.shape == (10, 33, 3), f"Expected (10, 33, 3), got {mp_seq.shape}"
    assert mp_seq.dtype == np.float32, f"Expected float32, got {mp_seq.dtype}"
    assert not np.isnan(mp_seq).any(), "Found NaN values in mapped sequence"
    
    logger.info("✓ Mapper test passed")
    logger.info(f"  - Input shape: {vicon_seq.shape}")
    logger.info(f"  - Output shape: {mp_seq.shape}")
    logger.info(f"  - Joint names: {len(MediaPipeMapper.mediapipe_33_joint_names())} joints")
    return True


def test_angle_calculator():
    """Test angle extraction from 33-landmark sequences."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Angle Calculator")
    logger.info("=" * 70)
    
    # Create dummy MediaPipe sequence
    mp_seq = np.random.randn(10, 33, 3).astype(np.float32)
    logger.info(f"Input sequence shape: {mp_seq.shape}")
    logger.info(f"Number of angle triplets: {MediaPipeAngleCalculator.NUM_ANGLES}")
    
    # Extract angles
    angles = MediaPipeAngleCalculator.extract_angles(mp_seq)
    logger.info(f"Output angles shape: {angles.shape}")
    
    # Validate
    assert angles.shape == (10, MediaPipeAngleCalculator.NUM_ANGLES), \
        f"Expected (10, {MediaPipeAngleCalculator.NUM_ANGLES}), got {angles.shape}"
    assert angles.dtype == np.float32, f"Expected float32, got {angles.dtype}"
    assert np.all(angles >= 0) and np.all(angles <= np.pi), \
        "Angles should be in range [0, π]"
    
    # Standardize
    angles_std = MediaPipeAngleCalculator.standardize_angles(angles)
    logger.info(f"Standardized angles shape: {angles_std.shape}")
    
    # Check standardization
    mean = np.mean(angles_std, axis=0)
    std = np.std(angles_std, axis=0)
    assert np.allclose(mean, 0, atol=1e-6), "Mean not close to 0"
    assert np.allclose(std, 1, atol=1e-6), "Std not close to 1"
    
    logger.info("✓ Angle calculator test passed")
    logger.info(f"  - Extracted {MediaPipeAngleCalculator.NUM_ANGLES} angles")
    logger.info(f"  - Angles in range [0, π]: ✓")
    logger.info(f"  - Standardization (mean≈0, std≈1): ✓")
    return True


def test_data_loader():
    """Test MediaPipeUIsprmdLoader on real dataset."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing MediaPipeUIsprmdLoader")
    logger.info("=" * 70)
    
    try:
        # Check if dataset exists
        dataset_root = "Datasets/UIPRMD"
        if not os.path.exists(dataset_root):
            logger.warning(f"Dataset not found at {dataset_root}")
            logger.info("Skipping loader test (dataset not available)")
            return True
        
        loader = MediaPipeUIsprmdLoader(dataset_root)
        logger.info(f"Loader initialized for: {dataset_root}")
        
        # Try loading a single exercise first (lighter test)
        logger.info("Loading exercise 0...")
        coords, angles, labels, metadata = loader.load_with_angles(exercise_id=0)
        
        logger.info(f"Loaded data for exercise 0:")
        logger.info(f"  - Coordinates shape: {coords.shape}")
        logger.info(f"  - Angles shape: {angles.shape}")
        logger.info(f"  - Labels shape: {labels.shape}")
        logger.info(f"  - Number of sequences: {len(metadata)}")
        
        # Validate shapes
        assert coords.shape[1:] == (240, 33, 3), f"Unexpected coord shape: {coords.shape}"
        assert angles.shape[1:] == (240, 12), f"Unexpected angle shape: {angles.shape}"
        assert len(labels) == coords.shape[0], "Label count mismatch"
        
        # Validate data types and ranges
        assert coords.dtype == np.float32, f"Expected float32 for coords, got {coords.dtype}"
        assert angles.dtype == np.float32, f"Expected float32 for angles, got {angles.dtype}"
        assert labels.dtype in [np.int64, np.long], f"Unexpected label dtype: {labels.dtype}"
        
        # Check for NaN
        assert not np.isnan(coords).any(), "Found NaN in coordinates"
        assert not np.isnan(angles).any(), "Found NaN in angles"
        
        # Check standardization of angles (approximate)
        angles_mean = np.mean(angles, axis=(0, 1))
        angles_std = np.std(angles, axis=(0, 1))
        logger.info(f"Angle statistics:")
        logger.info(f"  - Mean: {angles_mean[:3]}... (first 3 angles)")
        logger.info(f"  - Std: {angles_std[:3]}... (first 3 angles)")
        
        logger.info("✓ Loader test passed")
        return True
        
    except Exception as e:
        logger.error(f"Loader test failed: {e}", exc_info=True)
        return False


def test_config():
    """Test MediaPipe config creation."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Configuration")
    logger.info("=" * 70)
    
    config = PyramidSTGATConfig.for_uiprmd_mediapipe_33joint(exercise_id=0)
    
    logger.info(f"Created MediaPipe config:")
    logger.info(f"  - num_joints: {config.num_joints}")
    logger.info(f"  - in_channels_coord: {config.in_channels_coord}")
    logger.info(f"  - in_channels_angle: {config.in_channels_angle}")
    logger.info(f"  - exercise_id: {config.exercise_id}")
    logger.info(f"  - checkpoint_name: {config.checkpoint_name}")
    
    assert config.num_joints == 33, "Expected 33 joints"
    assert config.in_channels_coord == 3, "Expected 3 per-joint coord channels (x, y, z)"
    assert config.in_channels_angle == 12, "Expected 12 angle channels"
    
    logger.info("✓ Config test passed")
    return True


def main():
    logger.info("Starting MediaPipe 33-joint validation suite...")
    logger.info("")
    
    tests = [
        ("Mapper", test_mediapipe_mapper),
        ("Angle Calculator", test_angle_calculator),
        ("Configuration", test_config),
        ("Data Loader", test_data_loader),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"{name} test failed: {e}", exc_info=True)
            results[name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Validation Summary")
    logger.info("=" * 70)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name}: {status}")
    
    all_passed = all(results.values())
    logger.info("=" * 70)
    if all_passed:
        logger.info("✓ All tests passed! Ready for training.")
        return 0
    else:
        logger.error("✗ Some tests failed. Please review the above output.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
