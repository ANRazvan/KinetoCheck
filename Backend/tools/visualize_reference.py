"""
Visualize reference keypoints as a video with skeleton overlay.

This creates a video showing the "ideal" movement for an exercise based on
the averaged reference keypoints from correct samples.

Usage:
    python -m tools.visualize_reference --exercise_id 0
    python -m tools.visualize_reference --exercise_id 0 --output reference_ex0.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings


# Kinect joint names (25 joints)
JOINT_NAMES = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft",
    "HandTipRight", "ThumbRight"
]

# Kinect skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),  # spine + head
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # left arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # right arm
    (0, 12), (12, 13), (13, 14), (14, 15),  # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),  # right leg
]


def normalize_to_screen(keypoints: np.ndarray, width: int, height: int, padding: int = 50) -> np.ndarray:
    """
    Normalize 3D keypoints to 2D screen coordinates.
    
    Args:
        keypoints: (num_frames, num_joints, 3) array
        width, height: Video dimensions
        padding: Padding around skeleton
    
    Returns:
        (num_frames, num_joints, 2) array in pixel coordinates
    """
    # Use only X and Y (ignore Z depth)
    keypoints_2d = keypoints[:, :, :2].copy()
    
    # Flip Y axis (Kinect Y goes up, screen Y goes down)
    keypoints_2d[:, :, 1] = -keypoints_2d[:, :, 1]
    
    # Find min/max across all frames for consistent scaling
    x_min, y_min = keypoints_2d[:, :, 0].min(), keypoints_2d[:, :, 1].min()
    x_max, y_max = keypoints_2d[:, :, 0].max(), keypoints_2d[:, :, 1].max()
    
    # Scale to fit within video dimensions with padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if x_range == 0 or y_range == 0:
        return keypoints_2d
    
    scale = min((width - 2 * padding) / x_range, (height - 2 * padding) / y_range)
    
    # Center in frame
    keypoints_2d[:, :, 0] = (keypoints_2d[:, :, 0] - x_min) * scale + padding
    keypoints_2d[:, :, 1] = (keypoints_2d[:, :, 1] - y_min) * scale + padding
    
    return keypoints_2d


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Draw skeleton on frame with given color (BGR)."""
    kpts = keypoints.copy()
    
    # Draw connections (bones)
    for joint_a, joint_b in SKELETON_CONNECTIONS:
        if joint_a >= len(kpts) or joint_b >= len(kpts):
            continue
        
        x1, y1 = int(kpts[joint_a, 0]), int(kpts[joint_a, 1])
        x2, y2 = int(kpts[joint_b, 0]), int(kpts[joint_b, 1])
        
        # Skip invalid coordinates
        if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
            continue
        
        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw joints (circles)
    for i, (x, y) in enumerate(kpts):
        x, y = int(x), int(y)
        if x <= 0 or y <= 0:
            continue
        
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)  # white border
    
    return frame


def create_reference_video(
    reference_path: str,
    output_path: str,
    exercise_id: int,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    bg_color=(0, 0, 0)  # Black background
):
    """
    Create a video visualizing reference keypoints.
    
    Args:
        reference_path: Path to .npy file with reference keypoints
        output_path: Output video path
        exercise_id: Exercise ID for labeling
        width, height: Video dimensions
        fps: Frames per second
        bg_color: Background color (BGR)
    """
    # Load reference keypoints
    print(f"Loading reference from: {reference_path}")
    reference = np.load(reference_path)  # (num_frames, num_joints, 3)
    
    print(f"Reference shape: {reference.shape}")
    num_frames, num_joints, num_dims = reference.shape
    
    # Normalize to screen coordinates
    keypoints_2d = normalize_to_screen(reference, width, height)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Failed to create video writer: {output_path}")
    
    print(f"Creating video with {num_frames} frames @ {fps} fps...")
    
    # Draw each frame
    for frame_idx in range(num_frames):
        # Create blank frame
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Draw skeleton
        frame = draw_skeleton(frame, keypoints_2d[frame_idx], color=(0, 255, 0))
        
        # Add labels
        cv2.putText(
            frame, f"Reference Exercise {exercise_id}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"Frame {frame_idx + 1}/{num_frames}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )
        cv2.putText(
            frame, "CORRECT FORM (averaged from training data)", (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1
        )
        
        out.write(frame)
    
    out.release()
    print(f"✓ Video saved to: {output_path}")
    print(f"  Duration: {num_frames / fps:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Visualize reference keypoints as video")
    parser.add_argument("--exercise_id", type=int, required=True, help="Exercise ID (0-8)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    
    # Build paths
    reference_path = os.path.join(
        settings.WEIGHTS_DIR,
        f"reference_exercise_{args.exercise_id}.npy"
    )
    
    if not os.path.exists(reference_path):
        print(f"❌ Reference file not found: {reference_path}")
        print(f"   Run: python -m tools.compute_reference_keypoints")
        sys.exit(1)
    
    output_path = args.output or os.path.join(
        settings.WEIGHTS_DIR,
        f"reference_exercise_{args.exercise_id}_visualization.mp4"
    )
    
    # Create video
    create_reference_video(
        reference_path,
        output_path,
        args.exercise_id,
        args.width,
        args.height,
        args.fps
    )


if __name__ == "__main__":
    main()
