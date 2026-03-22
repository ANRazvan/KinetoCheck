"""
Explainability service for movement analysis.

Provides:
- Joint deviation analysis (comparing user keypoints to reference)
- Video overlay with color-coded skeleton visualization
"""

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import settings


# COCO joint names (17 joints) - what YOLO extracts
COCO_JOINT_NAMES = [
    "Nose", "LeftEye", "RightEye", "LeftEar", "RightEar",
    "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
    "LeftWrist", "RightWrist", "LeftHip", "RightHip",
    "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle"
]

# COCO skeleton connections
COCO_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Mapping from COCO joint index to Kinect joint name (where applicable)
# Used to apply Kinect deviation scores to COCO skeleton visualization
COCO_TO_KINECT_MAPPING = {
    0: "Head",  # Nose -> Head (approx)
    5: "ShoulderLeft",
    6: "ShoulderRight",
    7: "ElbowLeft",
    8: "ElbowRight",
    9: "WristLeft",
    10: "WristRight",
    11: "HipLeft",
    12: "HipRight",
    13: "KneeLeft",
    14: "KneeRight",
    15: "AnkleLeft",
    16: "AnkleRight",
}

# Kinect joint names (25 joints) - what the model uses internally
KINECT_JOINT_NAMES = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft",
    "HandTipRight", "ThumbRight"
]

# Kinect skeleton connections
KINECT_SKELETON_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),  # spine + head
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # left arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # right arm
    (0, 12), (12, 13), (13, 14), (14, 15),  # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),  # right leg
]


class ExplainabilityService:
    """
    Analyze movement deviations and generate visual feedback.

    Implemented as a **Singleton**: the service caches loaded reference
    keypoints in ``self.references``; sharing one instance across all
    requests avoids reloading those ``.npy`` files on every call.
    """

    _instance: "ExplainabilityService | None" = None

    def __new__(cls, deviation_threshold: float = 0.15) -> "ExplainabilityService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, deviation_threshold: float = 0.15):
        # Guard: skip re-initialisation on subsequent calls.
        if hasattr(self, "deviation_threshold"):
            return
        self.deviation_threshold = deviation_threshold
        self.references: Dict[Tuple[str, int], np.ndarray] = {}  # (dataset, exercise_id) -> reference keypoints
    
    def load_reference(self, exercise_id: int, dataset: str = "intellirehab") -> np.ndarray | None:
        """Load reference keypoints for an exercise."""
        dataset_key = settings._normalize_dataset_key(dataset)
        cache_key = (dataset_key, exercise_id)

        if cache_key in self.references:
            return self.references[cache_key]

        # Prefer dataset-specific reference directory first.
        ref_paths = [
            os.path.join(
                settings.weights_dir_for(dataset_key),
                f"reference_exercise_{exercise_id}.npy",
            ),
            # Backward-compatible legacy location at weights root.
            os.path.join(settings.WEIGHTS_DIR, f"reference_exercise_{exercise_id}.npy"),
        ]

        ref_path = next((p for p in ref_paths if os.path.exists(p)), None)
        if ref_path is None:
            return None

        reference = np.load(ref_path)
        self.references[cache_key] = reference
        return reference
    
    def compute_deviations(
        self,
        user_keypoints: np.ndarray,
        exercise_id: int,
        dataset: str = "intellirehab",
    ) -> Dict[str, float] | None:
        """
        Compute per-joint deviation scores.
        
        Args:
            user_keypoints: (seq_len, num_keypoints, keypoint_dim)
            exercise_id: Exercise identifier
        
        Returns:
            Dictionary mapping joint names to deviation scores, or None if no reference.
        """
        reference = self.load_reference(exercise_id, dataset=dataset)
        if reference is None:
            return None
        
        # Ensure shapes match (pad/truncate if needed)
        if user_keypoints.shape[0] != reference.shape[0]:
            # Simple interpolation to match reference length
            from scipy.interpolate import interp1d
            seq_len_ref = reference.shape[0]
            seq_len_user = user_keypoints.shape[0]
            
            x_old = np.linspace(0, 1, seq_len_user)
            x_new = np.linspace(0, 1, seq_len_ref)
            
            # Interpolate each joint
            user_resampled = np.zeros_like(reference)
            for joint_idx in range(user_keypoints.shape[1]):
                for dim_idx in range(user_keypoints.shape[2]):
                    f = interp1d(x_old, user_keypoints[:, joint_idx, dim_idx], kind='linear')
                    user_resampled[:, joint_idx, dim_idx] = f(x_new)
            user_keypoints = user_resampled
        
        # Compute L2 distance per joint (averaged over time and normalized)
        diff = user_keypoints - reference  # (seq_len, num_joints, dim)
        deviations_per_frame = np.linalg.norm(diff, axis=2)  # (seq_len, num_joints)
        deviations_mean = np.mean(deviations_per_frame, axis=0)  # (num_joints,)
        
        # Normalize by overall std of reference
        ref_std = np.std(reference)
        if ref_std > 0:
            deviations_mean = deviations_mean / ref_std
        
        # Map to Kinect joint names (reference data is in Kinect format)
        result = {}
        num_joints = min(len(KINECT_JOINT_NAMES), len(deviations_mean))
        for i in range(num_joints):
            result[KINECT_JOINT_NAMES[i]] = float(deviations_mean[i])
        
        return result
    
    def get_problematic_joints(self, deviations: Dict[str, float]) -> List[str]:
        """Return list of joints exceeding the deviation threshold."""
        return [
            joint for joint, dev in deviations.items()
            if dev > self.deviation_threshold
        ]
    
    def create_annotated_video(
        self,
        video_path: str,
        keypoints_sequence: np.ndarray,
        deviations: Dict[str, float],
        output_path: str
    ) -> str:
        """
        Create a new video with color-coded skeleton overlay.
        
        Args:
            video_path: Input video path
            keypoints_sequence: (num_frames, num_keypoints, 2 or 3) - extracted keypoints
            deviations: Per-joint deviation scores
            output_path: Output video path
        
        Returns:
            Path to output video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try H.264 codec first (better browser support), fallback to mp4v
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Use 'isOpened' to check if the writer was successfully initialized.
            # Sometimes OpenCV doesn't throw, but just returns an unopened writer.
            if not out.isOpened():
                raise cv2.error("Failed to open avc1 writer")
                
        except (cv2.error, Exception):
            # Fallback to mp4v if avc1 is missing (common on Windows without OpenH264 DLL)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(keypoints_sequence):
                frame = self._draw_skeleton(
                    frame, keypoints_sequence[frame_idx], deviations, width, height
                )
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return output_path
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        deviations: Dict[str, float],
        frame_width: int,
        frame_height: int
    ) -> np.ndarray:
        """
        Draw color-coded skeleton on a single frame.
        
        Green = low deviation, Yellow = medium, Red = high deviation.
        """
        kpts = keypoints.copy()
        
        # If keypoints are 3D (x, y, z), project to 2D (just use x, y)
        if kpts.shape[-1] == 3:
            kpts = kpts[:, :2]
        
        # Detect skeleton type based on number of joints
        num_joints = len(kpts)
        is_coco = num_joints == 17
        is_kinect = num_joints == 25
        
        if is_coco:
            skeleton_connections = COCO_SKELETON_CONNECTIONS
            # Map Kinect deviations to COCO joint indices
            coco_deviations = {}
            for coco_idx, kinect_name in COCO_TO_KINECT_MAPPING.items():
                if kinect_name in deviations:
                    coco_deviations[coco_idx] = deviations[kinect_name]
        elif is_kinect:
            skeleton_connections = KINECT_SKELETON_CONNECTIONS
            # Create index-based deviations dict for Kinect
            coco_deviations = {i: deviations.get(KINECT_JOINT_NAMES[i], 0.0) 
                              for i in range(num_joints)}
        else:
            # Unknown skeleton format, skip drawing
            return frame
        
        # Draw connections (bones)
        for joint_a, joint_b in skeleton_connections:
            if joint_a >= len(kpts) or joint_b >= len(kpts):
                continue
            
            x1, y1 = int(kpts[joint_a, 0]), int(kpts[joint_a, 1])
            x2, y2 = int(kpts[joint_b, 0]), int(kpts[joint_b, 1])
            
            # Skip if coordinates are invalid
            if x1 == 0 and y1 == 0:
                continue
            if x2 == 0 and y2 == 0:
                continue
            
            # Color based on max deviation of the two joints
            dev_a = coco_deviations.get(joint_a, 0.0)
            dev_b = coco_deviations.get(joint_b, 0.0)
            max_dev = max(dev_a, dev_b)
            
            color = self._deviation_to_color(max_dev)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw joints (circles)
        for i, (x, y) in enumerate(kpts):
            x, y = int(x), int(y)
            if x == 0 and y == 0:
                continue
            
            deviation = coco_deviations.get(i, 0.0)
            color = self._deviation_to_color(deviation)
            cv2.circle(frame, (x, y), 5, color, -1)
            
            # Optionally, label highly problematic joints
            if deviation > self.deviation_threshold:
                joint_label = (COCO_JOINT_NAMES[i] if is_coco and i < len(COCO_JOINT_NAMES)
                              else KINECT_JOINT_NAMES[i] if is_kinect and i < len(KINECT_JOINT_NAMES)
                              else f"J{i}")
                cv2.putText(
                    frame, joint_label, (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
        
        return frame
    
    def _deviation_to_color(self, deviation: float) -> Tuple[int, int, int]:
        """
        Map deviation score to BGR color.
        
        Green (low) -> Yellow (medium) -> Red (high)
        """
        # Clamp deviation to [0, 2 * threshold]
        max_dev = 2 * self.deviation_threshold
        ratio = min(deviation / max_dev, 1.0)
        
        if ratio < 0.5:
            # Green to Yellow
            green = 255
            red = int(255 * (ratio * 2))
            blue = 0
        else:
            # Yellow to Red
            green = int(255 * (1 - (ratio - 0.5) * 2))
            red = 255
            blue = 0
        
        return (blue, green, red)  # OpenCV uses BGR