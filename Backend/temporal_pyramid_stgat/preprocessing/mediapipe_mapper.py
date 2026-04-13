"""
Map between Vicon 39-joint format (UI-PRMD) and MediaPipe 33-landmark format.

MediaPipe 33 Landmarks:
  Head: 0=Nose, 1-4=Eyes(inner/outer), 5-10=Ears
  Torso: 11-12=Shoulders, 13-14=Elbows, 15-16=Wrists, 17-18=Palms, 19-20=Pinkies
         21-22=Index, 23-24=Thumbs
  Lower: 25-26=Hips, 27-28=Knees, 29-30=Ankles, 31-32=Heels/Toes
         (also includes palm finger tips)

Vicon 39 joints are typically anatomically labeled. We'll map the ones that exist
in both systems.
"""

import numpy as np
from typing import Tuple


class MediaPipeMapper:
    """
    Map Vicon 39-joint sequences to MediaPipe 33-landmark format.
    
    Vicon 39 typical layout (from UI-PRMD):
    0-2: Head, Neck, Thorax
    3-5: Right shoulder, elbow, wrist
    6-8: Left shoulder, elbow, wrist
    9-11: Right hip, knee, ankle
    12-14: Left hip, knee, ankle
    + fingers, toes, and other anatomical points
    
    MediaPipe 33:
    0: Nose
    1-4: Eyes (2-inner L/R, 3-outer L/R)
    5-10: Ears + some face points
    11-12: Shoulders (R/L)
    13-14: Elbows (R/L)
    15-16: Wrists (R/L)
    17-18: Palms (R/L)
    19-20: Pinkies (R/L)
    21-22: Index (R/L)
    23-24: Thumbs (R/L)
    25-26: Hips (R/L)
    27-28: Knees (R/L)
    29-30: Ankles (R/L)
    31-32: Foot Index/Heel (R/L)
    """

    # Simple heuristic mapping: best-effort alignment of Vicon->MediaPipe
    # Vicon index -> MediaPipe index
    # We assume Vicon has at least 39 joints; if not enough, pad with zeros
    VICON_TO_MEDIAPIPE = {
        # Head region (Vicon 0-2 → MediaPipe 0)
        0: 0,  # Head / Nose
        # Torso (Vicon 3-8 → MediaPipe 11-16)
        3: 11,   # Right shoulder
        4: 13,   # Right elbow
        5: 15,   # Right wrist
        6: 12,   # Left shoulder
        7: 14,   # Left elbow
        8: 16,   # Left wrist
        # Lower body (Vicon 9-14 → MediaPipe 25-30)
        9: 25,   # Right hip
        10: 27,  # Right knee
        11: 29,  # Right ankle
        12: 26,  # Left hip
        13: 28,  # Left knee
        14: 30,  # Left ankle
    }

    @staticmethod
    def vicon_to_mediapipe(vicon_seq: np.ndarray) -> np.ndarray:
        """
        Convert Vicon 39-joint sequence to MediaPipe 33-landmark format.
        
        Args:
            vicon_seq: (T, 39, 3) array of Vicon positions
            
        Returns:
            (T, 33, 3) array of MediaPipe-aligned positions
        """
        T = vicon_seq.shape[0]
        mediapipe_seq = np.zeros((T, 33, 3), dtype=np.float32)

        for vicon_idx, mp_idx in MediaPipeMapper.VICON_TO_MEDIAPIPE.items():
            if vicon_idx < vicon_seq.shape[1]:
                mediapipe_seq[:, mp_idx, :] = vicon_seq[:, vicon_idx, :]

        return mediapipe_seq

    @staticmethod
    def mediapipe_33_joint_names() -> list:
        """Return canonical names for 33 MediaPipe landmarks."""
        return [
            # Head/Face (0-10)
            "Nose",
            "Left Eye Inner", "Left Eye", "Left Eye Outer",
            "Right Eye Inner", "Right Eye", "Right Eye Outer",
            "Left Ear", "Right Ear",
            "Mouth Left", "Mouth Right",
            # Upper body (11-24)
            "Left Shoulder", "Right Shoulder",
            "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist",
            "Left Palm", "Right Palm",
            "Left Pinky", "Right Pinky",
            "Left Index", "Right Index",
            "Left Thumb", "Right Thumb",
            # Lower body (25-32)
            "Left Hip", "Right Hip",
            "Left Knee", "Right Knee",
            "Left Ankle", "Right Ankle",
            "Left Heel/Foot Index", "Right Heel/Foot Index",
        ]
