"""
MediaPipe Video Dataset Loader for Temporal Pyramid STGAT.

Extracts 33-joint MediaPipe poses from video files and creates (T, 33, 3) sequences.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MediaPipeVideoDataset:
    """Load skeleton sequences from video files using MediaPipe pose extraction."""

    def __init__(self, video_root: str, label_mapping: Optional[Dict[str, int]] = None):
        """
        Args:
            video_root: Path to folder containing video files or organized subfolders
            label_mapping: Dict mapping folder/file names to labels (e.g., {"correct": 0, "incorrect": 1})
        """
        self.video_root = Path(video_root)
        self.label_mapping = label_mapping or {"correct": 0, "incorrect": 1}
        self.samples: List[Tuple[str, int, Dict]] = []  # (video_path, label, metadata)
        self._scan_videos()

    def _scan_videos(self):
        """Recursively scan for video files and assign labels based on folder structure."""
        for label_name, label_id in self.label_mapping.items():
            label_dir = self.video_root / label_name
            if not label_dir.exists():
                continue

            for video_file in label_dir.glob("*.mp4"):
                self.samples.append((
                    str(video_file),
                    label_id,
                    {
                        "file": str(video_file),
                        "label": label_id,
                        "label_name": label_name,
                        "video_name": video_file.stem,
                    }
                ))

            # Also scan subdirectories (e.g., correct/exercise_0/...)
            for subdir in label_dir.iterdir():
                if subdir.is_dir():
                    for video_file in subdir.glob("*.mp4"):
                        exercise_id = subdir.name if subdir.name.startswith("exercise") else "unknown"
                        self.samples.append((
                            str(video_file),
                            label_id,
                            {
                                "file": str(video_file),
                                "label": label_id,
                                "label_name": label_name,
                                "exercise": exercise_id,
                                "video_name": video_file.stem,
                            }
                        ))

        logger.info(f"Scanned {len(self.samples)} video files from {self.video_root}")

    def load_all(self, exercise_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load all sequences from videos.

        Args:
            exercise_id: Filter by exercise_id (ignored for video dataset)

        Returns:
            data: (N, T, 33, 3) - N sequences, T timesteps, 33 MediaPipe joints
            labels: (N,) - 0=correct, 1=incorrect
            metadata: List of dicts with sequence info
        """
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise ImportError(
                "MediaPipe video loading requires opencv-python and mediapipe."
            ) from e

        # Download/cache MediaPipe model
        model_dir = Path("temporal_pyramid_stgat") / "weights"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "pose_landmarker_lite.task"

        if not model_path.exists():
            logger.info("Downloading MediaPipe Pose Landmarker model...")
            import urllib.request
            url = (
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            )
            urllib.request.urlretrieve(url, str(model_path))

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        data_list = []
        labels_list = []
        metadata_list = []

        for idx, (video_path, label, meta) in enumerate(self.samples):
            seq = self._extract_sequence(video_path, detector)
            if seq is not None:
                data_list.append(seq)
                labels_list.append(label)
                metadata_list.append(meta)
                logger.info(f"[{idx+1}/{len(self.samples)}] Loaded {video_path}: shape {seq.shape}, label={label}")
            else:
                logger.warning(f"Failed to extract from {video_path}")

        detector.close()

        if not data_list:
            raise ValueError(f"No valid sequences extracted from {self.video_root}")

        # Normalize to common length
        data = self._normalize_lengths(data_list)
        labels = np.array(labels_list, dtype=np.int64)

        return data, labels, metadata_list

    def _extract_sequence(self, video_path: str, detector) -> Optional[np.ndarray]:
        """
        Extract 33-joint MediaPipe pose sequence from a video.

        Returns:
            array of shape (T, 33, 3) or None if extraction failed
        """
        try:
            import cv2
            import mediapipe as mp

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            frames = []
            frame_id = 0

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = detector.detect_for_video(mp_image, frame_id)

                h, w = frame.shape[:2]
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    arr = np.zeros((33, 3), dtype=np.float32)
                    for i, lm in enumerate(landmarks[:33]):
                        arr[i, 0] = float(lm.x) * w
                        arr[i, 1] = float(lm.y) * h
                        arr[i, 2] = float(lm.z) * w
                    frames.append(arr)
                else:
                    frames.append(np.zeros((33, 3), dtype=np.float32))

                frame_id += 1

            cap.release()

            if not frames:
                return None

            return np.array(frames, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error extracting from {video_path}: {e}")
            return None

    def _normalize_lengths(self, sequences: List[np.ndarray], target_len: int = 240) -> np.ndarray:
        """Pad or truncate sequences to fixed length."""
        normalized = []

        for seq in sequences:
            T = seq.shape[0]

            if T >= target_len:
                # Take center window
                start = (T - target_len) // 2
                normalized.append(seq[start : start + target_len])
            else:
                # Pad with zeros
                pad_total = target_len - T
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                padded = np.pad(
                    seq,
                    ((pad_before, pad_after), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
                normalized.append(padded)

        data = np.array(normalized, dtype=np.float32)
        return data


if __name__ == "__main__":
    # Example usage
    dataset = MediaPipeVideoDataset("../../Video-kineto")
    data, labels, metadata = dataset.load_all()
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    for m in metadata:
        print(m)
