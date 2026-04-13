
import cv2
import mediapipe as mp
import numpy as np

from app.services.pose_extractor.base import BasePoseExtractor

class MediaPipePoseExtractor(BasePoseExtractor):
    """
    Extracts 33 keypoints using Google's MediaPipe Pose solution.
    Maps them to the required format (e.g. COCO-17 or Kinect-25).
    """
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.keypoint_mapping = {
            # Map MediaPipe (33) to COCO-17 (used by YOLO/Our system)
            # COCO: 0:Nose, 1:L-Eye, 2:R-Eye, 3:L-Ear, 4:R-Ear, 5:L-Sh, 6:R-Sh, 7:L-Elb, 8:R-Elb, 9:L-Wr, 10:R-Wr, 11:L-Hip, 12:R-Hip, 13:L-Knee, 14:R-Knee, 15:L-Ank, 16:R-Ank
            # MP: 0:Nose, 2:L-Eye, 5:R-Eye, 7:L-Ear, 8:R-Ear, 11:L-Sh, 12:R-Sh, 13:L-Elb, 14:R-Elb, 15:L-Wr, 16:R-Wr, 23:L-Hip, 24:R-Hip, 25:L-Knee, 26:R-Knee, 27:L-Ank, 28:R-Ank
            0: 0,
            1: 2, # Note: MP eyes are 2(L Inner) 5(R Inner) approx
            2: 5,
            3: 7,
            4: 8,
            5: 11,
            6: 12,
            7: 13,
            8: 14,
            9: 15,
            10: 16,
            11: 23,
            12: 24,
            13: 25,
            14: 26,
            15: 27,
            16: 28
        }

    def extract_from_video(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames_keypoints = []

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            # Draw the pose annotation on the image.
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            frame_kps = self._map_results_to_coco17_xyz(results, image.shape)

            frames_keypoints.append(frame_kps)

        cap.release()
        return np.array(frames_keypoints) # (Frames, 17, 3)

    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract COCO-17 keypoints (x, y, z) from one BGR frame."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        return self._map_results_to_coco17_xyz(results, image.shape)

    def _map_results_to_coco17_xyz(self, results, rgb_shape: tuple[int, int, int]) -> np.ndarray:
        """
        Convert MediaPipe landmarks to COCO-17 with 3D coordinates.

        We use image-space landmarks: x,y are pixel coordinates and z is
        MediaPipe-relative depth scaled by image width for consistent units.
        """
        if not results.pose_landmarks:
            return np.zeros((17, 3), dtype=np.float32)

        landmarks = results.pose_landmarks.landmark
        h, w, _ = rgb_shape
        temp_kps = np.zeros((17, 3), dtype=np.float32)

        for coco_idx, mp_idx in self.keypoint_mapping.items():
            lm = landmarks[mp_idx]
            temp_kps[coco_idx] = [lm.x * w, lm.y * h, lm.z * w]

        return temp_kps
