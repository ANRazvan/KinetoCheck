import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Import your existing local modules
from Preprocessing.UIPRMD_loader import UIPRMDLoader
from Preprocessing.UIPRMDPreprocessor import UIPRMDPreprocessor

# COCO17 Connections for plotting
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
MP_COCO17_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def extract_mediapipe_sequence(video_path: Path, pose_model_path: Path) -> np.ndarray:
    """Extracts raw MediaPipe landmarks from a video file."""
    base_options = mp_python.BaseOptions(model_asset_path=str(pose_model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sequence = []
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            det = landmarker.detect_for_video(mp_img, ts_ms)

            if det.pose_landmarks:
                lms = det.pose_landmarks[0]
                # Extract x, y, z for the 17 COCO joints
                pts = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in MP_COCO17_IDXS], dtype=np.float32)
                sequence.append(pts)
            frame_idx += 1

    cap.release()
    return np.stack(sequence, axis=0) if sequence else np.empty((0, 17, 3))

def normalize_skeleton(sequence: np.ndarray) -> np.ndarray:
    """Hip-centers and scales the skeleton by torso length for fair comparison."""
    # 1. Hip centering (Midpoint of joint 7 and 8)
    hip_center = (sequence[:, 7, :] + sequence[:, 8, :]) * 0.5
    centered = sequence - hip_center[:, np.newaxis, :]
    
    # 2. Torso scaling (Distance from hip center to shoulder center)
    shoulder_center = (centered[:, 1, :] + centered[:, 2, :]) * 0.5
    torso_length = np.linalg.norm(shoulder_center, axis=-1, keepdims=True)
    torso_length = np.maximum(torso_length, 1e-6)
    
    scaled = centered / torso_length[:, np.newaxis, :]
    return scaled

def match_lengths(seq1: np.ndarray, seq2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resamples the longer sequence to match the frame count of the shorter one."""
    t1, j, d = seq1.shape
    t2, _, _ = seq2.shape
    target_len = min(t1, t2)
    
    def resample(seq, target):
        x_old = np.linspace(0, 1, seq.shape[0])
        x_new = np.linspace(0, 1, target)
        out = np.zeros((target, j, d))
        for joint in range(j):
            for dim in range(d):
                out[:, joint, dim] = np.interp(x_new, x_old, seq[:, joint, dim])
        return out

    return resample(seq1, target_len), resample(seq2, target_len)

def plot_skeletons_3d(vicon_frame: np.ndarray, mp_frame: np.ndarray):
    """Plots a single frame of Vicon and MediaPipe data side-by-side."""
    fig = plt.figure(figsize=(12, 6))
    
    # Vicon Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Vicon (Ground Truth)")
    _draw_3d_skeleton(ax1, vicon_frame, color='blue')
    
    # MediaPipe Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("MediaPipe (Estimated)")
    _draw_3d_skeleton(ax2, mp_frame, color='red')
    
    plt.tight_layout()
    plt.show()

def _draw_3d_skeleton(ax, frame, color):
    # Scatter joints
    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c=color, s=20)
    
    # Draw bones
    for a, b in COCO17_EDGES:
        ax.plot(
            [frame[a, 0], frame[b, 0]], 
            [frame[a, 1], frame[b, 1]], 
            [frame[a, 2], frame[b, 2]], 
            c='black', linewidth=2
        )
        
    # Ensure consistent axis limits for fair visual comparison
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth (Z)')

def robust_load_vicon(file_path: Path) -> np.ndarray | None:
    if not file_path.exists():
        print(f"  [Error] File not found: {file_path}")
        return None
        
    try:
        # Try standard comma delimiter first
        arr = np.loadtxt(file_path, delimiter=",", ndmin=2)
    except Exception as e:
        print(f"  [Warning] Comma delimiter failed: {e}. Trying whitespace...")
        try:
            # Fallback to whitespace delimiter
            arr = np.loadtxt(file_path, ndmin=2)
        except Exception as e2:
            print(f"  [Error] Could not parse file as floats. Is it corrupted? {e2}")
            return None

    print(f"  [Debug] Successfully read txt file. Raw shape: {arr.shape}")

    if arr.shape[1] < 117:
        print(f"  [Error] Found {arr.shape[1]} columns, but need at least 117.")
        print("          Are you sure this is a Vicon 'Positions' file and not an 'Angles' file?")
        return None

    # Slice the first 117 columns and reshape (ignoring any trailing empty columns)
    coords = arr[:, :117].reshape(-1, 39, 3)
    return coords.astype(np.float32)

def animate_skeletons_3d(vicon_seq: np.ndarray, mp_seq: np.ndarray, fps: int = 30):
    """Animates Vicon and MediaPipe data side-by-side over time."""
    fig = plt.figure(figsize=(12, 6))
    
    # Setup Vicon Axis
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Vicon (Ground Truth)")
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])
    ax1.set_zlim([-2, 2])
    ax1.set_xlabel('X (Lateral)')
    ax1.set_ylabel('Y (Vertical)')
    ax1.set_zlabel('Z (Depth)')
    
    # Setup MediaPipe Axis
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("MediaPipe (Estimated)")
    ax2.set_xlim([-2, 2])
    ax2.set_ylim([-2, 2])
    ax2.set_zlim([-2, 2])
    ax2.set_xlabel('X (Lateral)')
    ax2.set_ylabel('Y (Vertical)')
    ax2.set_zlabel('Z (Depth)')

    # Initialize scatter points
    v_scatter = ax1.scatter([], [], [], c='blue', s=30)
    m_scatter = ax2.scatter([], [], [], c='red', s=30)
    
    # Initialize lines (bones)
    v_lines = [ax1.plot([], [], [], c='black', linewidth=2)[0] for _ in COCO17_EDGES]
    m_lines = [ax2.plot([], [], [], c='black', linewidth=2)[0] for _ in COCO17_EDGES]

    def update(frame_idx):
        v_frame = vicon_seq[frame_idx]
        m_frame = mp_seq[frame_idx]

        # Update scatter positions
        v_scatter._offsets3d = (v_frame[:, 0], v_frame[:, 1], v_frame[:, 2])
        m_scatter._offsets3d = (m_frame[:, 0], m_frame[:, 1], m_frame[:, 2])

        # Update bone lines
        for i, (a, b) in enumerate(COCO17_EDGES):
            # Vicon bones
            v_lines[i].set_data([v_frame[a, 0], v_frame[b, 0]], [v_frame[a, 1], v_frame[b, 1]])
            v_lines[i].set_3d_properties([v_frame[a, 2], v_frame[b, 2]])
            
            # MediaPipe bones
            m_lines[i].set_data([m_frame[a, 0], m_frame[b, 0]], [m_frame[a, 1], m_frame[b, 1]])
            m_lines[i].set_3d_properties([m_frame[a, 2], m_frame[b, 2]])
            
        fig.suptitle(f"Frame {frame_idx + 1} / {len(vicon_seq)}", fontsize=14)
        return [v_scatter, m_scatter] + v_lines + m_lines

    # Create the animation
    interval_ms = int(1000 / fps)
    ani = FuncAnimation(fig, update, frames=len(vicon_seq), interval=interval_ms, blit=False)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare Vicon vs MediaPipe domains.")
    parser.add_argument("--vicon-txt", type=Path, required=True, help="Path to Vicon .txt file")
    parser.add_argument("--video", type=Path, required=True, help="Path to corresponding video file")
    parser.add_argument("--pose-model", type=Path, default=Path(".cache/mediapipe/pose_landmarker_full.task"))
    args = parser.parse_args()

    print(f"Loading Vicon data from {args.vicon_txt}...")
    vicon_raw = robust_load_vicon(args.vicon_txt)
    
    if vicon_raw is None:
        print("Aborting: Failed to load Vicon data.")
        return
    
    # Preprocess Vicon down to 17 joints
    preprocessor = UIPRMDPreprocessor()
    vicon_17 = preprocessor.align_vicon_to_mediapipe(vicon_raw)
    
    # NEW: Swap Vicon axes to match MediaPipe (Vicon Y->X, Vicon Z->Y, Vicon X->Z)
    vicon_17 = vicon_17[:, :, [1, 2, 0]]
    vicon_17[:, :, 2] = -vicon_17[:, :, 2]
    
    print("Extracting MediaPipe data from video...")
    mp_raw = extract_mediapipe_sequence(args.video, args.pose_model)
    if len(mp_raw) == 0:
        print("Failed to extract MediaPipe poses.")
        return

    # Note: MediaPipe Y-axis is inverted (0 is top of image, 1 is bottom) compared to typical 3D spaces
    mp_raw[:, :, 1] = -mp_raw[:, :, 1]

    # Normalize both to common space (hip-centered, torso-scaled)
    vicon_norm = normalize_skeleton(vicon_17)
    mp_norm = normalize_skeleton(mp_raw)

    # Match sequence lengths for statistical comparison
    vicon_matched, mp_matched = match_lengths(vicon_norm, mp_norm)

    print("\n=== Statistical Comparison ===")
    
    # 1. Total Mean Squared Error
    mse = np.mean((vicon_matched - mp_matched) ** 2)
    print(f"Overall Mean Squared Error: {mse:.4f}")
    
    # 2. Axis-specific variance (Highlighting the Z-axis issue)
    print("\nVariance by Axis (Spread of movement):")
    for axis_idx, axis_name in enumerate(["X (Lateral)", "Y (Vertical)", "Z (Depth)"]):
        v_var = np.var(vicon_matched[:, :, axis_idx])
        m_var = np.var(mp_matched[:, :, axis_idx])
        print(f"  {axis_name}: Vicon = {v_var:.4f} | MediaPipe = {m_var:.4f}")

    print("\nRendering 3D Animation...")
    # Call the new animation function instead of the static plot
    animate_skeletons_3d(vicon_matched, mp_matched, fps=30)
    
if __name__ == "__main__":
    main()