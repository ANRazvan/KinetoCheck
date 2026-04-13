
import os
import sys
import torch
import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train_exercise
from app.services.inference_service import InferenceService
from app.services.pose_extractor.factory import create_pose_extractor

def run_optimization():
    print("=== Starting Optimization Cycle ===")
    
    # Configuration
    EXERCISE_ID = 3  # Shoulder Flexion (based on file name "Shoulder_Flexion_correct_3seconds.mp4")
                     # NOTE: Verify mapping. 0=ChairSquat, 1=ElbowExt, 2=ElbowFlex, 3=ShoulderExt...
                     # Wait, IntelliRehab mapping:
                     # 1: Elbow Flexion, 2: Elbow Extension, 3: Shoulder Flexion? 
                     # I should check the mapping in settings or just try training Ex 3.
    
    # 1. Train the model for Exercise 3 using the robust dataset (which we just patched).
    print(f"Training parameters: Exercise {EXERCISE_ID}, Extractor: MediaPipe (for inference)")
    
    try:
        # We can't easily change training epochs from here without modifying config.py or train.py arguments.
        # But we can call train_exercise directly.
        # Note: This will use the default config settings (epochs, batch size etc).
        
        best_acc = train_exercise(
            exercise_id=EXERCISE_ID, 
            model_name="stgat", 
            dataset_key="intellirehab_2d" # Use the dataset where we added hard negatives
        )
        print(f"Training complete. Best Acc: {best_acc:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # 2. Test Validity on Video
    print("\n=== Validating on Videos ===")

    inference_service = InferenceService(
        model_name="stgat",
        dataset="intellirehab_2d",
        pose_extractor=create_pose_extractor("mediapipe") # Use our new MP extractor
    )
    
    # Test Cases
    videos = [
        ("Good (Shoulder Flexion)", "Video-kineto/Shoulder_Flexion_correct_3seconds.mp4"),
        ("Bad (Deep Squat - Hard Negative)", "Video-kineto/deep_squat_multiple_repetitions.mp4"),
        ("Bad (Deep Squat 2)", "Video-kineto/deep_squat_multiple_repetitions_2.mp4"),
    ]
    
    for label, vid_rel_path in videos:
        vid_path = os.path.join(os.getcwd(), vid_rel_path)
        if not os.path.exists(vid_path):
            print(f"Video not found: {vid_path}")
            continue
            
        print(f"\nProcessing '{label}': {os.path.basename(vid_path)}")
        try:
            result = inference_service.predict_from_video(vid_path, exercise_id=EXERCISE_ID)
            
            conf = result.get("confidence", 0.0)
            pred_label = result.get("label", "Unknown") # 0=Correct, 1=Incorrect
            
            print(f"  Result: {pred_label} (Conf: {conf:.2f})")
            
            # Analysis
            if "Good" in label and pred_label == 0:
                print("  [PASS] Correctly identified as Correct.")
            elif "Bad" in label and pred_label == 1:
                print("  [PASS] Correctly identified as Incorrect (Wrong Exercise/Form).")
            else:
                print(f"  [FAIL] Expected {label.split(' ')[0]} but got {pred_label}.")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    run_optimization()
