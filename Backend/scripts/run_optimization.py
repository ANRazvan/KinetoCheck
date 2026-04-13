
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock args for train.py if needed, or import functions
from training.train import train_exercise
# We need to ensure we can import app
from app.services.inference_service import InferenceService
from app.services.pose_extractor.factory import create_pose_extractor, PoseExtractorFactory
from app.services.pose_extractor.mediapipe_extractor import MediaPipePoseExtractor

def run_pipeline():
    print("==================================================")
    print("   Robustness Optimization Pipeline")
    print("==================================================")
    
    # 0. Register MediaPipe just in case (though factory should have it)
    try:
        PoseExtractorFactory.register("mediapipe", MediaPipePoseExtractor)
    except:
        pass

    # 1. Config
    # We target 'Shoulder Flexion'. In IntelliRehab files this is usually Gesture 1.
    # config.py might use 0-based indexing for the map, but the dataset loader matches file IDs.
    # We will assume Gesture 1 = Shoulder Flexion.
    EXERCISE_ID = 1
    
    print(f"[Step 1] Training Exercise {EXERCISE_ID} (Shoulder Flexion) with HARD NEGATIVES...")
    print("         (Including other exercises as 'Incorrect' samples)")
    
    # Run training
    # This might take a while.
    try:
        # We use 'intellirehab_2d' dataset which we modified to include foreign negatives.
        # Ensure your config.py has reasonable epochs (e.g. 10-20) for quick testing.
        best_acc = train_exercise(
            exercise_id=EXERCISE_ID,
            model_name="stgat", 
            dataset_key="intellirehab_2d"
        )
        print(f"[Step 1] Training Finished. Validation Accuracy: {best_acc:.2%}")
    except Exception as e:
        print(f"[Step 1] Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Inference Test on Video
    print("\n[Step 2] Testing on Video (Generalization Check)...")
    
    try:
        service = InferenceService(
            model_name="stgat",
            dataset="intellirehab_2d",
            pose_extractor=create_pose_extractor("mediapipe")
        )
        
        # Paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        video_dir = os.path.join(base_dir, "Video-kineto")
        
        test_videos = [
            ("Shoulder_Flexion_correct_3seconds.mp4", "Correct (0)"),
            ("deep_squat_multiple_repetitions.mp4", "Incorrect (1) - Wrong Exercise"),
            ("Multiple_repetitions_different_exercises.mp4", "Incorrect (1) - Mixed")
        ]
        
        for vid_name, expected_desc in test_videos:
            vid_path = os.path.join(video_dir, vid_name)
            if not os.path.exists(vid_path):
                print(f"Skipping {vid_name} (Not found)")
                continue
                
            print(f"\nProcessing: {vid_name}")
            print(f"Expected: {expected_desc}")
            
            result = service.predict_from_video(vid_path, exercise_id=EXERCISE_ID)
            
            label = result.get("label") # "correct" or "incorrect" usually string? 
            # Wait, STGAT model usually returns index? or label map?
            # existing code in inference_service: result.get("label")
            # Let's check dictionary.
            
            probs = result.get("details", {}).get("raw_probs", [])
            conf = result.get("confidence", 0.0)
            
            print(f"  -> Prediction: {label} (Conf: {conf:.4f})")
            print(f"  -> Raw Probs: {probs}")
            
            # Check correctness
            # Label "correct" corresponds to Index 0?
            if "Correct" in expected_desc and label == "correct":
                 print("  [SUCCESS] Match.")
            elif "Incorrect" in expected_desc and label == "incorrect":
                 print("  [SUCCESS] Match.")
            else:
                 print("  [FAILURE] Mismatch.")

    except Exception as e:
        print(f"[Step 2] Inference Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()
