import os
import glob
from visualize import visualize_subject # Assuming visualize.py is in the same directory

def generate_sample_videos():
    # Setup directories
    output_base_dir = "Video-kineto"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # We will just use m01 (subject 1) for our samples
    base_data_path = os.path.join("Datasets", "UIPRMD")
    
    exercises = [
        "exercise_01", "exercise_02", "exercise_03", "exercise_04", "exercise_05",
        "exercise_06", "exercise_07", "exercise_08", "exercise_09", "exercise_10"
    ]
    
    for ex in exercises:
        print(f"Generating videos for {ex}...")
        
        # Paths for subject m01
        correct_path = os.path.join(base_data_path, "Positions", ex, "m01_s01.txt")
        incorrect_path = os.path.join(base_data_path, "Positions", ex, "m01_s02.txt")
        
        # Generate Correct Video
        if os.path.exists(correct_path):
            output_file = os.path.join(output_base_dir, f"{ex}_correct.mp4")
            # We pass save_video=True and specify the output path
            try:
                visualize_subject(correct_path, interval=20, save_video=True, video_path=output_file)
                print(f"  Saved: {output_file}")
            except Exception as e:
                print(f"  Error generating {output_file}: {e}")
        else:
             print(f"  Could not find input file: {correct_path}")
             
        # Generate Incorrect Video
        if os.path.exists(incorrect_path):
            output_file = os.path.join(output_base_dir, f"{ex}_incorrect.mp4")
            try:
                visualize_subject(incorrect_path, interval=20, save_video=True, video_path=output_file)
                print(f"  Saved: {output_file}")
            except Exception as e:
                print(f"  Error generating {output_file}: {e}")
        else:
             print(f"  Could not find input file: {incorrect_path}")

if __name__ == "__main__":
    generate_sample_videos()