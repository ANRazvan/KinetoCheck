"""
Convert IntelliRehab skeleton data from .txt files to .npy format for training.
Based on the original intellirehab_helper.py preprocessing logic.

Filename format: SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.txt
- CorrectLabel: 1 = correct, 2 = incorrect (label-1 in code, so 0 = correct, 1 = incorrect)

Usage:
    python tools/convert_intellirehab_data.py --input ../Datasets/SkeletonData/RawData --output ../Datasets/SkeletonData/Simplified
"""
import os
import sys
import argparse
from pathlib import Path

print("Basic imports successful", flush=True)

# Suppress NumPy warnings for Python 3.13
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*MINGW.*')

print("Loading numpy...", flush=True)
import numpy as np
print("Loading pandas...", flush=True)
import pandas as pd

print("Loading sklearn...", flush=True)
try:
    from sklearn.utils import resample
    SKLEARN_AVAILABLE = True
    print("sklearn loaded successfully", flush=True)
except ImportError as e:
    print(f"Warning: sklearn not available ({e}), downsampling disabled", flush=True)
    SKLEARN_AVAILABLE = False
    resample = None

print("All libraries loaded!", flush=True)


def extract_info_from_filename(filename: str):
    """
    Parse metadata from IntelliRehab filename format.
    
    Format: SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.txt
    
    Original code does: label = int(pieces[4]) - 1
    - CorrectLabel=1 (correct) → label=0
    - CorrectLabel=2 (incorrect) → label=1
    
    Returns:
        dict with keys: subject_id, date_id, gesture, repetition, label, position
        or None if cannot parse
    """
    pieces = filename.replace('.txt', '').split("_")
    
    if len(pieces) < 6:
        print(f"Warning: Cannot parse filename '{filename}' - expected 6 parts, got {len(pieces)}")
        return None
    
    subject_id = pieces[0]
    date_id = pieces[1]
    exercise_id = pieces[2]
    repetition_number = pieces[3]
    
    try:
        # Original code: label = int(pieces[4]) - 1
        # So CorrectLabel=1 → 0 (correct), CorrectLabel=2 → 1 (incorrect)
        correct_label_raw = int(pieces[4])
        label = correct_label_raw - 1
    except ValueError:
        print(f"Warning: Cannot parse label in filename '{filename}'")
        return None
    
    position_str = pieces[5].split('.')[0]  # remove .txt if still there
    
    # Map position (keeping original logic)
    if position_str == "stand":
        position = 1
    elif position_str == "chair":
        position = 2
    else:
        position = 3  # wheelchair or other
    
    return {
        'subject_id': subject_id,
        'date_id': date_id,
        'exercise_id': exercise_id,
        'repetition': repetition_number,
        'label': label,  # 0 = correct, 1 = incorrect
        'position': position,
        'position_str': position_str,
        'raw_correct_label': correct_label_raw,
    }


def read_txt_file(filename: str) -> np.ndarray:
    """
    Read skeleton data from IntelliRehab CSV format.
    Original implementation: pd.read_csv with comma separator, no header.
    
    Returns:
        np.ndarray of shape (num_frames, num_features)
        or None if read fails
    """
    try:
        df = pd.read_csv(filename, sep=',', header=None, index_col=False)
        return df.values
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def convert_dataset(
    input_dir: str,
    output_dir: str,
    skip_incorrect: bool = True,
    downsample: bool = False,
):
    """
    Convert all IntelliRehab .txt files to .npy and organize by correctness.
    
    Args:
        input_dir: Directory with .txt files
        output_dir: Output directory (will create correct/ and incorrect/ subdirs)
        skip_incorrect: If True, skip files with CorrectLabel=2 (label=1 after -1), matching original code
        downsample: If True, balance classes by downsampling majority class
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    correct_dir = output_path / "correct"
    incorrect_dir = output_path / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)
    
    all_files = list(input_path.glob("*.txt"))
    
    if len(all_files) == 0:
        print(f"ERROR: No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} text files in {input_dir}")
    print(f"Skip incorrect (label=2): {skip_incorrect}")
    print(f"Downsample: {downsample}\n")
    
    # First pass: load all data
    all_data = []
    skipped = 0
    
    for txt_file in all_files:
        # Parse metadata
        metadata = extract_info_from_filename(txt_file.name)
        if metadata is None:
            skipped += 1
            continue
        
        # Skip incorrect samples if requested (original code skips label=2, which becomes label=1 after -1)
        if skip_incorrect and metadata['label'] == 1:
            skipped += 1
            continue
        
        # Load skeleton data
        skeleton = read_txt_file(str(txt_file))
        if skeleton is None or skeleton.shape[0] == 0:
            print(f"Skipping {txt_file.name}: cannot load skeleton data")
            skipped += 1
            continue
        
        # Store for processing
        all_data.append({
            'filename': txt_file.stem + ".npy",
            'skeleton': skeleton,
            'label': metadata['label'],
            'metadata': metadata,
        })
        
        if len(all_data) % 50 == 0:
            print(f"Loaded {len(all_data)}/{len(all_files)} files...")
    
    print(f"\nLoaded {len(all_data)} valid samples, skipped {skipped}")
    
    if len(all_data) == 0:
        print("ERROR: No valid data to convert!")
        return
    
    # Apply downsampling if requested
    if downsample:
        if not SKLEARN_AVAILABLE:
            print("ERROR: Downsampling requested but sklearn is not available!")
            print("Install it with: pip install scikit-learn")
            return
        
        print("\nApplying downsampling to balance classes...")
        
        # Separate by class (label: 0=correct, 1=incorrect)
        correct_samples = [x for x in all_data if x['label'] == 0]
        incorrect_samples = [x for x in all_data if x['label'] == 1]
        
        print(f"Before downsampling: Correct={len(correct_samples)}, Incorrect={len(incorrect_samples)}")
        
        if len(correct_samples) > 0 and len(incorrect_samples) > 0:
            # Downsample to match minority class
            min_count = min(len(correct_samples), len(incorrect_samples))
            
            if len(correct_samples) > min_count:
                # Manual downsampling with numpy
                indices = np.random.RandomState(42).choice(len(correct_samples), min_count, replace=False)
                correct_samples = [correct_samples[i] for i in indices]
            if len(incorrect_samples) > min_count:
                indices = np.random.RandomState(42).choice(len(incorrect_samples), min_count, replace=False)
                incorrect_samples = [incorrect_samples[i] for i in indices]
            
            all_data = correct_samples + incorrect_samples
            print(f"After downsampling: Correct={len(correct_samples)}, Incorrect={len(incorrect_samples)}")
        else:
            print("Warning: Cannot downsample - only one class present")
    
    # Save all data
    correct_count = 0
    incorrect_count = 0
    
    for item in all_data:
        # Determine output directory (label: 0=correct, 1=incorrect)
        if item['label'] == 0:
            output_subdir = correct_dir
            correct_count += 1
        else:
            output_subdir = incorrect_dir
            incorrect_count += 1
        
        # Save as .npy
        output_filepath = output_subdir / item['filename']
        np.save(output_filepath, item['skeleton'])
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Total converted: {len(all_data)}")
    print(f"  Total skipped: {skipped}")
    print(f"  Correct samples (label=0): {correct_count}")
    print(f"  Incorrect samples (label=1): {incorrect_count}")
    print(f"{'='*60}")
    print(f"\nOutput directories:")
    print(f"  Correct:   {correct_dir}")
    print(f"  Incorrect: {incorrect_dir}")


def main():
    try:
        print("\n" + "="*60, flush=True)
        print("IntelliRehab Data Conversion Tool", flush=True)
        print("="*60, flush=True)
        print(f"Python: {sys.version.split()[0]}", flush=True)
        print(f"NumPy: {np.__version__}", flush=True)
        print(f"Pandas: {pd.__version__}", flush=True)
        print(f"Sklearn: {'available' if SKLEARN_AVAILABLE else 'not available'}", flush=True)
        print("="*60 + "\n", flush=True)
        
        parser = argparse.ArgumentParser(
            description="Convert IntelliRehab skeleton text files to .npy format",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic conversion (skip incorrect as in original code)
  python tools/convert_intellirehab_data.py --input ../Datasets/SkeletonData/RawData --output ../Datasets/SkeletonData/Simplified
  
  # Include all data (both correct and incorrect)
  python tools/convert_intellirehab_data.py --input ../Datasets/SkeletonData/RawData --output ../Datasets/SkeletonData/Simplified --no-skip-incorrect
  
  # Include all data + downsample to balance classes
  python tools/convert_intellirehab_data.py --input ../Datasets/SkeletonData/RawData --output ../Datasets/SkeletonData/Simplified --no-skip-incorrect --downsample
        """
        )
        parser.add_argument("--input", required=True, help="Input directory with .txt files")
        parser.add_argument("--output", required=True, help="Output directory")
        parser.add_argument(
            "--no-skip-incorrect",
            action="store_true",
            help="Include incorrect samples (label=2). By default, only correct samples are kept (matching original code).",
        )
        parser.add_argument(
            "--downsample",
            action="store_true",
            help="Balance classes by downsampling majority class",
        )
        
        print("Parsing arguments...", flush=True)
        args = parser.parse_args()
        
        print(f"\nConfiguration:", flush=True)
        print(f"  Input:  {args.input}", flush=True)
        print(f"  Output: {args.output}", flush=True)
        print(f"  Skip incorrect: {not args.no_skip_incorrect}", flush=True)
        print(f"  Downsample: {args.downsample}\n", flush=True)
        
        if not os.path.exists(args.input):
            print(f"ERROR: Input directory does not exist: {args.input}", flush=True)
            print(f"Current working directory: {os.getcwd()}", flush=True)
            sys.exit(1)
        
        print("Starting conversion...\n", flush=True)
        convert_dataset(
            args.input,
            args.output,
            skip_incorrect=not args.no_skip_incorrect,
            downsample=args.downsample,
        )
        
        print("\n✓ Script completed successfully!", flush=True)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
