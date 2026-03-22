"""
Simplified data converter that avoids NumPy import issues.
Converts IntelliRehab .txt files to .npy format.

Usage:
    python tools/convert_simple.py --input ../Datasets/SkeletonData/RawData --output ../Datasets/SkeletonData/Simplified
"""
import os
import sys
import argparse
from pathlib import Path
import csv

print("Starting conversion (no numpy in imports)...", flush=True)


def extract_info_from_filename(filename: str):
    """Parse metadata from filename."""
    pieces = filename.replace('.txt', '').split("_")
    if len(pieces) < 6:
        return None
    
    try:
        correct_label_raw = int(pieces[4])
        label = correct_label_raw - 1  # 1→0 (correct), 2→1 (incorrect)
    except ValueError:
        return None
    
    return {'label': label, 'raw_correct_label': correct_label_raw}


def read_skeleton_csv(filepath: str) -> list:
    """Read skeleton data without pandas, skip header lines."""
    frames = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) == 0:
                    continue
                
                # Try to convert the first value - if it fails, it's a header
                try:
                    float(row[0])
                    # If successful, try to convert all values
                    frame_data = [float(x) for x in row]
                    frames.append(frame_data)
                except ValueError:
                    # Skip this line (it's a header like "Version0.1")
                    continue
        
        return frames if len(frames) > 0 else None
    except Exception as e:
        print(f"Error reading {filepath}: {e}", flush=True)
        return None


def convert_to_npy(data: list, output_path: str):
    """Convert list to .npy file (imports numpy only here)."""
    try:
        import numpy as np
        array = np.array(data, dtype=np.float32)
        np.save(output_path, array)
        return True
    except Exception as e:
        print(f"Error saving {output_path}: {e}", flush=True)
        return False


def convert_dataset(input_dir: str, output_dir: str):
    """Convert all .txt files to .npy (keep all together, don't separate by label)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create single output directory (no separation)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_files = list(input_path.glob("*.txt"))
    
    if len(all_files) == 0:
        print(f"ERROR: No .txt files found in {input_dir}", flush=True)
        return
    
    print(f"Found {len(all_files)} text files", flush=True)
    print(f"Converting all files to: {output_path}\n", flush=True)
    
    converted = 0
    skipped = 0
    
    for i, txt_file in enumerate(all_files):
        # Read data
        skeleton_data = read_skeleton_csv(str(txt_file))
        if skeleton_data is None or len(skeleton_data) == 0:
            skipped += 1
            continue
        
        # Save as .npy in same directory
        output_filename = txt_file.stem + ".npy"
        output_filepath = output_path / output_filename
        
        if convert_to_npy(skeleton_data, str(output_filepath)):
            converted += 1
        else:
            skipped += 1
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(all_files)}...", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Conversion complete!", flush=True)
    print(f"  Total converted: {converted}", flush=True)
    print(f"  Total skipped: {skipped}", flush=True)
    print(f"{'='*60}\n", flush=True)
    print(f"Output: {output_path}", flush=True)


def main():
    try:
        parser = argparse.ArgumentParser(description="Convert IntelliRehab data (simplified version)")
        parser.add_argument("--input", required=True, help="Input directory with .txt files")
        parser.add_argument("--output", required=True, help="Output directory")
        
        args = parser.parse_args()
        
        print("="*60, flush=True)
        print("Simple IntelliRehab Converter", flush=True)
        print("="*60, flush=True)
        print(f"Input:  {args.input}", flush=True)
        print(f"Output: {args.output}\n", flush=True)
        
        if not os.path.exists(args.input):
            print(f"ERROR: Input directory does not exist: {args.input}", flush=True)
            sys.exit(1)
        
        convert_dataset(args.input, args.output)
        
        print("\n✓ Conversion completed successfully!", flush=True)
        
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
