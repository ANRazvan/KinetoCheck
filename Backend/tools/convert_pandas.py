"""
Convert IntelliRehab data using the EXACT same method as the original project.
Uses pandas.read_csv() exactly as in intellirehab_helper.py

Usage:
    python tools/convert_pandas.py --input ../SkeletonData/RawData --output ../SkeletonData/Simplified
"""
import os
import sys
import argparse
from pathlib import Path

print("Starting conversion using pandas (original method)...", flush=True)

import pandas as pd
import numpy as np

print("Libraries loaded", flush=True)


def read_txt_file(filename: str):
    """Original implementation from intellirehab_helper.py - skip Version0.1 header(s)"""
    # Some files have duplicate "Version0.1" headers - skip all header rows
    with open(filename, 'r') as f:
        lines_to_skip = 0
        for line in f:
            if line.strip() == "Version0.1":
                lines_to_skip += 1
            else:
                break
    
    df = pd.read_csv(filename, sep=',', header=None, index_col=False, skiprows=lines_to_skip)
    return df.values


def convert_dataset(input_dir: str, output_dir: str):
    """Convert all .txt files to .npy using the original read method."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_files = list(input_path.glob("*.txt"))
    
    if len(all_files) == 0:
        print(f"ERROR: No .txt files found in {input_dir}", flush=True)
        return
    
    print(f"Found {len(all_files)} text files", flush=True)
    print(f"Output directory: {output_path}\n", flush=True)
    
    converted = 0
    skipped = 0
    
    for i, txt_file in enumerate(all_files):
        try:
            # Use EXACT original method
            skeleton_data = read_txt_file(str(txt_file))
            
            # Debug first file
            if i == 0:
                print(f"\nFirst file debug:", flush=True)
                print(f"  Shape: {skeleton_data.shape}", flush=True)
                print(f"  Sample row: {skeleton_data[0][:10]}", flush=True)
            
            if skeleton_data is None or skeleton_data.shape[0] == 0:
                print(f"Skipping {txt_file.name}: empty data", flush=True)
                skipped += 1
                continue
            
            # Save as .npy
            output_filename = txt_file.stem + ".npy"
            output_filepath = output_path / output_filename
            np.save(output_filepath, skeleton_data)
            
            converted += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(all_files)}...", flush=True)
                
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}", flush=True)
            skipped += 1
            continue
    
    print(f"\n{'='*60}", flush=True)
    print(f"Conversion complete!", flush=True)
    print(f"  Total converted: {converted}", flush=True)
    print(f"  Total skipped: {skipped}", flush=True)
    print(f"{'='*60}\n", flush=True)
    print(f"Output: {output_path}", flush=True)


def main():
    try:
        parser = argparse.ArgumentParser(description="Convert IntelliRehab data (pandas method)")
        parser.add_argument("--input", required=True, help="Input directory with .txt files")
        parser.add_argument("--output", required=True, help="Output directory")
        
        args = parser.parse_args()
        
        print("="*60, flush=True)
        print("IntelliRehab Converter (Original Method)", flush=True)
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
