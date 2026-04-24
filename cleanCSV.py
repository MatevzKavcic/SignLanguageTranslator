import pandas as pd
import numpy as np
from pathlib import Path

def is_landmark_missing(values):
    """Check if a landmark (x, y, z) is missing (all zeros)."""
    return all(v == 0 for v in values)

def interpolate_landmark(before, after, steps):
    """Linear interpolation between two landmarks over 'steps' frames."""
    if before is None or after is None:
        return None
    
    interpolated = []
    for i in range(1, steps + 1):
        t = i / (steps + 1)  # parameter from 0 to 1
        point = [before[j] * (1 - t) + after[j] * t for j in range(3)]
        interpolated.append(point)
    return interpolated

def clean_landmarks(csv_file, output_file="full_dataset_cleaned.csv"):
    """
    Clean missing landmarks by interpolating between detected frames.
    
    Strategy:
    - 1 missing frame: interpolate between before and after
    - 2-6 missing frames: linear interpolation from last detected to next detected
    - >6 frames: treat as straight line (same linear interpolation)
    """
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Identify landmark columns (exclude 'frame', 'video', 'label')
    exclude_cols = {'frame', 'video', 'label'}
    landmark_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Group landmark columns by landmark (each landmark has x, y, z)
    landmarks = {}
    for col in landmark_cols:
        # Extract landmark name (everything except last 2 chars which are _x, _y, _z)
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, coord = parts
            if name not in landmarks:
                landmarks[name] = {}
            landmarks[name][coord] = col
    
    print(f"Found {len(landmarks)} landmarks to process")
    print("Interpolating missing landmarks...")
    
    # Process each landmark
    for landmark_name, coords in landmarks.items():
        x_col, y_col, z_col = coords['x'], coords['y'], coords['z']
        
        i = 0
        while i < len(df):
            # Check if current frame has missing landmark
            current_values = [df.loc[i, x_col], df.loc[i, y_col], df.loc[i, z_col]]
            
            if is_landmark_missing(current_values):
                # Find the range of missing frames
                missing_start = i
                missing_end = i
                
                while missing_end + 1 < len(df):
                    next_values = [df.loc[missing_end + 1, x_col], 
                                  df.loc[missing_end + 1, y_col], 
                                  df.loc[missing_end + 1, z_col]]
                    if is_landmark_missing(next_values):
                        missing_end += 1
                    else:
                        break
                
                num_missing = missing_end - missing_start + 1
                
                # Find last detected frame before gap
                before_values = None
                before_idx = missing_start - 1
                while before_idx >= 0:
                    vals = [df.loc[before_idx, x_col], 
                           df.loc[before_idx, y_col], 
                           df.loc[before_idx, z_col]]
                    if not is_landmark_missing(vals):
                        before_values = vals
                        break
                    before_idx -= 1
                
                # Find next detected frame after gap
                after_values = None
                after_idx = missing_end + 1
                while after_idx < len(df):
                    vals = [df.loc[after_idx, x_col], 
                           df.loc[after_idx, y_col], 
                           df.loc[after_idx, z_col]]
                    if not is_landmark_missing(vals):
                        after_values = vals
                        break
                    after_idx += 1
                
                # Interpolate
                if before_values and after_values:
                    interpolated = interpolate_landmark(before_values, after_values, num_missing)
                    
                    for j, point in enumerate(interpolated):
                        frame_idx = missing_start + j
                        df.loc[frame_idx, x_col] = point[0]
                        df.loc[frame_idx, y_col] = point[1]
                        df.loc[frame_idx, z_col] = point[2]
                    
                    print(f"  {landmark_name}: filled {num_missing} frame(s) at index {missing_start}")
                else:
                    if not before_values:
                        print(f"  {landmark_name}: {num_missing} missing frame(s) at start (no before reference)")
                    if not after_values:
                        print(f"  {landmark_name}: {num_missing} missing frame(s) at end (no after reference)")
                
                i = missing_end + 1
            else:
                i += 1
    
    print(f"Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("✅ Done! Cleaned CSV saved.")

if __name__ == "__main__":
    clean_landmarks("full_dataset.csv")
