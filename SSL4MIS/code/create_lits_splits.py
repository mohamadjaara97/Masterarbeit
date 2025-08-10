#!/usr/bin/env python3
"""
Script to create train/validation/test splits for LiTS dataset.
Groups files by volume to ensure all slices of a volume stay in the same split.
"""

import os
import re
from collections import defaultdict
import random
from pathlib import Path

def extract_volume_and_slice(filename):
    """Extract volume number and slice number from filename."""
    # Pattern: volume-{volume_num}_slice_{slice_num}.h5
    match = re.match(r'volume-(\d+)_slice_(\d+)\.h5', filename)
    if match:
        volume_num = int(match.group(1))
        slice_num = int(match.group(2))
        return volume_num, slice_num
    return None, None

def group_files_by_volume(slices_dir):
    """Group all slice files by volume number."""
    volume_groups = defaultdict(list)
    
    # Get all .h5 files in the slices directory
    for filename in os.listdir(slices_dir):
        if filename.endswith('.h5'):
            volume_num, slice_num = extract_volume_and_slice(filename)
            if volume_num is not None:
                volume_groups[volume_num].append((filename, slice_num))
    
    # Sort slices within each volume
    for volume_num in volume_groups:
        volume_groups[volume_num].sort(key=lambda x: x[1])
    
    return volume_groups

def create_splits(volume_groups, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits ensuring volumes don't get mixed."""
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Get list of volumes and shuffle them
    volumes = list(volume_groups.keys())
    volumes.sort()  # Sort for reproducibility
    
    # Calculate split sizes
    total_volumes = len(volumes)
    train_size = int(total_volumes * train_ratio)
    val_size = int(total_volumes * val_ratio)
    test_size = total_volumes - train_size - val_size
    
    # Create splits
    train_volumes = volumes[:train_size]
    val_volumes = volumes[train_size:train_size + val_size]
    test_volumes = volumes[train_size + val_size:]
    
    # Create file lists for each split
    train_files = []
    val_files = []
    test_files = []
    
    for volume_num in train_volumes:
        for filename, _ in volume_groups[volume_num]:
            train_files.append(filename)
    
    for volume_num in val_volumes:
        for filename, _ in volume_groups[volume_num]:
            val_files.append(filename)
    
    for volume_num in test_volumes:
        for filename, _ in volume_groups[volume_num]:
            test_files.append(filename)
    
    return train_files, val_files, test_files, train_volumes, val_volumes, test_volumes

def write_file_list(files, output_path):
    """Write list of files to a text file."""
    with open(output_path, 'w') as f:
        for filename in files:
            # Remove .h5 extension from filename
            filename_without_ext = filename.replace('.h5', '')
            f.write(f"{filename_without_ext}\n")

def main():
    # Configuration
    slices_dir = "/home/sc.uni-leipzig.de/mj49xire/Masterarbeit/SSL4MIS/code/data/LiTS/data/slices"
    output_dir = "/home/sc.uni-leipzig.de/mj49xire/Masterarbeit/SSL4MIS/code/data/LiTS"
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    print(f"Reading files from: {slices_dir}")
    
    # Group files by volume
    volume_groups = group_files_by_volume(slices_dir)
    
    print(f"Found {len(volume_groups)} volumes")
    total_slices = sum(len(slices) for slices in volume_groups.values())
    print(f"Total slices: {total_slices}")
    
    # Create splits
    train_files, val_files, test_files, train_volumes, val_volumes, test_volumes = create_splits(
        volume_groups, train_ratio, val_ratio, test_ratio
    )
    
    print(f"\nSplit statistics:")
    print(f"Train: {len(train_volumes)} volumes, {len(train_files)} slices")
    print(f"Validation: {len(val_volumes)} volumes, {len(val_files)} slices")
    print(f"Test: {len(test_volumes)} volumes, {len(test_files)} slices")
    
    # Write output files
    write_file_list(train_files, os.path.join(output_dir, "train_slices.list"))
    write_file_list(val_files, os.path.join(output_dir, "val_slices.list"))
    write_file_list(test_files, os.path.join(output_dir, "test_slices.list"))
    
    print(f"\nOutput files written to:")
    print(f"  Train: {os.path.join(output_dir, 'train_slices.list')}")
    print(f"  Validation: {os.path.join(output_dir, 'val_slices.list')}")
    print(f"  Test: {os.path.join(output_dir, 'test_slices.list')}")
    
    # Print some sample volumes for verification
    print(f"\nSample train volumes: {train_volumes[:5]}")
    print(f"Sample validation volumes: {val_volumes[:5]}")
    print(f"Sample test volumes: {test_volumes[:5]}")

if __name__ == "__main__":
    main() 