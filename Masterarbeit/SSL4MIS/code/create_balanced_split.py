import os
import h5py
import numpy as np
from collections import defaultdict, Counter
import random
from tqdm import tqdm

def analyze_volume_distribution(data_root_path):
    """Analyze which volumes contain which classes"""
    print("Analyzing volume distribution...")
    
    # Read all slice names
    train_list_file = os.path.join(data_root_path, 'train_slices_new.list')
    val_list_file = os.path.join(data_root_path, 'val_new.list')
    
    all_slices = []
    if os.path.exists(train_list_file):
        with open(train_list_file, 'r') as f:
            all_slices.extend([line.strip() for line in f.readlines()])
    
    if os.path.exists(val_list_file):
        with open(val_list_file, 'r') as f:
            all_slices.extend([line.strip() for line in f.readlines()])
    
    # Group slices by volume
    volume_slices = defaultdict(list)
    for slice_name in all_slices:
        volume_name = slice_name.split('_slice_')[0]
        volume_slices[volume_name].append(slice_name)
    
    # Analyze each volume
    volume_analysis = {}
    print(f"Analyzing {len(volume_slices)} volumes...")
    
    for volume_name, slice_names in tqdm(volume_slices.items(), desc="Processing volumes"):
        volume_classes = set()
        total_pixels_per_class = defaultdict(int)
        
        for slice_name in slice_names:
            h5_file_path = os.path.join(data_root_path, "data", "slices", f"{slice_name}.h5")
            if os.path.exists(h5_file_path):
                try:
                    with h5py.File(h5_file_path, 'r') as h5f:
                        if 'label' in h5f:
                            label = h5f['label'][:]
                            unique_labels = np.unique(label)
                            for label_val in unique_labels:
                                if label_val > 0:  # Skip background
                                    volume_classes.add(int(label_val))
                                    total_pixels_per_class[int(label_val)] += np.sum(label == label_val)
                except Exception as e:
                    print(f"Error reading {h5_file_path}: {e}")
        
        volume_analysis[volume_name] = {
            'classes': list(volume_classes),
            'num_classes': len(volume_classes),
            'total_slices': len(slice_names),
            'pixels_per_class': dict(total_pixels_per_class)
        }
    
    return volume_analysis

def create_balanced_split(volume_analysis, train_ratio=0.8):
    """Create balanced train/val split based on volumes"""
    print(f"\nCreating balanced split with {train_ratio*100:.0f}% train...")
    
    # Separate volumes by class content
    volumes_with_class1 = [v for v, info in volume_analysis.items() if 1 in info['classes']]
    volumes_with_class2 = [v for v, info in volume_analysis.items() if 2 in info['classes']]
    volumes_with_both = [v for v, info in volume_analysis.items() if 1 in info['classes'] and 2 in info['classes']]
    volumes_background_only = [v for v, info in volume_analysis.items() if len(info['classes']) == 0]
    
    print(f"Volumes with class 1: {len(volumes_with_class1)}")
    print(f"Volumes with class 2: {len(volumes_with_class2)}")
    print(f"Volumes with both classes: {len(volumes_with_both)}")
    print(f"Volumes with background only: {len(volumes_background_only)}")
    
    # Create balanced split
    random.seed(42)  # For reproducibility
    
    # Split volumes with class 1
    train_volumes_class1 = random.sample(volumes_with_class1, 
                                        int(len(volumes_with_class1) * train_ratio))
    val_volumes_class1 = [v for v in volumes_with_class1 if v not in train_volumes_class1]
    
    # Split volumes with class 2
    train_volumes_class2 = random.sample(volumes_with_class2, 
                                        int(len(volumes_with_class2) * train_ratio))
    val_volumes_class2 = [v for v in volumes_with_class2 if v not in train_volumes_class2]
    
    # Split volumes with both classes
    train_volumes_both = random.sample(volumes_with_both, 
                                      int(len(volumes_with_both) * train_ratio))
    val_volumes_both = [v for v in volumes_with_both if v not in train_volumes_both]
    
    # Split background-only volumes
    train_volumes_bg = random.sample(volumes_background_only, 
                                    int(len(volumes_background_only) * train_ratio))
    val_volumes_bg = [v for v in volumes_background_only if v not in train_volumes_bg]
    
    # Combine all train and val volumes
    train_volumes = set(train_volumes_class1 + train_volumes_class2 + 
                       train_volumes_both + train_volumes_bg)
    val_volumes = set(val_volumes_class1 + val_volumes_class2 + 
                      val_volumes_both + val_volumes_bg)
    
    return train_volumes, val_volumes

def create_slice_lists(data_root_path, train_volumes, val_volumes, volume_analysis):
    """Create new train_slices.list and val.list files"""
    print("\nCreating new slice lists...")
    
    # Read all slice names
    train_list_file = os.path.join(data_root_path, 'train_slices_new.list')
    val_list_file = os.path.join(data_root_path, 'val_new.list')
    
    all_slices = []
    if os.path.exists(train_list_file):
        with open(train_list_file, 'r') as f:
            all_slices.extend([line.strip() for line in f.readlines()])
    
    if os.path.exists(val_list_file):
        with open(val_list_file, 'r') as f:
            all_slices.extend([line.strip() for line in f.readlines()])
    
    # Group slices by volume
    volume_slices = defaultdict(list)
    for slice_name in all_slices:
        volume_name = slice_name.split('_slice_')[0]
        volume_slices[volume_name].append(slice_name)
    
    # Create new lists
    new_train_slices = []
    new_val_slices = []
    
    for volume_name, slice_names in volume_slices.items():
        if volume_name in train_volumes:
            new_train_slices.extend(slice_names)
        elif volume_name in val_volumes:
            new_val_slices.extend(slice_names)
    
    # Sort for reproducibility
    new_train_slices.sort()
    new_val_slices.sort()
    
    # Write new files
    backup_dir = os.path.join(data_root_path, 'backup_original')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Backup original files
    if os.path.exists(train_list_file):
        os.system(f'cp "{train_list_file}" "{backup_dir}/train_slices.list.backup"')
    if os.path.exists(val_list_file):
        os.system(f'cp "{val_list_file}" "{backup_dir}/val.list.backup"')
    
    # Write new train file
    with open(train_list_file, 'w') as f:
        for slice_name in new_train_slices:
            f.write(f"{slice_name}\n")
    
    # Write new val file
    with open(val_list_file, 'w') as f:
        for slice_name in new_val_slices:
            f.write(f"{slice_name}\n")
    
    print(f"New train_slices.list: {len(new_train_slices)} slices")
    print(f"New val.list: {len(new_val_slices)} slices")
    print(f"Original files backed up to: {backup_dir}")
    
    return new_train_slices, new_val_slices

def main():
    data_root_path = 'data/LiTS'
    
    print("Creating Balanced LiTS Dataset Split")
    print("="*50)
    
    # Analyze current volume distribution
    volume_analysis = analyze_volume_distribution(data_root_path)
    
    # Create balanced split
    train_volumes, val_volumes = create_balanced_split(volume_analysis, train_ratio=0.8)
    
    # Create new slice lists
    new_train_slices, new_val_slices = create_slice_lists(data_root_path, train_volumes, val_volumes, volume_analysis)
    
    # Analyze new split
    print(f"\n{'='*50}")
    print("ANALYSIS OF NEW SPLIT")
    print(f"{'='*50}")
    
    # Analyze train volumes
    train_class_counts = defaultdict(int)
    for volume in train_volumes:
        if volume in volume_analysis:
            for class_id in volume_analysis[volume]['classes']:
                train_class_counts[class_id] += 1
    
    # Analyze val volumes
    val_class_counts = defaultdict(int)
    for volume in val_volumes:
        if volume in volume_analysis:
            for class_id in volume_analysis[volume]['classes']:
                val_class_counts[class_id] += 1
    
    print(f"Train volumes: {len(train_volumes)}")
    print(f"Val volumes: {len(val_volumes)}")
    print(f"Train slices: {len(new_train_slices)}")
    print(f"Val slices: {len(new_val_slices)}")
    
    print(f"\nVolumes containing each class:")
    for class_id in [1, 2]:
        print(f"- Class {class_id}: Train={train_class_counts[class_id]}, Val={val_class_counts[class_id]}")
    
    print(f"\nSplit completed successfully!")

if __name__ == "__main__":
    main() 