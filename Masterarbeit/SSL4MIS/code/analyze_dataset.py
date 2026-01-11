import os
import h5py
import numpy as np
from collections import defaultdict, Counter
import argparse
from tqdm import tqdm

def analyze_slice_labels(h5_file_path):
    """Analyze the labels in a single slice file"""
    try:
        with h5py.File(h5_file_path, 'r') as h5f:
            if 'label' in h5f:
                label = h5f['label'][:]
                unique_labels = np.unique(label)
                label_counts = {}
                for label_val in unique_labels:
                    label_counts[int(label_val)] = np.sum(label == label_val)
                return label_counts
            else:
                return {}
    except Exception as e:
        print(f"Error reading {h5_file_path}: {e}")
        return {}

def analyze_dataset(data_root_path, split_name):
    """Analyze dataset split (train or val)"""
    print(f"\n{'='*50}")
    print(f"Analyzing {split_name} dataset")
    print(f"{'='*50}")
    
    # Read the list file
    if split_name == 'train':
        list_file = os.path.join(data_root_path, f'{split_name}_slices.list')
    else:  # val
        list_file = os.path.join(data_root_path, f'{split_name}.list')
    
    if not os.path.exists(list_file):
        print(f"List file not found: {list_file}")
        return
    
    with open(list_file, 'r') as f:
        slice_names = [line.strip() for line in f.readlines()]
    
    print(f"Total slices in {split_name}: {len(slice_names)}")
    
    # Analyze each slice
    total_slices = len(slice_names)
    slices_with_labels = 0
    class_distribution = defaultdict(int)
    slice_analysis = []
    
    print(f"\nAnalyzing {total_slices} slices...")
    for slice_name in tqdm(slice_names, desc=f"Processing {split_name}"):
        h5_file_path = os.path.join(data_root_path, "data", "slices", f"{slice_name}.h5")
        
        if os.path.exists(h5_file_path):
            label_counts = analyze_slice_labels(h5_file_path)
            
            if label_counts:
                slices_with_labels += 1
                # Count classes in this slice
                for class_id, count in label_counts.items():
                    class_distribution[class_id] += count
                
                # Store analysis for this slice
                slice_analysis.append({
                    'name': slice_name,
                    'classes': list(label_counts.keys()),
                    'total_pixels': sum(label_counts.values())
                })
            else:
                slice_analysis.append({
                    'name': slice_name,
                    'classes': [],
                    'total_pixels': 0
                })
        else:
            print(f"Warning: File not found: {h5_file_path}")
    
    # Print results
    print(f"\nResults for {split_name} dataset:")
    print(f"- Total slices: {total_slices}")
    print(f"- Slices with labels: {slices_with_labels}")
    print(f"- Slices without labels: {total_slices - slices_with_labels}")
    print(f"- Percentage with labels: {(slices_with_labels/total_slices)*100:.2f}%")
    
    print(f"\nClass distribution across all slices:")
    for class_id in sorted(class_distribution.keys()):
        print(f"- Class {class_id}: {class_distribution[class_id]:,} pixels")
    
    # Analyze per-slice class distribution
    class_per_slice = defaultdict(int)
    for slice_info in slice_analysis:
        num_classes = len(slice_info['classes'])
        class_per_slice[num_classes] += 1
    
    print(f"\nSlices by number of classes present:")
    for num_classes in sorted(class_per_slice.keys()):
        print(f"- {num_classes} class(es): {class_per_slice[num_classes]} slices")
    
    # Show some examples
    print(f"\nSample slices with their classes:")
    for i, slice_info in enumerate(slice_analysis[:10]):
        if slice_info['classes']:
            print(f"- {slice_info['name']}: Classes {slice_info['classes']} ({slice_info['total_pixels']} pixels)")
    
    return {
        'total_slices': total_slices,
        'slices_with_labels': slices_with_labels,
        'class_distribution': dict(class_distribution),
        'slice_analysis': slice_analysis
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze LiTS dataset')
    parser.add_argument('--data_root', type=str, 
                       default='../Masterarbeit/SSL4MIS/code/data/LiTS',
                       help='Root path to the dataset')
    args = parser.parse_args()
    
    print("LiTS Dataset Analysis")
    print("="*50)
    
    # Analyze both train and val datasets
    train_results = analyze_dataset(args.data_root, 'train')
    val_results = analyze_dataset(args.data_root, 'val')
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY COMPARISON")
    print(f"{'='*50}")
    
    if train_results and val_results:
        print(f"Train vs Validation comparison:")
        print(f"- Train slices: {train_results['total_slices']:,}")
        print(f"- Val slices: {val_results['total_slices']:,}")
        print(f"- Train with labels: {train_results['slices_with_labels']:,} ({(train_results['slices_with_labels']/train_results['total_slices'])*100:.2f}%)")
        print(f"- Val with labels: {val_results['slices_with_labels']:,} ({(val_results['slices_with_labels']/val_results['total_slices'])*100:.2f}%)")
        
        print(f"\nClass distribution comparison:")
        all_classes = set(train_results['class_distribution'].keys()) | set(val_results['class_distribution'].keys())
        for class_id in sorted(all_classes):
            train_count = train_results['class_distribution'].get(class_id, 0)
            val_count = val_results['class_distribution'].get(class_id, 0)
            print(f"- Class {class_id}: Train={train_count:,}, Val={val_count:,}")

if __name__ == "__main__":
    main() 