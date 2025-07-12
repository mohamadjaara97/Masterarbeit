import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import re

def normalize_image(image):
    """Normalize image to [0,1] range"""
    return (image - image.min()) / (image.max() - image.min())

def get_case_number(filename, prefix):
    """Extract the case number from a filename given a prefix (e.g., 'volume-' or 'segmentation-')"""
    match = re.match(rf"{prefix}(\d+)", os.path.basename(filename))
    return match.group(1) if match else None

def process_lits_data(base_dir, output_dir):
    """Process LiTS data into 2D slices and save as HDF5 files"""
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data/slices'), exist_ok=True)
    
    # Get all CT and mask volumes
    ct_files = sorted(glob.glob(os.path.join(base_dir, 'train_CT', 'volume-*.nii')))
    mask_files = sorted(glob.glob(os.path.join(base_dir, 'train_mask', 'segmentation-*.nii')))
    
    print(f"Found {len(ct_files)} CT files and {len(mask_files)} mask files")
    
    # Build a mapping from case number to file path
    ct_dict = {get_case_number(f, 'volume-'): f for f in ct_files}
    mask_dict = {get_case_number(f, 'segmentation-'): f for f in mask_files}
    
    # Remove None keys
    ct_dict = {k: v for k, v in ct_dict.items() if k is not None}
    mask_dict = {k: v for k, v in mask_dict.items() if k is not None}
    
    print(f"First 5 CT dict: {list(ct_dict.items())[:5]}")
    print(f"First 5 mask dict: {list(mask_dict.items())[:5]}")
    
    print(f"Found {len(ct_dict)} CT cases and {len(mask_dict)} mask cases")
    
    # Find common case numbers
    common_cases = sorted(set(ct_dict.keys()) & set(mask_dict.keys()), key=lambda x: int(x))
    
    print(f"Found {len(common_cases)} matching cases")
    if len(common_cases) > 0:
        print(f"First few cases: {common_cases[:5]}")
    
    total_slices = 0
    train_list = []
    val_list = []
    
    # Process each volume
    for case_num in tqdm(common_cases):
        ct_file = ct_dict[case_num]
        mask_file = mask_dict[case_num]
        
        print(f"\nProcessing case {case_num}")
        print(f"CT file: {ct_file}")
        print(f"Mask file: {mask_file}")
        
        # Read volume
        ct_volume = sitk.GetArrayFromImage(sitk.ReadImage(ct_file))
        mask_volume = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        
        print(f"CT volume shape: {ct_volume.shape}")
        print(f"Mask volume shape: {mask_volume.shape}")
        
        # Normalize CT volume
        ct_volume = normalize_image(ct_volume)
        
        # Convert to float32
        ct_volume = ct_volume.astype(np.float32)
        
        # Get case name
        case_name = f"volume-{case_num}"
        
        # Save each slice
        for slice_idx in range(ct_volume.shape[0]):
            slice_name = f"{case_name}_slice_{slice_idx}"
            
            # Save as HDF5
            with h5py.File(os.path.join(output_dir, 'data/slices', f'{slice_name}.h5'), 'w') as f:
                f.create_dataset('image', data=ct_volume[slice_idx], compression="gzip")
                f.create_dataset('label', data=mask_volume[slice_idx], compression="gzip")
            
            total_slices += 1
            train_list.append(slice_name)
    
    # Split into train and validation sets (80/20 split)
    np.random.shuffle(train_list)
    split_idx = int(len(train_list) * 0.8)
    train_list, val_list = train_list[:split_idx], train_list[split_idx:]
    
    # Save train and validation lists
    with open(os.path.join(output_dir, 'train.list'), 'w') as f:
        f.write('\n'.join(train_list))
    
    with open(os.path.join(output_dir, 'val.list'), 'w') as f:
        f.write('\n'.join(val_list))
    
    # Save train_slices.list (same as train.list for LiTS)
    with open(os.path.join(output_dir, 'train_slices.list'), 'w') as f:
        f.write('\n'.join(train_list))
    
    print(f"\nProcessed {len(common_cases)} volumes into {total_slices} slices")
    print(f"Training set: {len(train_list)} slices")
    print(f"Validation set: {len(val_list)} slices")

if __name__ == "__main__":
    base_dir = "/home/mohamad/Masterarbeit/LiTS(train_test)"  # Absolute path to your LiTS dataset
    output_dir = "/home/mohamad/Masterarbeit/SSL4MIS/data/LiTS"  # Absolute path to output directory
    process_lits_data(base_dir, output_dir)