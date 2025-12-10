from tqdm import tqdm
import numpy as np
import gc
from utils.util_funcs import warp_image_affine
from rust_lib import run_y_correction_compute_rust

## Y-Motion Functions (Memory optimized and vectorized)

def all_trans_y(data, static_y_motion,disable_tqdm,scan_num):
    data_depth_z = data.shape[0]
    transforms_all = np.tile(np.eye(3),(data_depth_z,1,1))
    sampled_data = data[:,:,::20] # dont need too much info surface registration
    static_data = sampled_data[static_y_motion,:,:]
    del data
    
    computed_transforms = run_y_correction_compute_rust(static_data, sampled_data)
    transforms_all[:,1,2] = np.array(computed_transforms)[:,1]

    del sampled_data, static_data
    gc.collect()
    return transforms_all

def y_motion_correcting(data, slice_coords, top_surf, partition_coord, disable_tqdm, scan_num):
    """Memory optimized Y-motion correction that works in-place."""
    # Create view of sliced data to avoid unnecessary copy
    slice_indices = np.r_[tuple(np.r_[start:end] for start, end in slice_coords)]
    temp_sliced_data = data[:, slice_indices, :]

    # Find reference B-scan for motion correction
    static_y_motion = np.argmax(np.sum(temp_sliced_data, axis=(1, 2)))

    # Compute all transforms
    tr_y = all_trans_y(temp_sliced_data, static_y_motion, disable_tqdm, scan_num)

    # Clean up the temporary sliced data reference
    del temp_sliced_data

    # Apply transforms in-place based on partition settings
    if partition_coord is None:
        # Apply to entire dataset
        for i in tqdm(range(data.shape[0]), desc='Y-motion warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            # data[i] = warp(data[i], AffineTransform(matrix=tr_y[i]), order=3)
            data[:, :, i] = warp_image_affine(data[:, :, i], (0, -tr_y[i, 1, 2]))
        gc.collect()
        return data
    elif top_surf:
        # Apply only to top portion
        for i in tqdm(range(data.shape[0]), desc='Y-motion warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i, :partition_coord] = warp_image_affine(data[i, :partition_coord], (0, -tr_y[i, 1, 2]))
    else:
        # Apply only to bottom portion
        for i in tqdm(range(data.shape[0]), desc='Y-motion warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i, partition_coord:] = warp_image_affine(data[i, partition_coord:], (0, -tr_y[i, 1, 2]))

    # Clean up transforms array
    gc.collect()
    return data
