from tqdm import tqdm
import numpy as np
import gc
from utils.util_funcs import warp_image_affine
from rust_lib import run_flat_correction_compute_rust

## Flattening Functions
def all_tran_flat(data, static_flat, disable_tqdm, scan_num):
    data_depth_y = data.shape[2]
    transforms_all = np.tile(np.eye(3),(data_depth_y,1,1))
    sampled_data = data[::20]
    static_data = sampled_data[:,:,static_flat]
    del data

    computed_transforms = run_flat_correction_compute_rust(static_data, sampled_data)
    transforms_all[:,0,2] = np.array(computed_transforms)[:,0]

    del sampled_data, static_data
    gc.collect()
    return transforms_all

def flatten_data(data, slice_coords, top_surf, partition_coord, disable_tqdm, scan_num):
    """Memory optimized flattening that works in-place."""
    # Create view of sliced data to avoid unnecessary copy
    slice_indices = np.r_[tuple(np.r_[start:end] for start, end in slice_coords)]
    temp_sliced_data = data[:, slice_indices, :]  # This is a view, not a copy

    # Find reference slice for flattening
    static_flat = np.argmax(np.sum(temp_sliced_data, axis=(0, 1)))

    # Compute all transforms at once
    tr_flat = all_tran_flat(temp_sliced_data, static_flat, disable_tqdm, scan_num)

    # Clean up the temporary sliced data reference
    del temp_sliced_data

    # Apply transforms in-place based on partition settings
    if partition_coord is None:
        # Apply to entire data volume
        for i in tqdm(range(data.shape[2]), desc='Flat warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:, :, i] = warp_image_affine(data[:, :, i], (-tr_flat[i, 0, 2], 0))
        gc.collect()
        return data
    elif top_surf:
        # Apply only to top portion
        for i in tqdm(range(data.shape[2]), desc='Flat warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:, :partition_coord, i] = warp_image_affine(data[:, :partition_coord, i], (-tr_flat[i, 0, 2], 0))
    else:
        # Apply only to bottom portion
        for i in tqdm(range(data.shape[2]), desc='Flat warping', disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:, partition_coord:, i] = warp_image_affine(data[:, partition_coord:, i], (-tr_flat[i, 0, 2], 0))

    gc.collect()
    return data
