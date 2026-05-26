from skimage.transform import warp, AffineTransform
import numpy as np
from utils.util_funcs import ncc
from scipy.optimize import minimize as minz
from scipy import ndimage as scp
from utils.util_funcs import warp_image_affine
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utils.native_dlls import configure_torch_dll_path
configure_torch_dll_path()
from rust_lib import run_x_correction_compute_rust

## X-Motion Functions (Memory optimized and vectorized)

def shift_func(shif, x, y, past_shift):
    """Optimized shift function for line-based corrections."""
    warped_x = scp.shift(x, -shif[0]-past_shift, order=3, mode='nearest')
    warped_y = scp.shift(y, shif[0]+past_shift, order=3, mode='nearest')

    corr = ncc(warped_x, warped_y)
    return 1 - corr

def err_fun_x(shif, x, y, past_shift):
    """Optimized error function for patch-based corrections."""

    warped_x = warp_image_affine(x, [-shif[0]-past_shift, 0])
    warped_y = warp_image_affine(y, [shif[0]+past_shift, 0])

    corr = ncc(warped_x, warped_y)
    return float(1 - corr)

def get_line_shift(line_1d_stat, line_1d_mov):
    past_shift = 0
    for _ in range(7):
        move = minz(method='powell',fun = shift_func,x0 = np.array([0.0]),bounds =[(-4,4)],
                args = (line_1d_stat
                        ,line_1d_mov
                        ,past_shift))['x']
        past_shift += move[0]
    return -past_shift*2 # Negative because scipy shift returns opposite direction shift

def get_cell_patch_shift(patch_stat, patch__mov):
    past_shift = 0
    for _ in range(7):
        move = minz(method='powell',fun = err_fun_x,x0 = np.array([0.0]), bounds=[(-4,4)],
                    args = (patch_stat
                            ,patch__mov
                            ,past_shift))['x']
        past_shift += move[0]
    return past_shift*2

def check_best_warp(stat, mov, value, is_shift_value = False):
    err = ncc(stat,warp(mov, AffineTransform(translation=(-value,0)),order=3))
    return err

def check_multiple_warps(stat_img, mov_img, *args):
    errors = []
    warps = args[0]
    for warp_value in range(len(warps)):
        errors.append(check_best_warp(stat_img, mov_img, warps[warp_value]))
    return np.argmax(errors)

def compute_transform_for_pair(i, static_img, moving_img, cells_coords, enface_extraction_rows, valid_args):
    if i not in valid_args:
        return AffineTransform(translation=(0,0))
    if (cells_coords is not None):
        # if MODEL_X_TRANSLATION is not None:
        #     cell_warps = x_correction_cell_model(static_img, moving_img, cells_coords, MODEL_X_TRANSLATION)
        # else:
        cell_warps = x_correction_cell_manual(static_img, moving_img, cells_coords)
    else:
        cell_warps = [(float('inf'), 0.0)]
    if len(enface_extraction_rows)>0:
        # if MODEL_X_TRANSLATION is not None:
        #     enface_wraps = x_correction_enface_model(static_img, moving_img, enface_extraction_rows, MODEL_X_TRANSLATION)
        # else:
        enface_wraps = x_correction_enface_manual(static_img, moving_img, enface_extraction_rows)
    else:
        enface_wraps = [(float('inf'), 0.0)]
    all_warps = [*cell_warps,*enface_wraps]
    all_warps = sorted(all_warps, key=lambda x: x[0])  # Sort by error
    temp_tform_manual = AffineTransform(translation=(all_warps[0][1],0))
    return temp_tform_manual

# def x_correction_cell_model(static_img, moving_img, cells_coords, MODEL_X_TRANSLATION):
#     cell_warps = []
#     for UP_x, DOWN_x in cells_coords:
#         stat = static_img[UP_x:DOWN_x, :]
#         temp_manual = moving_img[UP_x:DOWN_x, :]
#         temp_cell_shift, inv_temp_cell_shift = infer_x_translation(MODEL_X_TRANSLATION, stat, temp_manual)
#         error_cell = abs(temp_cell_shift + inv_temp_cell_shift)
#         cell_warps.append((error_cell, temp_cell_shift))
#     return cell_warps

def x_correction_cell_manual(static_img, moving_img, cells_coords):
    if cells_coords.shape[0]==1:
        UP_x, DOWN_x = cells_coords[0,0], cells_coords[0,1]
        stat = static_img[ UP_x:DOWN_x, :]
        temp_manual = moving_img[ UP_x:DOWN_x, :]
    else:
        stat = static_img[np.r_[tuple(np.r_[start:end] for start, end in cells_coords)],:]
        temp_manual = moving_img[np.r_[tuple(np.r_[start:end] for start, end in cells_coords)],:]
    # MANUAL
    temp_cell_patch_shift = get_cell_patch_shift(stat,temp_manual)
    inv_temp_cell_patch_shift = get_cell_patch_shift(temp_manual,stat)
    error_cell = abs(temp_cell_patch_shift + inv_temp_cell_patch_shift)
    cell_warps = [(error_cell, temp_cell_patch_shift)]
    return cell_warps

# def x_correction_enface_model(static_img, moving_img, enface_extraction_rows, MODEL_X_TRANSLATION):
#     enface_wraps = []
#     for enf_idx in range(len(enface_extraction_rows)):
#         bottom_row = max(0, enface_extraction_rows[enf_idx]-32)
#         stat = static_img[bottom_row:enface_extraction_rows[enf_idx]+32]
#         temp_manual = moving_img[bottom_row:enface_extraction_rows[enf_idx]+32]
#         temp_enface_shift, inv_temp_enface_shift = infer_x_translation(MODEL_X_TRANSLATION, stat, temp_manual)
#         error_enface = abs(temp_enface_shift + inv_temp_enface_shift)
#         enface_wraps.append((error_enface, temp_enface_shift))
#     return enface_wraps

def x_correction_enface_manual(static_img, moving_img, enface_extraction_rows):
    enface_wraps = []
    for enf_idx in range(len(enface_extraction_rows)):
        stat = static_img[ enface_extraction_rows[enf_idx]]
        temp_manual = moving_img[ enface_extraction_rows[enf_idx]]
        temp_enface_shift = get_line_shift(stat, temp_manual)
        inv_temp_enface_shift = get_line_shift(temp_manual, stat)
        error_enface = abs(temp_enface_shift + inv_temp_enface_shift)
        enface_wraps.append((error_enface, temp_enface_shift))
    return enface_wraps

def x_motion_correction(data, cells_coords, valid_args, enface_extraction_rows, disable_tqdm, scan_num, MODEL_X_TRANSLATION_PATH):
    tr_x = np.tile(np.eye(3),(data.shape[0],1,1))
    indices = [i for i in range(0, data.shape[0]-1, 2)]
    static_imgs = data[indices]
    moving_imgs = data[np.array(indices) + 1]


    if (MODEL_X_TRANSLATION_PATH is not None) and (cells_coords is not None):
        transform_results = run_x_correction_compute_rust(
            stat_data = static_imgs, 
            mov_data = moving_imgs, 
            indices = indices, 
            enface_extraction_rows = enface_extraction_rows,
            cells_coords = cells_coords, 
            valid_args = valid_args,
            model_path = MODEL_X_TRANSLATION_PATH)
        for i, result in zip(indices, transform_results):
            tr_x[i+1][0,2] = result[0]
    else:
        with ThreadPoolExecutor() as executor:
            compute_fn = partial(compute_transform_for_pair, cells_coords=cells_coords, enface_extraction_rows=enface_extraction_rows, valid_args=valid_args)
            results = list(executor.map(compute_fn, indices, static_imgs, moving_imgs))
        for i, result in zip(indices, results):
            tr_x[i+1] = np.dot(tr_x[i+1], result)
    return tr_x
