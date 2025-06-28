import gc
from pydicom import dcmread
from skimage.transform import warp, AffineTransform
from tqdm import tqdm
import numpy as np
from utils.util_funcs import *
from collections import defaultdict
from scipy.optimize import minimize as minz
from scipy import ndimage as scp
import h5py
import torch
from torchvision import transforms
import torch.nn.functional as F

## Flattening Functions
 
def mse_fun_tran_flat(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(-past_shift,0)),order=1)
    y = warp(y, AffineTransform(translation=(past_shift,0)),order=1)
    warped_x_stat = warp(x, AffineTransform(translation=(-shif[0],0)),order=1)
    warped_y_mov = warp(y, AffineTransform(translation=(shif[0],0)),order=1)
    err = np.squeeze(1-ncc(warped_x_stat ,warped_y_mov))
    return float(err)
    
def all_tran_flat(data,static_flat,disable_tqdm, scan_num):
    transforms_all = np.tile(np.eye(3),(data.shape[2],1,1))
    for i in tqdm(range(data.shape[2]),desc='Flattening surfaces',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        try:
            # stat = data[:,UP_flat:DOWN_flat,static_flat][::20].copy()
            # temp_img = data[:,UP_flat:DOWN_flat,i][::20].copy()
            stat = data[:,:,static_flat][::20]
            temp_img = data[:,:,i][::20]
            # MANUAL
            past_shift = 0
            for _ in range(10):
                move = minz(method='powell',fun = mse_fun_tran_flat,x0 = np.array([0.0]), bounds=[(-4,4)],
                            args = (stat
                                    ,temp_img
                                    ,past_shift))['x']

                past_shift += move[0]
            temp_tform_manual = AffineTransform(translation=(past_shift*2,0))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
        except:
            with open(f'debugs/debug{scan_num}.txt', 'a') as f:
                f.write(f'FLAT motion EVERYTHIN FAILED HERE\n')
                f.write(f'NAME: {scan_num}\n')
                f.write(f'Ith: {i}\n')
            temp_tform_manual = AffineTransform(translation=(0,0))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
    return transforms_all

def flatten_data(data,slice_coords,top_surf, partition_coord,disable_tqdm, scan_num):
    temp_sliced_data = data[:, np.r_[tuple(np.r_[start:end] for start, end in slice_coords)], :].copy()
    static_flat = np.argmax(np.sum(temp_sliced_data,axis=(0,1)))
    # finding the bright points in all images in standard interference
    # temp_rotated_data = data[:,UP_flat:DOWN_flat,:].transpose(2,1,0)
    # nn = [np.argmax(np.sum(temp_rotated_data[i],axis=1)) for i in range(temp_rotated_data.shape[0])]
    # tf_all_nn = np.tile(np.eye(3),(temp_rotated_data.shape[0],1,1))
    # for i in range(tf_all_nn.shape[0]):
    #     tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(-(nn[0]-nn[i]),0)))
    

    # if partition_coord:
    #     for i in tqdm(range(data.shape[2]),desc='Flat warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
    #         data[:,:DOWN_flat,i]  = warp(data[:,:DOWN_flat,i] ,AffineTransform(matrix=tf_all_nn[i]),order=3)
    # else:
    #     for i in tqdm(range(data.shape[2]),desc='Flat warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
    #         data[:,UP_flat:,i]  = warp(data[:,UP_flat:,i] ,AffineTransform(matrix=tf_all_nn[i]),order=3)
    tr_all = all_tran_flat(temp_sliced_data,static_flat,disable_tqdm,scan_num)
    if top_surf:
        for i in tqdm(range(data.shape[2]),desc='Flat warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:,:partition_coord,i]  = warp(data[:,:partition_coord,i] ,AffineTransform(matrix=tr_all[i]),order=3)
    else:
        for i in tqdm(range(data.shape[2]),desc='Flat warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:,partition_coord:,i]  = warp(data[:,partition_coord:,i] ,AffineTransform(matrix=tr_all[i]),order=3)
    return data

## Y-Motion Functions

def mse_fun_tran_y(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(0,-past_shift)),order=3)
    y = warp(y, AffineTransform(translation=(0,past_shift)),order=3)
    warped_x_stat = warp(x, AffineTransform(translation=(0,-shif[0])),order=3)
    warped_y_mov = warp(y, AffineTransform(translation=(0,shif[0])),order=3)
    err = np.squeeze(1-ncc(warped_x_stat ,warped_y_mov))
    return float(err)

def all_trans_y(data,static_y_motion,disable_tqdm,scan_num):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(data.shape[0]-1),desc='Y-motion Correction',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        try:
            stat = data[static_y_motion][:,::20].copy()
            temp_img = data[i][:,::20].copy()
            # MANUAL
            past_shift = 0
            for _ in range(10):
                move = minz(method='powell',fun = mse_fun_tran_y,x0 = np.array([0.0]), bounds=[(-5,5)],
                            args = (stat
                                    ,temp_img
                                    ,past_shift))['x']
                past_shift += move[0]
            temp_tform_manual = AffineTransform(matrix = AffineTransform(translation=(0,past_shift*2)))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
        except:
            with open(f'debugs/debug{scan_num}.txt', 'a') as f:
                f.write(f'Y motion EVERYTHIN FAILED HERE\n')
                f.write(f'NAME: {scan_num}\n')
                f.write(f'Ith: {i}\n')
            temp_tform_manual = AffineTransform(translation=(0,0))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
    return transforms_all

def y_motion_correcting(data,slice_coords,top_surf,partition_coord,disable_tqdm,scan_num):
    # static_y_motion = np.argmax(np.sum(data[:,UP_y:DOWN_y,:],axis=(1,2)))
    # # finding the bright points in all images in standard interference
    # nn = [np.argmax(np.sum(data[i][UP_y:DOWN_y],axis=1)) for i in range(data.shape[0])]
    # tf_all_nn = np.tile(np.eye(3),(data.shape[0],1,1))
    # for i in range(tf_all_nn.shape[0]):
    #     tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    # if top_surf:
    #     for i in tqdm(range(data.shape[0]),desc='y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
    #         data[i,:DOWN_y]  = warp(data[i,:DOWN_y],AffineTransform(matrix=tf_all_nn[i]),order=3)
    # else:
    #     for i in tqdm(range(data.shape[0]),desc='y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
    #         data[i,UP_y:]  = warp(data[i,UP_y:],AffineTransform(matrix=tf_all_nn[i]),order=3)
    # tr_all_y = all_trans_y(data,UP_y,DOWN_y,static_y_motion,disable_tqdm,scan_num)
    # if top_surf:
    #     for i in tqdm(range(data.shape[0]),desc='y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
    #         data[i,:DOWN_y]  = warp(data[i,:DOWN_y],AffineTransform(matrix=tr_all_y[i]),order=3)
    # else:
    #     for i in tqdm(range(data.shape[0]),desc='y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
    #         data[i,UP_y:]  = warp(data[i,UP_y:],AffineTransform(matrix=tr_all_y[i]),order=3)

    temp_sliced_data = data[:, np.r_[tuple(np.r_[start:end] for start, end in slice_coords)], :].copy()
    static_y_motion = np.argmax(np.sum(temp_sliced_data,axis=(1,2)))
    tr_all_y = all_trans_y(temp_sliced_data,static_y_motion,disable_tqdm,scan_num)
    if top_surf:
        for i in tqdm(range(data.shape[0]),desc='Y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i,:partition_coord]  = warp(data[i,:partition_coord],AffineTransform(matrix=tr_all_y[i]),order=3)
    else:
        for i in tqdm(range(data.shape[0]),desc='Y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i,partition_coord:]  = warp(data[i,partition_coord:],AffineTransform(matrix=tr_all_y[i]),order=3)
    return data

## X-Motion Functions

def shift_func(shif, x, y , past_shift):
    x = scp.shift(x, -past_shift,order=3,mode='nearest')
    y = scp.shift(y, past_shift,order=3,mode='nearest')
    warped_x_stat = scp.shift(x, -shif[0],order=3,mode='nearest')
    warped_y_mov = scp.shift(y, shif[0],order=3,mode='nearest')
    return (1-ncc1d(warped_x_stat ,warped_y_mov))

def ncc1d(array1, array2):
    correlation = np.correlate(array1, array2, mode='valid')
    array1_norm = np.linalg.norm(array1)
    array2_norm = np.linalg.norm(array2)
    if array1_norm == 0 or array2_norm == 0:
        return np.zeros_like(correlation)
    normalized_correlation = correlation / (array1_norm * array2_norm)
    return normalized_correlation

def mse_fun_tran_x(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(-past_shift,0)),order=3)
    y = warp(y, AffineTransform(translation=(past_shift,0)),order=3)
    warped_x_stat = warp(x, AffineTransform(translation=(-shif[0],0)),order=3)
    warped_y_mov = warp(y, AffineTransform(translation=(shif[0],0)),order=3)
    err = np.squeeze(1-ncc(warped_x_stat ,warped_y_mov))
    return float(err)

def get_line_shift(line_1d_stat, line_1d_mov,enface_shape):
    st = line_1d_stat
    mv = line_1d_mov
    past_shift = 0
    for _ in range(10):
        move = minz(method='powell',fun = shift_func,x0 = np.array([0.0]),bounds =[(-4,4)],
                args = (st
                        ,mv
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

def all_trans_x(data,UP_x,DOWN_x,valid_args,enface_extraction_rows,disable_tqdm,scan_num, MODEL_X_TRANSLATION):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(0,data.shape[0]-1,2),desc='X-motion Correction',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        try:
            if i not in valid_args:
                continue
            try:
                if (UP_x is not None) and (DOWN_x is not None):
                    UP_x , DOWN_x = np.squeeze(np.array(UP_x)), np.squeeze(np.array(DOWN_x))
                    if UP_x.size>1 and DOWN_x.size>1:
                        stat = data[i,np.r_[UP_x[0]:DOWN_x[0],UP_x[1]:DOWN_x[1]],:]
                        temp_manual = data[i+1,np.r_[UP_x[0]:DOWN_x[0],UP_x[1]:DOWN_x[1]],:]
                    else:
                        stat = data[i,UP_x:DOWN_x,:]
                        temp_manual = data[i+1,UP_x:DOWN_x,:]
                    # MANUAL
                    temp_tform_manual = AffineTransform(translation=(0,0))
                    past_shift = 0
                    if MODEL_X_TRANSLATION is not None:
                        past_shift = np.squeeze(infer_x_translation(MODEL_X_TRANSLATION, stat, temp_manual, DEVICE = 'cpu'))[0]
                        cross_section = -(past_shift)
                    else:
                        for _ in range(10):
                            move = minz(method='powell',fun = mse_fun_tran_x,x0 = np.array([0.0]), bounds=[(-4,4)],
                                        args = (stat
                                                ,temp_manual
                                                ,past_shift))['x']
                            past_shift += move[0]
                        cross_section = -(past_shift*2)
                else:
                    cross_section = 0
            except Exception as e:
                with open(f'debugs/debug{scan_num}.txt', 'a') as f:
                    f.write(f'Cell cross_section failed here\n')
                    f.write(f'UP_x: {UP_x}, DOWN_x: {DOWN_x}\n')
                    f.write(f'NAME: {scan_num}\n')
                    f.write(f'Ith: {i}\n')
                    f.write(f'enface_extraction_rows: {enface_extraction_rows}\n')
                raise Exception(e)
                cross_section = 0
            enface_shape = data[:,0,:].shape[1]
            enface_wraps = []
            if len(enface_extraction_rows)>0:
                for enf_idx in range(len(enface_extraction_rows)):
                    try:
                        if MODEL_X_TRANSLATION is not None:
                            bottom_row = min(0, enface_extraction_rows[enf_idx]-20)
                            temp_enface_shift = np.squeeze(infer_x_translation(MODEL_X_TRANSLATION, data[i,bottom_row:enface_extraction_rows[enf_idx]+20]
                                                                                                    ,data[i+1,bottom_row:enface_extraction_rows[enf_idx]+20]
                                                                                                    ,DEVICE = 'cpu'))[0]
                        else:
                            temp_enface_shift = get_line_shift(data[i,enface_extraction_rows[enf_idx]]
                                                               ,data[i+1,enface_extraction_rows[enf_idx]],enface_shape)
                    except Exception as e:
                        with open(f'debugs/debug{scan_num}.txt', 'a') as f:
                            f.write(f'TEMP enface shift failed here\n')
                            f.write(f'UP_x: {UP_x}, DOWN_x: {DOWN_x}\n')
                            f.write(f'NAME: {scan_num}\n')
                            f.write(f'Ith: {i}\n')
                            f.write(f'enface_extraction_rows: {enface_extraction_rows}\n')
                        raise Exception(e)
                        temp_enface_shift = 0
                    enface_wraps.append(temp_enface_shift)
            all_warps = [cross_section,*enface_wraps]
            best_warp = check_multiple_warps(data[i], data[i+1], all_warps)
            temp_tform_manual = AffineTransform(translation=(-(all_warps[best_warp]),0))
            transforms_all[i+1] = np.dot(transforms_all[i+1],temp_tform_manual)
            gc.collect()
        except Exception as e:
            with open(f'debugs/debug{scan_num}.txt', 'a') as f:
                f.write(f'X motion EVERYTHIN FAILED HERE\n')
                f.write(f'UP_x: {UP_x}, DOWN_x: {DOWN_x}\n')
                f.write(f'NAME: {scan_num}\n')
                f.write(f'Ith: {i}\n')
                f.write(f'enface_extraction_rows: {enface_extraction_rows}\n')
            raise Exception(e)
            temp_tform_manual = AffineTransform(translation=(0,0))
            transforms_all[i+1] = np.dot(transforms_all[i+1],temp_tform_manual)
    return transforms_all

## Misc Functions

def filter_list(result_list,expected_num):
    grouped = defaultdict(list)
    for item in result_list:
        grouped[item['name']].append(item)
    filtered_summary = []
    for group in grouped.values():
        top_two = sorted(group, key=lambda x: x['confidence'], reverse=True)[:expected_num]
        filtered_summary.extend(top_two)
    return filtered_summary

def detect_areas(result_list, pad_val, img_shape, expected_num = 2):
    if len(result_list)==0:
        return None
    result_list = filter_list(result_list, expected_num)
    coords = []
    for detections in result_list:
        coords.append([int(detections['box']['y1'])-pad_val,int(detections['box']['y2'])+pad_val])
    if len(coords)==0:
        return None
    coords = np.squeeze(np.array(coords))
    coords = np.where(coords<0,0,coords)
    coords = np.where(coords>img_shape,img_shape-1,coords)
    if coords.ndim==1:
        coords = coords.reshape(1,-1)
    if coords.shape[0]>1:
        coords = np.sort(coords,axis=0)
    return coords

def crop_data(data,surface_coords,cells_coords,max_crop_shape):
    uncroped_data = data.copy()
    merged_coords = []
    if surface_coords is not None:
        surface_coords[:,0],surface_coords[:,1] = surface_coords[:,0]-30, surface_coords[:,1]+30
        surface_coords = np.where(surface_coords<0,0,surface_coords)
        surface_coords = np.where(surface_coords>max_crop_shape,max_crop_shape-1,surface_coords)
        merged_coords.extend([*surface_coords])
    if cells_coords is not None:
        cells_coords[:,0],cells_coords[:,1] = cells_coords[:,0]-30, cells_coords[:,1]+30
        cells_coords = np.where(cells_coords<0,0,cells_coords)
        cells_coords = np.where(cells_coords>max_crop_shape,max_crop_shape-1,cells_coords)
        merged_coords.extend([*cells_coords])
    merged_coords = merge_intervals([*merged_coords])
    uncroped_data = uncroped_data[:, np.r_[tuple(np.r_[start:end] for start, end in merged_coords)], :]
    return uncroped_data

class CropOrPad():
    def __init__(self, target_shape: tuple):
        if not isinstance(target_shape, (tuple, list)) or len(target_shape) != 2:
            raise ValueError("target_shape must be a tuple or list of two integers (height, width).")
        self.target_height, self.target_width = target_shape

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        is_grayscale = False
        if img.dim() == 2: # (H, W) grayscale
            is_grayscale = True
            img = img.unsqueeze(0) # Add a channel dimension: (1, H, W)
        elif img.dim() == 3: # (C, H, W) color
            pass
        else:
            raise ValueError(f"Unsupported image tensor dimensions: {img.dim()}. Expected 2 or 3.")

        current_channels, current_height, current_width = img.shape

        # --- Padding Logic ---
        pad_top = max(0, (self.target_height - current_height) // 2)
        pad_bottom = max(0, self.target_height - current_height - pad_top)
        pad_left = max(0, (self.target_width - current_width) // 2)
        pad_right = max(0, self.target_width - current_width - pad_left)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            # F.pad expects padding in the order (left, right, top, bottom) for 2D spatial dims
            img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # --- Cropping Logic ---
        # Recalculate dimensions after potential padding
        _, current_height_padded, current_width_padded = img.shape

        if current_height_padded > self.target_height or current_width_padded > self.target_width:
            crop_start_h = max(0, (current_height_padded - self.target_height) // 2)
            crop_end_h = crop_start_h + self.target_height
            crop_start_w = max(0, (current_width_padded - self.target_width) // 2)
            crop_end_w = crop_start_w + self.target_width

            # Crop the image
            img = img[:, crop_start_h:crop_end_h, crop_start_w:crop_end_w]

        if is_grayscale:
            img = img.squeeze(0) # Remove the channel dimension if it was grayscale initially

        return img

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()

    # Prevent division by zero if all values are the same
    if max_val == min_val:
        return torch.zeros_like(tensor)

    return (tensor - min_val) / (max_val - min_val)

transform = transforms.Compose([
    transforms.ToTensor(),
    CropOrPad((64,416)),
])

def infer_x_translation(model_obj, static_np, moving_np, DEVICE):
    # Ensure float32 numpy arrays
    static_np = transform(static_np.astype(np.float32))
    moving_np = transform(moving_np.astype(np.float32))
    
    # Add batch and channel dim: (1, 1, H, W)
    static_np = normalize(static_np.unsqueeze(0)).to(DEVICE)
    moving_np = normalize(moving_np.unsqueeze(0)).to(DEVICE)

    # Concat and infer
    with torch.no_grad():
        input_pair = torch.cat([static_np, moving_np], dim=1).double()  # shape: (1, 2, H, W)
        moved_img, pred_translation = model_obj(input_pair)
        # warped = warper(moving.double(), pred_translation)
    # warped_np = warped.squeeze().numpy()
    return pred_translation.squeeze().numpy()

