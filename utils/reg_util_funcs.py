import gc
from pydicom import dcmread
from skimage.transform import warp, AffineTransform
from tqdm import tqdm
import numpy as np
from util_funcs import *
from collections import defaultdict
from scipy.optimize import minimize as minz
from scipy import ndimage as scp
import h5py

def mse_fun_tran_flat(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(-past_shift,0)),order=1)
    y = warp(y, AffineTransform(translation=(past_shift,0)),order=1)
    warped_x_stat = warp(x, AffineTransform(translation=(-shif[0],0)),order=1)
    warped_y_mov = warp(y, AffineTransform(translation=(shif[0],0)),order=1)
    err = np.squeeze(1-ncc(warped_x_stat ,warped_y_mov))
    return float(err)
    
def all_tran_flat(data,UP_flat,DOWN_flat,static_flat,disable_tqdm, scan_num):
    transforms_all = np.tile(np.eye(3),(data.shape[2],1,1))
    for i in tqdm(range(data.shape[2]),desc='Flattening surfaces',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        try:
            stat = data[:,UP_flat:DOWN_flat,static_flat][::20].copy()
            temp_img = data[:,UP_flat:DOWN_flat,i][::20].copy()
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
                f.write(f'UP_flat: {UP_flat}, DOWN_flat: {DOWN_flat}\n')
                f.write(f'NAME: {scan_num}\n')
                f.write(f'Ith: {i}\n')
            temp_tform_manual = AffineTransform(translation=(0,0))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
    return transforms_all

def flatten_data(data,UP_flat,DOWN_flat,top_surf,disable_tqdm, scan_num):
    static_flat = np.argmax(np.sum(data[:,UP_flat:DOWN_flat,:],axis=(0,1)))
    # finding the bright points in all images in standard interference
    temp_rotated_data = data[:,UP_flat:DOWN_flat,:].transpose(2,1,0)
    nn = [np.argmax(np.sum(temp_rotated_data[i],axis=1)) for i in range(temp_rotated_data.shape[0])]
    tf_all_nn = np.tile(np.eye(3),(temp_rotated_data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(-(nn[0]-nn[i]),0)))
    if top_surf:
        for i in tqdm(range(data.shape[2]),desc='Flat warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:,:DOWN_flat,i]  = warp(data[:,:DOWN_flat,i] ,AffineTransform(matrix=tf_all_nn[i]),order=3)
    else:
        for i in tqdm(range(data.shape[2]),desc='Flat warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:,UP_flat:,i]  = warp(data[:,UP_flat:,i] ,AffineTransform(matrix=tf_all_nn[i]),order=3)
    tr_all = all_tran_flat(data,UP_flat,DOWN_flat,static_flat,disable_tqdm, scan_num)
    if top_surf:
        for i in tqdm(range(data.shape[2]),desc='Flat warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:,:DOWN_flat,i]  = warp(data[:,:DOWN_flat,i] ,AffineTransform(matrix=tr_all[i]),order=3)
    else:
        for i in tqdm(range(data.shape[2]),desc='Flat warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[:,UP_flat:,i]  = warp(data[:,UP_flat:,i] ,AffineTransform(matrix=tr_all[i]),order=3)
    return data

def mse_fun_tran_y(shif, x, y , past_shift):
    x = warp(x, AffineTransform(translation=(0,-past_shift)),order=3)
    y = warp(y, AffineTransform(translation=(0,past_shift)),order=3)
    warped_x_stat = warp(x, AffineTransform(translation=(0,-shif[0])),order=3)
    warped_y_mov = warp(y, AffineTransform(translation=(0,shif[0])),order=3)
    err = np.squeeze(1-ncc(warped_x_stat ,warped_y_mov))
    return float(err)

def all_trans_y(data,UP_y,DOWN_y,static_y_motion,disable_tqdm,scan_num):
    transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in tqdm(range(data.shape[0]-1),desc='Y-motion Correction',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        try:
            stat = data[static_y_motion][UP_y:DOWN_y][:,::20].copy()
            temp_img = data[i][UP_y:DOWN_y][:,::20].copy()
            # MANUAL
            past_shift = 0
            for _ in range(10):
                move = minz(method='powell',fun = mse_fun_tran_y,x0 = np.array([0.0]), bounds=[(-2,2)],
                            args = (stat
                                    ,temp_img
                                    ,past_shift))['x']
                past_shift += move[0]
            temp_tform_manual = AffineTransform(matrix = AffineTransform(translation=(0,past_shift*2)))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
        except:
            with open(f'debugs/debug{scan_num}.txt', 'a') as f:
                f.write(f'Y motion EVERYTHIN FAILED HERE\n')
                f.write(f'UP_y: {UP_y}, DOWN_y: {DOWN_y}\n')
                f.write(f'NAME: {scan_num}\n')
                f.write(f'Ith: {i}\n')
            temp_tform_manual = AffineTransform(translation=(0,0))
            transforms_all[i] = np.dot(transforms_all[i],temp_tform_manual)
    return transforms_all

def y_motion_correcting(data,UP_y,DOWN_y,top_surf,disable_tqdm,scan_num):
    static_y_motion = np.argmax(np.sum(data[:,UP_y:DOWN_y,:],axis=(1,2)))
    # finding the bright points in all images in standard interference
    nn = [np.argmax(np.sum(data[i][UP_y:DOWN_y],axis=1)) for i in range(data.shape[0])]
    tf_all_nn = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    if top_surf:
        for i in tqdm(range(data.shape[0]),desc='y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i,:DOWN_y]  = warp(data[i,:DOWN_y],AffineTransform(matrix=tf_all_nn[i]),order=3)
    else:
        for i in tqdm(range(data.shape[0]),desc='y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i,UP_y:]  = warp(data[i,UP_y:],AffineTransform(matrix=tf_all_nn[i]),order=3)
    tr_all_y = all_trans_y(data,UP_y,DOWN_y,static_y_motion,disable_tqdm,scan_num)
    if top_surf:
        for i in tqdm(range(data.shape[0]),desc='y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i,:DOWN_y]  = warp(data[i,:DOWN_y],AffineTransform(matrix=tr_all_y[i]),order=3)
    else:
        for i in tqdm(range(data.shape[0]),desc='y-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
            data[i,UP_y:]  = warp(data[i,UP_y:],AffineTransform(matrix=tr_all_y[i]),order=3)
    return data

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

def all_trans_x(data,UP_x,DOWN_x,valid_args,enface_extraction_rows,disable_tqdm,scan_num):
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
                    for _ in range(10):
                        move = minz(method='powell',fun = mse_fun_tran_x,x0 = np.array([0.0]), bounds=[(-4,4)],
                                    args = (stat
                                            ,temp_manual
                                            ,past_shift))['x']
                        past_shift += move[0]
                    cross_section = -(past_shift*2)
                else:
                    cross_section = 0
            except:
                with open(f'debugs/debug{scan_num}.txt', 'a') as f:
                    f.write(f'Cell corss_section failed here\n')
                    f.write(f'UP_x: {UP_x}, DOWN_x: {DOWN_x}\n')
                    f.write(f'NAME: {scan_num}\n')
                    f.write(f'Ith: {i}\n')
                    f.write(f'enface_extraction_rows: {enface_extraction_rows}\n')
                cross_section = 0
            enface_shape = data[:,0,:].shape[1]
            enface_wraps = []
            if len(enface_extraction_rows)>0:
                for enf_idx in range(len(enface_extraction_rows)):
                    try:
                        temp_enface_shift = get_line_shift(data[i,enface_extraction_rows[enf_idx]],data[i+1,enface_extraction_rows[enf_idx]],enface_shape)
                    except:
                        with open(f'debugs/debug{scan_num}.txt', 'a') as f:
                            f.write(f'TEMP enface shift failed here\n')
                            f.write(f'UP_x: {UP_x}, DOWN_x: {DOWN_x}\n')
                            f.write(f'NAME: {scan_num}\n')
                            f.write(f'Ith: {i}\n')
                            f.write(f'enface_extraction_rows: {enface_extraction_rows}\n')
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
            # raise e
            temp_tform_manual = AffineTransform(translation=(0,0))
            transforms_all[i+1] = np.dot(transforms_all[i+1],temp_tform_manual)
    return transforms_all

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
    uncroped_data = uncroped_data[:,np.r_[tuple(np.r_[start:end] for start, end in merged_coords)],:]
    return uncroped_data
