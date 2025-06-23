from pydicom import dcmread
# import matplotlib.pylab as plt
import numpy as np
import os
from skimage.transform import warp, AffineTransform
import cv2
import h5py
from natsort import natsorted
from scipy.fftpack import fft2, fftshift, ifft2, fft, ifft
from skimage.filters import threshold_otsu
from skimage.metrics import normalized_mutual_information as nmi
from tifffile import imread as tiffread
from scipy.signal import correlate2d


# def load_data(path_num,path_all = False):
#     if path_all:
#         path = path_all
#     else:
#         path = f'registered/{path_num}/'
#     pic_paths = []
#     for i in os.listdir(path):
#         if i.endswith('.dcm') or  i.endswith('.tiff') or i.endswith('.PNG'):
#             pic_paths.append(i)
#     pic_paths = natsorted(pic_paths)

#     temp_img = cv2.imread(path+pic_paths[0],cv2.IMREAD_UNCHANGED) 
#     imgs_from_folder = np.zeros((len(pic_paths),temp_img.shape[0],temp_img.shape[1]))
#     # imgs_from_folder = []
#     for i,j in enumerate(pic_paths):
#         aa = cv2.imread(path+j,cv2.IMREAD_UNCHANGED)
#         imgs_from_folder[i] = aa.copy()
#     imgs_from_folder = imgs_from_folder.astype(np.float32)
#     return imgs_from_folder

def ncc1d(array1, array2):
    correlation = np.correlate(array1, array2, mode='valid')
    array1_norm = np.linalg.norm(array1)
    array2_norm = np.linalg.norm(array2)
    if array1_norm == 0 or array2_norm == 0:
        return np.zeros_like(correlation)
    normalized_correlation = correlation / (array1_norm * array2_norm)
    return normalized_correlation

def ncc(array1, array2):
    correlation = correlate2d(array1, array2, mode='valid')
    array1_norm = np.linalg.norm(array1)
    array2_norm = np.linalg.norm(array2)
    if array1_norm == 0 or array2_norm == 0:
        return np.zeros_like(correlation)
    normalized_correlation = correlation / (array1_norm * array2_norm)
    return normalized_correlation

def min_max(data1, global_min=None, global_max=None):    
    min_val = np.min(data1) if global_min is None else global_min
    max_val = np.max(data1) if global_max is None else global_max
    if min_val == max_val:
        return data1 
    return (data1 - min_val) / (max_val - min_val)

'''
# def mse_fun_tran(shif, x, y , past_shift):
#     x = warp(x, AffineTransform(translation=(0,-past_shift)),order=3)
#     y = warp(y, AffineTransform(translation=(0,past_shift)),order=3)

#     warped_x_stat = warp(x, AffineTransform(translation=(0,-shif[0])),order=3)
#     warped_y_mov = warp(y, AffineTransform(translation=(0,shif[0])),order=3)

#     return (1-ncc(warped_x_stat ,warped_y_mov))


# def ants_all_trans(data,UP,DOWN):
#     transforms_all = np.tile(np.eye(3),(data.shape[0],1,1))
#     for i in tqdm(range(data.shape[0]-1),desc='tr_all'):
#         temp_img = data[i+1][UP:DOWN][:,-50:].copy()
#         stat = data[i][UP:DOWN][:,-50:].copy()
#         # PHASE
#         coords = phase_cross_correlation((stat)
#                                         ,(temp_img)
#                                         ,normalization=None,upsample_factor=20)[0]
#         if np.abs(coords[0])<=5:
#             temp_img = warp(temp_img,AffineTransform(translation = (0,-coords[0])),order=3)
#             tff = AffineTransform(translation = (0,-coords[0]))
#             transforms_all[i+1:] = np.dot(transforms_all[i+1:],tff)

#         # MANUAL
#         temp_tform_manual = AffineTransform(translation=(0,0))
#         temp_manual = temp_img.copy()
#         past_shift = 0
#         for _ in range(5):
#             move = minz(method='powell',fun = mse_fun_tran,x0 =(0), bounds=[(-5,5)],
#                         args = (stat
#                                 ,temp_manual
#                                 ,past_shift))['x']
#             temp_transform = AffineTransform(translation=(0,move[0]))
#             past_shift += move[0]
#             # temp_manual = warp(temp_manual, temp_transform,order=3)
#             temp_tform_manual = np.dot(temp_tform_manual,temp_transform)
#         temp_tform_manual = AffineTransform(matrix = temp_tform_manual)
#         # if np.abs(np.array(temp_tform_manual)[1,2])<=2:
#         #     temp_img = warp(temp_img,temp_tform_manual,order=3)
#         transforms_all[i+1:] = np.dot(transforms_all[i+1:],temp_tform_manual)

#     return transforms_all

'''

def bottom_extract(data,mid):
    test = np.max(data.transpose(2,1,0),axis=0).copy()
    kk = fftshift(fft2(test[-(data[0].shape[0]-mid):-80]))
    filt = np.ones_like(kk)
    filt[(filt.shape[0]//2)-5:(filt.shape[0]//2)+5,(filt.shape[1]//2)-5:(filt.shape[1]//2)+5] = 0
    kk = kk*filt
    kk = np.abs(ifft2(fftshift(kk)))
    max_list = np.max(kk,axis=1)
    thresh = threshold_otsu(max_list)
    mir_UP_x, mir_DOWN_x = np.where(max_list>=thresh)[0][0]+mid, np.where(max_list>=thresh)[0][-1]+mid
    UP_x,DOWN_x = ((2*mid - mir_UP_x)-(mir_DOWN_x - mir_UP_x)), (2*mid - mir_UP_x)
    return UP_x,DOWN_x,mir_UP_x,mir_DOWN_x

def top_extract(data,mid):
    test = np.max(data.transpose(2,1,0),axis=0).copy()
    bright_point = np.argmax(np.sum(test[:mid],axis=1))+30
    kk = fftshift(fft2(test[bright_point:mid]))
    filt = np.ones_like(kk)
    filt[(filt.shape[0]//2)-5:(filt.shape[0]//2)+5,(filt.shape[1]//2)-5:(filt.shape[1]//2)+5] = 0
    kk = kk*filt
    kk = np.abs(ifft2(fftshift(kk)))
    max_list = np.max(kk,axis=1)
    thresh = threshold_otsu(max_list)
    UP_x, DOWN_x = np.where(max_list>=thresh)[0][0]+bright_point, np.where(max_list>=thresh)[0][-1]+bright_point
    mir_UP_x, mir_DOWN_x = 2*mid-(np.where(max_list>=thresh)[0][-1]+bright_point), 2*mid-(np.where(max_list>=thresh)[0][0]+bright_point)
    return UP_x,DOWN_x,mir_UP_x,mir_DOWN_x

def denoise_fft(data):
    kk = fftshift(fft2(data))
    filt = np.ones_like(kk)
    filt[(filt.shape[0]//2)-5:(filt.shape[0]//2)+5,(filt.shape[1]//2)-5:(filt.shape[1]//2)+5] = 0
    kk = kk*filt
    kk = np.abs(ifft2(fftshift(kk)))
    return kk

def find_mid(data):
    n = data.shape[1]
    mid = (np.argmax(np.sum(data[0][:n//2],axis=1)) + data[0].shape[0])//2
    return mid

def denoise_signal(errs , rows = 10):
    kk = fft(errs)
    kk[rows:] = 0
    kk = abs(ifft(kk))
    return kk

def non_zero_crop(a,b):
    mini = max(np.min(np.where(a[0]!=0)),np.min(np.where(b[0]!=0)))
    maxi = min(np.max(np.where(a[0]!=0)),np.max(np.where(b[0]!=0)))
    return mini, maxi

def denoise_signal1D_err_calc(errs , rows = 20):
    kk = fft(errs)
    kk[rows:] = 0
    kk = abs(ifft(kk))
    return kk

def preprocess_img(data):
    data = data.transpose(1,0)
    data = min_max(data)
    data = (data*255).astype(np.uint8)
    data = np.dstack([[data]*3]).transpose(1,2,0)
    data = np.ascontiguousarray(data)
    return data

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # overlap
            last[1] = max(last[1], current[1])  # merge
        else:
            merged.append(current)
    return merged


def load_h5_data(dirname, scan_num):
    path = f'{dirname}/{scan_num}/'
    # path = 'intervolume_registered/self_inter/scan5/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.h5'):
            pic_paths.append(i)
    with h5py.File(path+pic_paths[0], 'r') as hf:
        original_data = hf['volume'][:,100:-100,:].astype(np.float32)
    return original_data

def load_data_dcm(dirname, scan_num):
    path = f'{dirname}/{scan_num}/'
    # path = path_num
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    temp_img = dcmread(path+pic_paths[0]).pixel_array
    imgs_from_folder = np.zeros((len(pic_paths),*temp_img.shape))
    for i,j in enumerate(pic_paths):
        aa = dcmread(path+j)
        imgs_from_folder[i] = aa.pixel_array
    imgs_from_folder = imgs_from_folder[:,600:-100,:].astype(np.float32)
    return imgs_from_folder

def GUI_load_dcm(path_dir):
    # path = path_num
    if not path_dir.endswith('/'):
        path_dir = path_dir+'/'
    pic_paths = []
    for i in os.listdir(path_dir):
        if i.endswith('.dcm') or  i.endswith('.DCM'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    temp_img = dcmread(path_dir+pic_paths[0]).pixel_array
    imgs_from_folder = np.zeros((len(pic_paths),*temp_img.shape))
    for i,j in enumerate(pic_paths):
        aa = dcmread(path_dir+j)
        imgs_from_folder[i] = aa.pixel_array
    imgs_from_folder = imgs_from_folder[:,:,:].astype(np.float32)
    return imgs_from_folder

def GUI_load_h5(path_h5):
    if not path_h5.endswith('.h5'):
        raise Exception ("Not HDF5 data format")
    with h5py.File(path_h5, 'r') as hf:
        original_data = hf['volume'][:,:,:].astype(np.float32)
    return original_data
