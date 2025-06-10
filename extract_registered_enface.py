import numpy as np
import os
from skimage.transform import warp, AffineTransform
from natsort import natsorted
from tqdm import tqdm
import h5py
from ultralytics import YOLO
from reg_util_funcs import *
from util_funcs import *
import time
# from dask_jobqueue import SLURMCluster
# from dask.distributed import Client, progress,performance_report
from dask import delayed, compute
import logging
import yaml

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DATA_KEY = 'batch_epi'
with open('datapaths.yaml', 'r') as f:
    config = yaml.safe_load(f)[f'server_enface_paths_epi']

MODEL = YOLO(config['MODEL_PATH'])
# SURFACE_Y_PAD = 20
# SURFACE_X_PAD = 10
# CELLS_X_PAD = 5
DATA_LOAD_DIR = config['DATA_LOAD_DIR']
DATA_SAVE_DIR = config['DATA_SAVE_DIR']
EXPECTED_SURFACES = 2
# EXPECTED_CELLS = 1

def return_enface(data, coord):
    return data[:,coord,:]

def extract_enfaces(dirname):
    path = f'{dirname}/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.h5'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    standard_enfaces = []
    self_enfaces = []
    for i in range(len(pic_paths)):
        with h5py.File(path+pic_paths[i], 'r') as hf:
            original_data = hf['volume'][:].astype(np.float32)
            static_flat = np.argmax(np.sum(original_data[:,:,:],axis=(0,1)))
            test_detect_img = preprocess_img(original_data[:,:,static_flat])
            res_surface = MODEL.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes=0, device='cpu',agnostic_nms = True, augment = True)
            surface_coords = detect_areas(res_surface[0].summary(),pad_val = 1, img_shape = test_detect_img.shape[0], expected_num = EXPECTED_SURFACES)
            if surface_coords is None:
                with open(f'debugs/debug{pic_paths[i]}.txt', 'a') as f:
                    f.write(f'NO SURFACE DETECTED: {pic_paths[i]}\n')
                return None
            enface_extraction_rows = []
            for i in range(surface_coords.shape[0]):
                val = np.argmax(np.sum(np.max(original_data[:,surface_coords[i,0]:surface_coords[i,1],:],axis=0),axis=1))
                enface_extraction_rows.append(int(surface_coords[i,0]+val))
            if len(enface_extraction_rows)==1:
                standard_enfaces.append(return_enface(original_data, enface_extraction_rows[0]))
            else:
                standard_enfaces.append(return_enface(original_data, enface_extraction_rows[0]))
                self_enfaces.append(return_enface(original_data, enface_extraction_rows[1]))
    return standard_enfaces, self_enfaces

def main(args):
    dirname, scan_num,disable_tqdm,save_detections = args
    if not os.path.exists(dirname):
        raise FileNotFoundError(f"Directory {dirname} not found")
    standard_enfaces, self_enfaces = extract_enfaces(dirname)
    if len(standard_enfaces)>0:
        standard_enfaces = np.array(standard_enfaces)
        folder_save = DATA_SAVE_DIR
        save_file_name = DATA_SAVE_DIR+f'standard_enfaces_{DATA_KEY}'
        os.makedirs(folder_save,exist_ok=True)
        hdf5_filename = f'{save_file_name}.h5'
        with h5py.File(hdf5_filename, 'w') as hf:
            hf.create_dataset('volume', data=standard_enfaces, compression='gzip',compression_opts=5)
    
    if len(self_enfaces)>0:
        self_enfaces = np.array(self_enfaces)
        folder_save = DATA_SAVE_DIR
        save_file_name = DATA_SAVE_DIR+f'self_enfaces_{DATA_KEY}'
        os.makedirs(folder_save,exist_ok=True)
        hdf5_filename = f'{save_file_name}.h5'
        with h5py.File(hdf5_filename, 'w') as hf:
            hf.create_dataset('volume', data=self_enfaces, compression='gzip',compression_opts=5)
    
    return
    
if __name__ == "__main__":
    arg_list = [DATA_LOAD_DIR, None, None, None]
    main(arg_list)
