
import sys
import matplotlib.pylab as plt
import numpy as np
import os
from skimage.transform import warp, AffineTransform, pyramid_expand, pyramid_reduce
from natsort import natsorted
from tqdm import tqdm
from tqdm.utils import envwrap
import h5py
from ultralytics import YOLO
from utils.reg_util_funcs import *
from utils.util_funcs import *
import yaml
import torch

with open('datapaths.yaml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_FEATURE_DETECT = YOLO(config['PATHS']['MODEL_FEATURE_DETECT_PATH'])
USE_MODEL_X = config['PATHS']['USE_MODEL_X']
MODEL_X_TRANSLATION_PATH = config['PATHS']['MODEL_X_TRANSLATION_PATH']
SURFACE_Y_PAD = 20
SURFACE_X_PAD = 10
CELLS_X_PAD = 5
DATA_LOAD_DIR = config['PATHS']['DATA_LOAD_DIR']
DATA_SAVE_DIR = config['PATHS']['DATA_SAVE_DIR']
EXPECTED_SURFACES = config['PATHS']['EXPECTED_SURFACES']
EXPECTED_CELLS = config['PATHS']['EXPECTED_CELLS']
BATCH_FLAG = config['PATHS']['BATCH_FLAG']
DISABLE_TQDM = config['PATHS']['DISABLE_TQDM']
ENABLE_MULTIPROC_SLURM = config['PATHS']['ENABLE_MULTIPROC_SLURM']

if not ENABLE_MULTIPROC_SLURM:
    if USE_MODEL_X:
        try:
            DEVICE = 'cpu'
            MODEL_X_TRANSLATION = torch.load(MODEL_X_TRANSLATION_PATH, map_location=DEVICE, weights_only=False)
            MODEL_X_TRANSLATION.eval()
            print("Model X loaded successfully.")
        except Exception as e:
            print(f"Error loading Model X: {e}")
            print("Proceeding without Model X translation.")
            MODEL_X_TRANSLATION = None
    else:
        MODEL_X_TRANSLATION = None

def main(dirname, scan_num, pbar, data_type, disable_tqdm, save_detections, ):
    global MODEL_FEATURE_DETECT
    global MODEL_X_TRANSLATION
    if data_type=='h5':
        if not os.path.exists(dirname):
            raise FileNotFoundError(f"Scan {dirname} not found")
        original_data = load_h5_data(dirname,scan_num)
    elif data_type=='dcm':
        if not os.path.exists(dirname):
            raise FileNotFoundError(f"Directory {dirname} not found")
        original_data = load_data_dcm(dirname,scan_num)
    # MODEL_FEATURE_DETECT PART
    print(original_data.shape)

    pbar.set_description(desc = f'Loading Model_FEATURE_DETECT for {scan_num}')
    static_flat = np.argmax(np.sum(original_data[:,:,:],axis=(0,1)))
    test_detect_img = preprocess_img(original_data[:,:,static_flat])
    res_surface = MODEL_FEATURE_DETECT.predict(test_detect_img,iou = 0.5, save = save_detections, project = 'Detected Areas',name = scan_num, verbose=False,classes=[0,1], device='cpu',agnostic_nms = True, augment = True)
    surface_crop_coords = [i for i in res_surface[0].summary() if i['name']=='surface']
    cells_crop_coords = [i for i in res_surface[0].summary() if i['name']=='cells']
    surface_crop_coords = detect_areas(surface_crop_coords, pad_val = 20, img_shape = test_detect_img.shape[0], expected_num = EXPECTED_SURFACES)
    cells_crop_coords = detect_areas(cells_crop_coords, pad_val = 20, img_shape = test_detect_img.shape[0], expected_num = EXPECTED_CELLS)
    if surface_crop_coords is None:
        print(f'NO SURFACE DETECTED: {scan_num}')
        return None
    cropped_original_data = crop_data(original_data,surface_crop_coords,cells_crop_coords,original_data.shape[1])
    del original_data

    static_flat = np.argmax(np.sum(cropped_original_data[:,:,:],axis=(0,1)))
    test_detect_img = preprocess_img(cropped_original_data[:,:,static_flat])
    res_surface = MODEL_FEATURE_DETECT.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes=0, device='cpu',agnostic_nms = True, augment = True)
    # result_list = res[0].summary()
    surface_coords = detect_areas(res_surface[0].summary(),pad_val = SURFACE_Y_PAD, img_shape = test_detect_img.shape[0], expected_num = EXPECTED_SURFACES)
    if surface_coords is None:
        print(f'NO SURFACE DETECTED: {scan_num}')
        return None
    if EXPECTED_SURFACES>1:
        partition_coord = np.ceil(np.mean(np.mean(surface_coords[-2:],axis=1))).astype(int)
    else:
        partition_coord = None

    # FLATTENING PART
    pbar.set_description(desc = f'Flattening {scan_num}.....')
    # print('SURFACE COORDS:',surface_coords)
    static_flat = np.argmax(np.sum(cropped_original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(0,1)))
    top_surf = True
    if surface_coords.shape[0]>1:
        for _ in range(2):
            if top_surf:
                cropped_original_data = flatten_data(cropped_original_data,surface_coords[:-1],top_surf,partition_coord,disable_tqdm,scan_num)
            else:
                cropped_original_data = flatten_data(cropped_original_data,surface_coords[-1:],top_surf,partition_coord,disable_tqdm,scan_num)
            top_surf = False
    else:
        cropped_original_data = flatten_data(cropped_original_data,surface_coords,top_surf,partition_coord,disable_tqdm,scan_num)

    # Y-MOTION PART
    pbar.set_description(desc = f'Correcting {scan_num} Y-Motion.....')
    top_surf = True
    if surface_coords.shape[0]>1:
        for _ in range(2):
            if top_surf:
                cropped_original_data = y_motion_correcting(cropped_original_data,surface_coords[:-1],top_surf,partition_coord,disable_tqdm,scan_num)
            else:
                cropped_original_data = y_motion_correcting(cropped_original_data,surface_coords[-1:],top_surf,partition_coord,disable_tqdm,scan_num)
            top_surf = False
    else:
        cropped_original_data = y_motion_correcting(cropped_original_data,surface_coords,top_surf,partition_coord,disable_tqdm,scan_num)

    # X-MOTION PART
    pbar.set_description(desc = f'Correcting {scan_num} X-Motion.....')
    test_detect_img = preprocess_img(cropped_original_data[:,:,static_flat])
    res_surface = MODEL_FEATURE_DETECT.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes = 0, device='cpu',agnostic_nms = True, augment = True)
    res_cells = MODEL_FEATURE_DETECT.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes = 1, device='cpu',agnostic_nms = True, augment = True)
    # result_list = res[0].summary()
    surface_coords = detect_areas(res_surface[0].summary(),pad_val = SURFACE_X_PAD, img_shape = test_detect_img.shape[0], expected_num = EXPECTED_SURFACES)
    cells_coords = detect_areas(res_cells[0].summary(),pad_val = CELLS_X_PAD, img_shape = test_detect_img.shape[0], expected_num = EXPECTED_CELLS)

    if (cells_coords is None) and (surface_coords is None):
        print(f'NO SURFACE OR CELLS DETECTED: {scan_num}')
        return None
    
    enface_extraction_rows = []
    if surface_coords is not None:
        static_y_motion = np.argmax(np.sum(cropped_original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(1,2)))    
        errs = []
        for i in range(cropped_original_data.shape[0]):
            errs.append(ncc(cropped_original_data[static_y_motion,:,:],cropped_original_data[i,:,:])[0])
        errs = np.squeeze(errs)
        valid_args = np.squeeze(np.argwhere(errs>0.7))
        for i in range(surface_coords.shape[0]):
            val = np.argmax(np.sum(np.max(cropped_original_data[:,surface_coords[i,0]:surface_coords[i,1],:],axis=0),axis=1))
            enface_extraction_rows.append(surface_coords[i,0]+val)
    else:
        valid_args = np.arange(cropped_original_data.shape[0])

    if cells_coords is not None:
        if cells_coords.shape[0]==1:
            UP_x, DOWN_x = (cells_coords[0,0]), (cells_coords[0,1])
        else:
            UP_x, DOWN_x = (cells_coords[:,0]), (cells_coords[:,1])
    else:
        UP_x, DOWN_x = None,None

    # print('UP_x:',UP_x)
    # print('DOWN_x:',DOWN_x)
    # # print('VALID ARGS: ',valid_args)
    # print('ENFACE EXTRACTION ROWS: ',enface_extraction_rows)
    tr_all = all_trans_x(cropped_original_data,UP_x,DOWN_x,valid_args,enface_extraction_rows
                         ,disable_tqdm,scan_num, MODEL_X_TRANSLATION)
    for i in tqdm(range(1,cropped_original_data.shape[0],2),desc='X-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        cropped_original_data[i]  = warp(cropped_original_data[i],AffineTransform(matrix=tr_all[i]),order=3)

    pbar.set_description(desc = 'Saving Data.....')
    if cropped_original_data.dtype != np.float64:
        cropped_original_data = cropped_original_data.astype(np.float64)
    folder_save = DATA_SAVE_DIR
    if not folder_save.endswith('/'):
        folder_save = folder_save + '/'
    os.makedirs(folder_save,exist_ok=True)
    hdf5_filename = f'{folder_save}{scan_num}.h5'
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=cropped_original_data, compression='gzip',compression_opts=5)

def init_worker():
    global MODEL_FEATURE_DETECT
    global MODEL_X_TRANSLATION
    
    with open('datapaths.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    MODEL_FEATURE_DETECT = YOLO(config['PATHS']['MODEL_FEATURE_DETECT_PATH'])
    
    if config['PATHS']['USE_MODEL_X']:
        try:
            DEVICE = 'cpu'
            MODEL_X_TRANSLATION = torch.load(config['PATHS']['MODEL_X_TRANSLATION_PATH'], 
                                           map_location=DEVICE, weights_only=False)
            MODEL_X_TRANSLATION.eval()
            print("Model X loaded successfully on worker")
        except Exception as e:
            print(f"Error loading Model X on worker: {e}")
            print("Proceeding without Model X translation on worker")
            MODEL_X_TRANSLATION = None
    else:
        MODEL_X_TRANSLATION = None

if __name__ == "__main__":
    data_dirname = DATA_LOAD_DIR
    if data_dirname.endswith('/'):
        data_dirname = data_dirname[:-1]
    if not BATCH_FLAG:
        if data_dirname.lower().endswith('.h5'):
            data_type = 'h5'
            scans = [data_dirname.split('/')[-1].removesuffix('.h5')]
        else:
            data_type = 'dcm'
            scans = [data_dirname.split('/')[-1]]
    else:
        scans = [i for i in os.listdir(data_dirname) if (i.startswith('scan'))]
        scans = natsorted(scans)
        data_type = 'h5'

    pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█")
    if not ENABLE_MULTIPROC_SLURM:
        disable_tqdm = DISABLE_TQDM
        save_detections = False
        for scan_num in pbar:
            pbar.set_description(desc = f'Processing {scan_num}')
            main(data_dirname, scan_num, pbar, data_type, disable_tqdm, save_detections)

    elif ENABLE_MULTIPROC_SLURM:
        try:
            from dask_jobqueue import SLURMCluster
            from dask.distributed import Client, progress
            from dask import delayed, compute
        except:
            raise ImportError("Dask and dask_jobqueue modules are required for multiprocessing with SLURM") from e

        disable_tqdm = True # Has to be True for multiprocessing
        save_detections = False
        multiproc_args_list = [(data_dirname, scan_num, pbar, data_type, disable_tqdm, save_detections) for scan_num in scans]
        print("Setting up Dask SLURM cluster...")
        cluster = SLURMCluster(
            queue='general',
            account='r00970',
            cores=1, 
            processes=1,
            memory='15GB',
            walltime='01:00:00',
            job_extra_directives=[
                "--cpus-per-task=1",
                "--nodes=1",
                "--job-name=my_job",
                "--output=my_job.out",
                "--error=my_job.err"
            ],
            python=sys.executable,
        )
        cluster.scale(jobs=len(scans))
        # Attach client
        client = Client(cluster)
        print(client)
        client.run(init_worker)
        tasks = [delayed(main)(*args) for args in multiproc_args_list]
        print(client.dashboard_link)
        print("Submitting tasks to the cluster...")
        results = compute(*tasks)
        results = client.gather(results)
        client.close()
        cluster.close()
