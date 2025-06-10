# from pydicom import dcmread
# import matplotlib.pylab as plt
import numpy as np
import os
from skimage.transform import warp, AffineTransform, pyramid_expand, pyramid_reduce
from natsort import natsorted
from tqdm import tqdm
import h5py
from ultralytics import YOLO
from reg_util_funcs import *
from util_funcs import *
import time
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, progress,performance_report
from dask import delayed, compute
import logging
import yaml


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

with open('datapaths.yaml', 'r') as f:
    config = yaml.safe_load(f)['server_data_paths_endo']

MODEL = YOLO(config['MODEL_PATH'])
SURFACE_Y_PAD = 20
SURFACE_X_PAD = 10
CELLS_X_PAD = 5
DATA_LOAD_DIR = config['DATA_LOAD_DIR']
DATA_SAVE_DIR = config['DATA_SAVE_DIR']

# MODEL = YOLO(config['local_data_paths']['MODEL_PATH'])
# SURFACE_Y_PAD = 20
# SURFACE_X_PAD = 10
# CELLS_X_PAD = 5
# DATA_LOAD_DIR = config['local_data_paths']['DATA_LOAD_DIR']
# DATA_SAVE_DIR = config['local_data_paths']['DATA_SAVE_DIR']

def main(args):
    global MODEL
    dirname, scan_num,disable_tqdm,save_detections = args
    if not os.path.exists(dirname):
        raise FileNotFoundError(f"Directory {dirname} not found")
    if not os.path.exists(os.path.join(dirname, scan_num)):
        raise FileNotFoundError(f"Scan {scan_num} not found in {dirname}")
    original_data = load_h5_data(dirname,scan_num)
    # MODEL PART
    # pbar.set_description(desc = f'Loading Model for {scan_num}')
    static_flat = np.argmax(np.sum(original_data[:,:,:],axis=(0,1)))
    test_detect_img = preprocess_img(original_data[:,:,static_flat])
    res_surface = MODEL.predict(test_detect_img,iou = 0.5, save = save_detections, project = 'Detected Areas',name = scan_num, verbose=False,classes=[0,1], device='cpu', augment = True)
    surface_crop_coords = [i for i in res_surface[0].summary() if i['name']=='surface']
    cells_crop_coords = [i for i in res_surface[0].summary() if i['name']=='cells']
    surface_crop_coords = detect_areas(surface_crop_coords, pad_val = 20, img_shape = test_detect_img.shape[0])
    cells_crop_coords = detect_areas(cells_crop_coords, pad_val = 20, img_shape = test_detect_img.shape[0])
    cropped_original_data = crop_data(original_data,surface_crop_coords,cells_crop_coords,original_data.shape[1])
    del original_data

    static_flat = np.argmax(np.sum(cropped_original_data[:,:,:],axis=(0,1)))
    test_detect_img = preprocess_img(cropped_original_data[:,:,static_flat])
    res_surface = MODEL.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes=0, device='cpu')
    # result_list = res[0].summary()
    surface_coords = detect_areas(res_surface[0].summary(),pad_val = SURFACE_Y_PAD, img_shape = test_detect_img.shape[0])
    if surface_coords is None:
        with open(f'debug{scan_num}.txt', 'a') as f:
            f.write(f'NO SURFACE DETECTED: {scan_num}\n')
        # print(f'NO SURFACE DETECTED: {scan_num}')
        return None
    
    # FLATTENING PART
    # pbar.set_description(desc = f'Flattening {scan_num}.....')
    # print('SURFACE COORDS:',surface_coords)
    static_flat = np.argmax(np.sum(cropped_original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(0,1)))
    top_surf = True
    for i in range(surface_coords.shape[0]):
        UP_flat,DOWN_flat = surface_coords[i,0], surface_coords[i,1]
        UP_flat = max(UP_flat,0)
        DOWN_flat = min(DOWN_flat, cropped_original_data.shape[2])
        cropped_original_data = flatten_data(cropped_original_data,UP_flat,DOWN_flat,top_surf,disable_tqdm)
        top_surf = False

    # Y-MOTION PART
    # pbar.set_description(desc = f'Correcting {scan_num} Y-Motion.....')
    top_surf = True
    for i in range(surface_coords.shape[0]):
        UP_y,DOWN_y = surface_coords[i,0], surface_coords[i,1]
        UP_y = max(UP_y,0)
        DOWN_y = min(DOWN_y, cropped_original_data.shape[2])
        cropped_original_data = y_motion_correcting(cropped_original_data,UP_y,DOWN_y,top_surf,disable_tqdm)
        top_surf = False

    # X-MOTION PART
    # pbar.set_description(desc = f'Correcting {scan_num} X-Motion.....')
    test_detect_img = preprocess_img(cropped_original_data[:,:,static_flat])
    res_surface = MODEL.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes = 0, device='cpu')
    res_cells = MODEL.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes = 1, device='cpu')
    # result_list = res[0].summary()
    surface_coords = detect_areas(res_surface[0].summary(),pad_val = SURFACE_X_PAD, img_shape = test_detect_img.shape[0])
    cells_coords = detect_areas(res_cells[0].summary(),pad_val = CELLS_X_PAD, img_shape = test_detect_img.shape[0])

    if (cells_coords is None) and (surface_coords is None):
        with open(f'debug{scan_num}.txt', 'a') as f:
            f.write(f'NO SURFACE OR CELLS DETECTED: {scan_num}\n')
            # f.write(f'UP_x: {UP_x}, DOWN_x: {DOWN_x}\n')
            # f.write(f'NAME: {scan_num}\n')
            # f.write(f'Ith: {i}\n')
            # f.write(f'enface_extraction_rows: {enface_extraction_rows}\n')
        # print()
        return
    
    enface_extraction_rows = []
    if surface_coords is not None:
        # print('SURFACE COORDS:',surface_coords)
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
    
    tr_all = ants_all_trans_x(cropped_original_data,UP_x,DOWN_x,valid_args,enface_extraction_rows,disable_tqdm,scan_num)
    for i in tqdm(range(1,cropped_original_data.shape[0],2),desc='warping',disable=disable_tqdm):
        cropped_original_data[i]  = warp(cropped_original_data[i],AffineTransform(matrix=tr_all[i]),order=3)

    # pbar.set_description(desc = 'Saving Data.....')
    if cropped_original_data.dtype != np.float32:
        cropped_original_data = cropped_original_data.astype(np.float32)
    folder_save = DATA_SAVE_DIR
    os.makedirs(folder_save,exist_ok=True)
    hdf5_filename = f'{folder_save}{scan_num}.h5'
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=cropped_original_data, compression='gzip',compression_opts=5)

def init_worker():
    global MODEL
    MODEL = YOLO(config['MODEL_PATH'])
    # MODEL = YOLO('/Users/akapatil/Documents/feature_extraction/yolo_feature_extraction/yolov12s_best.pt')

if __name__ == "__main__":
    data_dirname = DATA_LOAD_DIR
    if os.path.exists(DATA_SAVE_DIR):
        done_scans = set([i for i in os.listdir(DATA_SAVE_DIR) if (i.startswith('scan'))])
        print(done_scans)
    else:
        done_scans={}
    scans = [i for i in os.listdir(data_dirname) if (i.startswith('scan')) and (i+'.h5' not in done_scans)]
    scans = natsorted(scans)
    print('REMAINING',scans)
    args_list = [(data_dirname, sc, True, True) for sc in scans]
    # pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█")
    # num_processes = min(len(scans), cpu_count() - 1)  # Leave one CPU free
    logger.debug("LOGERR WORKING")
    print("Setting up Dask SLURM cluster...")
    cluster = SLURMCluster(
                queue='general',
                account='r00970',
                cores=1, 
                processes=1,
                memory='7GB',
                walltime='01:00:00',
                job_extra_directives=[
                    "--cpus-per-task=1",
                    "--nodes=1",
                    "--job-name=my_job",
                    "--output=my_job.out",
                    "--error=my_job.err"
                ],
                python='/N/project/OCT_preproc/registration_with_ODalgo/OCT_reg/bin/python',
            )
    cluster.scale(jobs=len(scans))
    # Attach client
    client = Client(cluster)
    print(client)
    client.run(init_worker)
    tasks = [delayed(main)(args) for args in args_list]
    print(client.dashboard_link)
    print("Submitting tasks to the cluster...")
    start = time.time()
    with performance_report(filename="dask-report.html"):
        results = compute(*tasks)
        progress(results) 
        results = client.gather(results)
    # results = compute(*tasks)
    print(f"Done in {time.time() - start:.2f} seconds")
    client.close()
    cluster.close()

