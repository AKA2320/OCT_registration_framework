# import matplotlib.pylab as plt
import numpy as np
import os
from skimage.transform import warp, AffineTransform
# from natsort import natsorted
from tqdm import tqdm
# from tqdm.utils import envwrap
import h5py
from ultralytics import YOLO
from utils.reg_util_funcs import *
from utils.util_funcs import *
import yaml
import torch
import sys
# import click

try:
    with open('datapaths.yaml', 'r') as f:
        config = yaml.safe_load(f)
except:
    with open(f'{os.path.join(os.path.dirname(sys.executable), '_internal')}/datapaths.yaml', 'r') as f:
        config = yaml.safe_load(f)
        

MODEL_FEATURE_DETECT = YOLO(config['PATHS']['MODEL_FEATURE_DETECT_PATH'])
# USE_MODEL_X = config['PATHS']['USE_MODEL_X'] # This will now come from command line
MODEL_X_TRANSLATION_PATH = config['PATHS']['MODEL_X_TRANSLATION_PATH']
SURFACE_Y_PAD = 20
SURFACE_X_PAD = 10
CELLS_X_PAD = 5
# DATA_LOAD_DIR = config['PATHS']['DATA_LOAD_DIR'] # This will now come from command line
# DATA_SAVE_DIR = config['PATHS']['DATA_SAVE_DIR'] # Keep this for default if no save_dirname is provided
# EXPECTED_SURFACES = config['PATHS']['EXPECTED_SURFACES']
# EXPECTED_CELLS = config['PATHS']['EXPECTED_CELLS']


def main(dirname, scan_num, pbar, data_type, disable_tqdm, save_detections, use_model_x, save_dirname, expected_cells, expected_surfaces, cancellation_flag=None):
    global MODEL_FEATURE_DETECT
    global MODEL_X_TRANSLATION
    global EXPECTED_SURFACES
    global EXPECTED_CELLS
    print(f"Expected Cells: {expected_cells}")
    print(f"Expected Surfaces: {expected_surfaces}")
    EXPECTED_SURFACES = int(expected_surfaces)
    EXPECTED_CELLS = int(expected_cells)
    MODEL_X_TRANSLATION = None # Initialize to None
    if use_model_x:
        try:
            DEVICE = 'cpu' # Assuming CPU for the model based on original code
            MODEL_X_TRANSLATION = torch.load(MODEL_X_TRANSLATION_PATH, map_location=DEVICE, weights_only=False)
            MODEL_X_TRANSLATION.eval()
            print("Model X loaded successfully.")
        except Exception as e:
            print(f"Error loading Model X: {e}")
            print("Proceeding without Model X translation.")
            MODEL_X_TRANSLATION = None

    if data_type=='h5':
        original_data = load_h5_data(dirname,scan_num)
    elif data_type=='dcm':
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
        # with open(f'debugs/debug{scan_num}.txt', 'a') as f:
        #     f.write(f'NO SURFACE DETECTED: {scan_num}\n')
        #     f.write(f'min range: {cropped_original_data.min(),cropped_original_data.max()}\n')
        print(f'NO SURFACE DETECTED: {scan_num}')
        return None
    if EXPECTED_SURFACES>1:
        partition_coord = np.ceil(np.mean(np.mean(surface_coords[-2:],axis=1))).astype(int)
    else:
        partition_coord = None

    # FLATTENING PART
    print('Starting Flattening')
    pbar.set_description(desc = f'Flattening {scan_num}.....')
    static_flat = np.argmax(np.sum(cropped_original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(0,1)))
    top_surf = True
    if surface_coords.shape[0]>1:
        for _ in range(2):
            if cancellation_flag and cancellation_flag():
                print("Registration cancelled during flattening")
                return None
            if top_surf:
                cropped_original_data = flatten_data(cropped_original_data,surface_coords[:-1],top_surf,partition_coord,disable_tqdm,scan_num)
            else:
                cropped_original_data = flatten_data(cropped_original_data,surface_coords[-1:],top_surf,partition_coord,disable_tqdm,scan_num)
            top_surf = False
    else:
        if cancellation_flag and cancellation_flag():
            print("Registration cancelled during flattening")
            return None
        cropped_original_data = flatten_data(cropped_original_data,surface_coords,top_surf,partition_coord,disable_tqdm,scan_num)

    # Y-MOTION PART
    if cancellation_flag and cancellation_flag():
        print("Registration cancelled before Y-motion correction")
        return None
        
    print('Starting Y-motion')
    pbar.set_description(desc = f'Correcting {scan_num} Y-Motion.....')
    top_surf = True
    if surface_coords.shape[0]>1:
        for _ in range(2):
            if cancellation_flag and cancellation_flag():
                print("Registration cancelled during Y-motion correction")
                return None
            if top_surf:
                cropped_original_data = y_motion_correcting(cropped_original_data,surface_coords[:-1],top_surf,partition_coord,disable_tqdm,scan_num)
            else:
                cropped_original_data = y_motion_correcting(cropped_original_data,surface_coords[-1:],top_surf,partition_coord,disable_tqdm,scan_num)
            top_surf = False
    else:
        if cancellation_flag and cancellation_flag():
            print("Registration cancelled during Y-motion correction")
            return None
        cropped_original_data = y_motion_correcting(cropped_original_data,surface_coords,top_surf,partition_coord,disable_tqdm,scan_num)


    # X-MOTION PART
    if cancellation_flag and cancellation_flag():
        print("Registration cancelled before X-motion correction")
        return None
        
    print('Starting X-motion')
    pbar.set_description(desc = f'Correcting {scan_num} X-Motion.....')
    test_detect_img = preprocess_img(cropped_original_data[:,:,static_flat])
    res_surface = MODEL_FEATURE_DETECT.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes = 0, device='cpu',agnostic_nms = True, augment = True)
    res_cells = MODEL_FEATURE_DETECT.predict(test_detect_img,iou = 0.5, save = False, verbose=False,classes = 1, device='cpu',agnostic_nms = True, augment = True)
    # result_list = res[0].summary()
    surface_coords = detect_areas(res_surface[0].summary(),pad_val = SURFACE_X_PAD, img_shape = test_detect_img.shape[0], expected_num = EXPECTED_SURFACES)
    cells_coords = detect_areas(res_cells[0].summary(),pad_val = CELLS_X_PAD, img_shape = test_detect_img.shape[0], expected_num = EXPECTED_CELLS)

    if (cells_coords is None) and (surface_coords is None):
        print(f'NO SURFACE OR CELLS DETECTED: {scan_num}')
        # with open(f'debugs/debug{scan_num}.txt', 'a') as f:
        #     f.write(f'NO SURFACE OR CELLS DETECTED: {scan_num}\n')
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

    if cancellation_flag and cancellation_flag():
        print("Registration cancelled before saving data")
        return None
        
    pbar.set_description(desc = 'Saving Data.....')
    if cropped_original_data.dtype != np.float64:
        cropped_original_data = cropped_original_data.astype(np.float64)
    os.makedirs(save_dirname,exist_ok=True)
    if not save_dirname.endswith('/'):
        save_dirname = save_dirname + '/'
    hdf5_filename = f'{save_dirname}{scan_num}.h5'
    try:
        with h5py.File(hdf5_filename, 'w') as hf:
            hf.create_dataset('volume', data=cropped_original_data, compression='gzip',compression_opts=5)
    except Exception as e:
        if cancellation_flag and cancellation_flag():
            print("Registration cancelled during file save")
            try:
                os.remove(hdf5_filename)
            except:
                pass
            return None
        raise e

def run_pipeline(dirname, disable_tqdm, use_model_x, save_dirname, expected_cells, expected_surfaces, cancellation_flag=None):
    data_dirname = dirname
    if data_dirname.endswith('/'):
        data_dirname = data_dirname[:-1]
    # Use provided save_dirname for checking existing files as well
    # check_save_dir = save_dirname if save_dirname else DATA_SAVE_DIR
    # if os.path.exists(save_dirname):
    #     done_scans = set([i.removesuffix('.h5') for i in os.listdir(save_dirname) if (i.startswith('scan'))])
    #     print(done_scans)
    # else:
    #     done_scans={}
    if data_dirname.lower().endswith('.h5'):
        data_type = 'h5'
        scans = [data_dirname.split('/')[-1].removesuffix('.h5')]
    else:
        data_type = 'dcm'
        scans = [data_dirname.split('/')[-1]]
    # scans = [i for i in os.listdir(data_dirname) if (i.startswith('scan')) and (i+'.h5' not in done_scans)]
    # scans = natsorted(scans)
    # scans = ['data'] ################ remove while running
    # data_type = scans[0].split('.')[-1]
    # data_type = 'dcm'
    # print('REMAINING',scans)
 
    pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█", disable=disable_tqdm)
    for scan_num in pbar:
        if cancellation_flag and cancellation_flag():
            print("Registration cancelled by user")
            return
        pbar.set_description(desc = f'Processing {scan_num}')
        main(
            data_dirname, scan_num, pbar, data_type,
            disable_tqdm=disable_tqdm,
            save_detections=False,
            use_model_x=use_model_x,
            save_dirname=save_dirname,
            expected_cells=expected_cells,
            expected_surfaces=expected_surfaces,
            cancellation_flag=cancellation_flag
        )

def gui_input(dirname, use_model_x, disable_tqdm, save_dirname, expected_cells, expected_surfaces, cancellation_flag=None):
    print(f"Data Load Directory: {dirname}")
    print(f"Use Model X: {use_model_x}")
    print(f"Disable Tqdm: {disable_tqdm}")
    print(f"Expected Cells: {expected_cells}")
    print(f"Expected Surfaces: {expected_surfaces}")
    print(f"Save Data Directory: {save_dirname}")
    
    run_pipeline(
        dirname=dirname,
        disable_tqdm=disable_tqdm,
        use_model_x=use_model_x,
        save_dirname=save_dirname,
        expected_cells=expected_cells,
        expected_surfaces=expected_surfaces,
        cancellation_flag=cancellation_flag
    )


if __name__ == "__main__":
    gui_input()
