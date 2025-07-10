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
import click

with open('datapaths.yaml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_FEATURE_DETECT = YOLO(config['PATHS']['MODEL_FEATURE_DETECT_PATH'])
# USE_MODEL_X = config['PATHS']['USE_MODEL_X'] # This will now come from command line
MODEL_X_TRANSLATION_PATH = config['PATHS']['MODEL_X_TRANSLATION_PATH']
SURFACE_Y_PAD = 20
SURFACE_X_PAD = 10
CELLS_X_PAD = 5
# DATA_LOAD_DIR = config['PATHS']['DATA_LOAD_DIR'] # This will now come from command line
DATA_SAVE_DIR = config['PATHS']['DATA_SAVE_DIR'] # Keep this for default if no save_dirname is provided
EXPECTED_SURFACES = config['PATHS']['EXPECTED_SURFACES']
EXPECTED_CELLS = config['PATHS']['EXPECTED_CELLS']

# Remove global MODEL_X_TRANSLATION loading, will be done in main conditionally
# if USE_MODEL_X:
#     DEVICE = 'cpu'
#     MODEL_X_TRANSLATION = torch.load(MODEL_X_TRANSLATION_PATH, map_location=DEVICE, weights_only=False)
#     MODEL_X_TRANSLATION.eval()
# else:
#     MODEL_X_TRANSLATION = None

def main(dirname, scan_num, pbar, data_type, disable_tqdm, save_detections, use_model_x, save_dirname):
    global MODEL_FEATURE_DETECT
    # global MODEL_X_TRANSLATION # Removed global

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

    if not os.path.exists(dirname):
        raise FileNotFoundError(f"Directory {dirname} not found")
    if not os.path.exists(os.path.join(dirname, scan_num)):
        raise FileNotFoundError(f"Scan {scan_num} not found in {dirname}")
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
        with open(f'debugs/debug{scan_num}.txt', 'a') as f:
            f.write(f'NO SURFACE DETECTED: {scan_num}\n')
            f.write(f'min range: {cropped_original_data.min(),cropped_original_data.max()}\n')
        print(f'NO SURFACE DETECTED: {scan_num}')
        return None

    # FLATTENING PART
    pbar.set_description(desc = f'Flattening {scan_num}.....')
    # print('SURFACE COORDS:',surface_coords)
    static_flat = np.argmax(np.sum(cropped_original_data[:,surface_coords[0,0]:surface_coords[0,1],:],axis=(0,1)))
    top_surf = True
    for i in range(surface_coords.shape[0]):
        UP_flat,DOWN_flat = surface_coords[i,0], surface_coords[i,1]
        UP_flat = max(UP_flat,0)
        DOWN_flat = min(DOWN_flat, cropped_original_data.shape[2])
        cropped_original_data = flatten_data(cropped_original_data,UP_flat,DOWN_flat,top_surf,disable_tqdm,scan_num)
        top_surf = False

    # Y-MOTION PART
    pbar.set_description(desc = f'Correcting {scan_num} Y-Motion.....')
    top_surf = True
    for i in range(surface_coords.shape[0]):
        UP_y,DOWN_y = surface_coords[i,0], surface_coords[i,1]
        UP_y = max(UP_y,0)
        DOWN_y = min(DOWN_y, cropped_original_data.shape[2])
        cropped_original_data = y_motion_correcting(cropped_original_data,UP_y,DOWN_y,top_surf,disable_tqdm,scan_num)
        top_surf = False

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
        with open(f'debugs/debug{scan_num}.txt', 'a') as f:
            f.write(f'NO SURFACE OR CELLS DETECTED: {scan_num}\n')
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
    # Pass MODEL_X_TRANSLATION (which might be None) to all_trans_x
    tr_all = all_trans_x(cropped_original_data,UP_x,DOWN_x,valid_args,enface_extraction_rows,disable_tqdm,scan_num, MODEL_X_TRANSLATION)
    for i in tqdm(range(1,cropped_original_data.shape[0],2),desc='X-motion warping',disable=disable_tqdm, ascii="░▖▘▝▗▚▞█", leave=False):
        cropped_original_data[i]  = warp(cropped_original_data[i],AffineTransform(matrix=tr_all[i]),order=3)

    pbar.set_description(desc = 'Saving Data.....')
    if cropped_original_data.dtype != np.float64:
        cropped_original_data = cropped_original_data.astype(np.float64)
    # Use the provided save_dirname if available, otherwise use the default DATA_SAVE_DIR
    folder_save = save_dirname if save_dirname else DATA_SAVE_DIR
    os.makedirs(folder_save,exist_ok=True)
    hdf5_filename = os.path.join(folder_save, f'{scan_num}.h5') # Use os.path.join for path construction
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('volume', data=cropped_original_data, compression='gzip',compression_opts=5)

def run_pipeline(disable_tqdm=False, use_model_x=False, save_dirname=None):
    data_dirname = DATA_LOAD_DIR
    if data_dirname.endswith('/'):
        data_dirname = data_dirname[:-1]
    # Use provided save_dirname for checking existing files as well
    check_save_dir = save_dirname if save_dirname else DATA_SAVE_DIR
    if os.path.exists(check_save_dir):
        done_scans = set([i.removesuffix('.h5') for i in os.listdir(check_save_dir) if (i.startswith('scan'))])
        print(done_scans)
    else:
        done_scans={}
    if os.listdir(DATA_LOAD_DIR)[0].endswith('.dcm'):
        data_type = 'dcm'
    # scans = [i for i in os.listdir(data_dirname) if (i.startswith('scan')) and (i+'.h5' not in done_scans)]
    # scans = natsorted(scans)
    scans = ['data'] ################ remove while running
    # data_type = scans[0].split('.')[-1]
    data_type = 'dcm'
    print('REMAINING',scans)
    pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█", disable=disable_tqdm)
    for scan_num in pbar:
        pbar.set_description(desc = f'Processing {scan_num}')
        # Pass use_model_x and save_dirname to the main function
        main(data_dirname, scan_num, pbar, data_type, disable_tqdm = disable_tqdm, save_detections = False, use_model_x=use_model_x, save_dirname=save_dirname) # Pass disable_tqdm here too


@click.command()
@click.option('--dirname', type=click.Path(exists=True), help='Directory to load data from')
@click.option('--use-model-x', is_flag=True, default=True, help='Use Model X for translation correction')
@click.option('--disable-tqdm', is_flag=True, default=False, help='Disable tqdm progress bars')
@click.option('--save-dirname', type=click.Path(), default=None, help='Directory to save output data') # Added save-dirname option
def cli(dirname, use_model_x, disable_tqdm, save_dirname):
    global DATA_LOAD_DIR
    if dirname:
        DATA_LOAD_DIR = dirname
    print(f"Data Load Directory: {DATA_LOAD_DIR}")
    print(f"Use Model X: {use_model_x}")
    print(f"Save Data Directory: {save_dirname}") # Print the save directory
    # Pass all arguments to run_pipeline
    run_pipeline(disable_tqdm=disable_tqdm, use_model_x=use_model_x, save_dirname=save_dirname)

if __name__ == "__main__":
    cli()
