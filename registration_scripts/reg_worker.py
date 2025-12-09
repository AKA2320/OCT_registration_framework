import numpy as np
import os
from tqdm import tqdm
import h5py
import shutil
from utils.transmorph_helper_funcs import preprocess_img, detect_areas, crop_data
from utils.load_data_funcs import load_data_dcm, load_h5_data
from utils.flatten_correction_util_funcs import flatten_data
from utils.y_correction_util_funcs import y_motion_correcting
from utils.x_correction_util_funcs import x_motion_correction
from utils.util_funcs import ncc, warp_image_affine
import gc
import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout )
class ProcessStdout:
    """A helper class to redirect stdout of a process to a multiprocessing.Queue."""
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass


class RegistrationWorker:
    def __init__(self, config, models, scan_num, pbar, DATA_LOAD_DIR,
                data_type, save_detections, DEVICE, DISABLE_TQDM,
                EXPECTED_SURFACES, EXPECTED_CELLS):
        self.config = config
        self.MODEL_FEATURE_DETECT = models['feature_yolo']
        self.MODEL_X_TRANSLATION = models['x_translation']
        self.scan_num = scan_num
        self.pbar = pbar
        self.DATA_LOAD_DIR = DATA_LOAD_DIR
        self.data_type = data_type
        self.save_detections = save_detections
        self.DEVICE = DEVICE
        self.DISABLE_TQDM = DISABLE_TQDM
        # Paths
        self.DATA_SAVE_DIR = self.config['PATHS']['DATA_SAVE_DIR']
        # Values
        self.SURFACE_Y_PAD = self.config['VALUES']['SURFACE_Y_PAD']
        self.SURFACE_X_PAD = self.config['VALUES']['SURFACE_X_PAD'] 
        self.CELLS_X_PAD = self.config['VALUES']['CELLS_X_PAD'] 
        self.EXPECTED_SURFACES = EXPECTED_SURFACES
        self.EXPECTED_CELLS = EXPECTED_CELLS

    def _load_data(self):
        """Load data based on type (h5 or dcm)"""
        if self.data_type == 'h5':
            if not os.path.exists(self.DATA_LOAD_DIR):
                raise FileNotFoundError(f"Scan {self.DATA_LOAD_DIR} not found")
            return load_h5_data(self.DATA_LOAD_DIR, self.scan_num)
        elif self.data_type == 'dcm':
            if not os.path.exists(self.DATA_LOAD_DIR):
                raise FileNotFoundError(f"Directory {self.DATA_LOAD_DIR} not found")
            return load_data_dcm(self.DATA_LOAD_DIR, self.scan_num)
        else:
            raise ValueError("Unsupported data type. Use 'h5' or 'dcm'.")
    
    def process_scan(self):
        """Process a single scan through the full registration pipeline"""
        try:
            # Load data
            self.pbar.set_description(desc = f'Loading data for {self.scan_num}')
            original_data = self._load_data()
            if original_data is None:
                raise ValueError("Data loading returned None, check the input paths.")

            # Feature detection and initial cropping
            self.pbar.set_description(desc = f'Cropping data for {self.scan_num}')
            cropped_data = self._detect_features_crop_data(original_data)
            if cropped_data is None:
                return None
            del original_data  # Free memory, we now only use cropped_data
            # Save unregistered data
            self._save_data(cropped_data, '_unregistered')

            # Get surface coordinates
            self.surface_coords, self.partition_coord = self._get_surface_coords_cropped(cropped_data)
            if self.surface_coords is None:
                logging.warning(f'No surface detected in cropped data, cropping error: {self.scan_num}')
                return None

            # Data processing pipeline
            cropped_data = self._process_data_pipeline(cropped_data)

            # Save final registered data
            self._save_data(cropped_data, '_registered')
            return cropped_data

        except Exception as e:
            logging.error(f"Error processing scan {self.scan_num}: {e}", exc_info=True)
            return None

    
    def _detect_features_crop_data(self, original_data):
        """Detect features using YOLO model and crop data"""
        static_flat = np.argmax(np.sum(original_data[:,:,:], axis=(0,1)))
        test_detect_img = preprocess_img(original_data[:,:,static_flat])
        detections_save_dir = os.path.join(self.DATA_SAVE_DIR, 'detections')
        if os.path.exists(os.path.join(detections_save_dir,self.scan_num)):
            try:
                shutil.rmtree(os.path.join(detections_save_dir,self.scan_num))
            except OSError as e:
                print(f"Error removing existing detection directory: {e}")
        res_surface_cells = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=self.save_detections,
                                                            max_det = self.EXPECTED_SURFACES + self.EXPECTED_CELLS + 5, # some extra detctions to save but not use
                                                            project = detections_save_dir, name=self.scan_num, verbose=False,
                                                            classes= 0, device=self.DEVICE, agnostic_nms=True, augment=True)
        
        res_surface = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=False, max_det = self.EXPECTED_SURFACES,
                                                              verbose=False, classes = 0, device=self.DEVICE, agnostic_nms=True, augment=True)
        # surface_crop_coords = [i for i in res_surface_cells[0].summary() if i['name']=='surface']

        res_cells = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=False, max_det = self.EXPECTED_CELLS,
                                                        verbose=False, classes = 1, device=self.DEVICE, agnostic_nms=True, augment=True)
        # cells_crop_coords = [i for i in res_surface_cells[0].summary() if i['name']=='cells']
        
        surface_crop_coords = detect_areas(res_surface[0].summary(), pad_val = self.SURFACE_Y_PAD,
                                           img_shape=test_detect_img.shape[0], expected_num=self.EXPECTED_SURFACES)
        cells_crop_coords = detect_areas(res_cells[0].summary(), pad_val = self.SURFACE_Y_PAD,
                                         img_shape=test_detect_img.shape[0], expected_num=self.EXPECTED_CELLS)
        if surface_crop_coords is None:
            logging.warning(f'No surface detected in orignal data for cropping: {self.scan_num}')
            return None
        return crop_data(original_data, surface_crop_coords, cells_crop_coords, original_data.shape[1])

    def _get_surface_coords_cropped(self, cropped_data):
        """Get surface coordinates and partition coordinates from cropped data"""
        static_flat = np.argmax(np.sum(cropped_data[:,:,:], axis=(0,1)))
        test_detect_img = preprocess_img(cropped_data[:,:,static_flat])
        
        res_surface = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=False, verbose=False, max_det = self.EXPECTED_SURFACES,
                                                        classes=0, device=self.DEVICE, agnostic_nms=True, augment=True)
        
        surface_coords = detect_areas(res_surface[0].summary(), pad_val = self.SURFACE_Y_PAD,
                                    img_shape= test_detect_img.shape[0], expected_num = self.EXPECTED_SURFACES)
        
        if surface_coords is None:
            return None
            
        if self.EXPECTED_SURFACES > 1:
            # We only flatten/correct Y motion in standard interference
            # partition_coord tells us where to split the data
            partition_coord = np.ceil(np.mean(np.mean(surface_coords[-2:], axis=1))).astype(int) 
        else:
            partition_coord = None
        return surface_coords, partition_coord

    def _process_data_pipeline(self, cropped_data):
        """Memory optimized processing pipeline with explicit cleanup."""

        # Flatten data
        self.pbar.set_description(desc = f'Flattening {self.scan_num} surfaces.....')
        if self.DISABLE_TQDM:
            logging.info("Starting Flattening for data")
        cropped_data = self._flatten_data(cropped_data)

        # Force garbage collection after flattening
        gc.collect()

        # Correct Y motion
        self.pbar.set_description(desc = f'Correcting {self.scan_num} Y-Motion.....')
        if self.DISABLE_TQDM:
            logging.info("Starting Y-Motion correction for data")
        cropped_data = self._correct_y_motion(cropped_data)

        # Force garbage collection after Y-motion correction
        gc.collect()

        # Correct X motion
        self.pbar.set_description(desc = f'Correcting {self.scan_num} X-Motion.....')
        if self.DISABLE_TQDM:
            logging.info("Starting X-Motion correction for data")
        cropped_data = self._correct_x_motion(cropped_data)

        # Final cleanup
        gc.collect()

        return cropped_data

    def _flatten_data(self, data):
        """Flatten the data based on surface coordinates"""
        top_surf = True
        if self.surface_coords.shape[0] > 1:
            for _ in range(2):
                if top_surf:
                    data = flatten_data(data, self.surface_coords[:-1], top_surf,
                                         self.partition_coord, self.DISABLE_TQDM, self.scan_num)
                else:
                    data = flatten_data(data, self.surface_coords[-1:], top_surf,
                                         self.partition_coord, self.DISABLE_TQDM, self.scan_num)
                top_surf = False
        else:
            data = flatten_data(data, self.surface_coords, top_surf, self.partition_coord, self.DISABLE_TQDM, self.scan_num)
        return data

    def _correct_y_motion(self, data):
        """Correct Y-axis motion in the data"""
        top_surf = True
        if self.surface_coords.shape[0] > 1:
            for _ in range(2):
                if top_surf:
                    data = y_motion_correcting(data, self.surface_coords[:-1], top_surf,
                                                self.partition_coord, self.DISABLE_TQDM, self.scan_num)
                else:
                    data = y_motion_correcting(data, self.surface_coords[-1:], top_surf,
                                                self.partition_coord, self.DISABLE_TQDM, self.scan_num)
                top_surf = False
        else:
            data = y_motion_correcting(data, self.surface_coords, top_surf, self.partition_coord,
                                        self.DISABLE_TQDM, self.scan_num)
        return data

    def _correct_x_motion(self, data):
        """Correct X-axis motion in the data"""
        static_flat = np.argmax(np.sum(data[:,self.surface_coords[0,0]:self.surface_coords[0,1],:], axis=(0,1)))
        test_detect_img = preprocess_img(data[:,:,static_flat])
        res_surface = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=False, verbose=False, max_det = self.EXPECTED_SURFACES,
                                                        classes=0, device=self.DEVICE, agnostic_nms=True, augment=True)
        res_cells = self.MODEL_FEATURE_DETECT.predict(test_detect_img, iou=0.5, save=False, verbose=False, max_det = self.EXPECTED_CELLS,
                                                        classes=1, device=self.DEVICE, agnostic_nms=True, augment=True)
        surface_coords_for_x = detect_areas(res_surface[0].summary(), pad_val=self.SURFACE_X_PAD,
                                            img_shape=test_detect_img.shape[0], expected_num=self.EXPECTED_SURFACES)
        cells_coords_for_x = detect_areas(res_cells[0].summary(), pad_val=self.CELLS_X_PAD,
                                            img_shape=test_detect_img.shape[0],expected_num=self.EXPECTED_CELLS)
        
        if (cells_coords_for_x is None) and (surface_coords_for_x is None):
            logging.warning(f'No surface or cells detected: {self.scan_num}')
            return None
            
        enface_extraction_rows = []
        if surface_coords_for_x is not None:
            static_y_motion = np.argmax(np.sum(data[:,surface_coords_for_x[0,0]:surface_coords_for_x[0,1],:], axis=(1,2)))    
            errs = []
            for i in range(data.shape[0]):
                errs.append(ncc(data[static_y_motion,:,:], data[i,:,:]))
            errs = np.squeeze(errs)
            valid_args = np.squeeze(np.argwhere(errs>0.7))
            for i in range(surface_coords_for_x.shape[0]):
                val = np.argmax(np.sum(np.max(data[:,surface_coords_for_x[i,0]:surface_coords_for_x[i,1],:], axis=0), axis=1))
                enface_extraction_rows.append(surface_coords_for_x[i,0]+val) # val is argmax so always u32 and surface_coords_for_x is u32 check detect_areas()
        else:
            valid_args = np.arange(data.shape[0])

        tr_all = x_motion_correction(data, cells_coords_for_x, valid_args, enface_extraction_rows,
                            self.DISABLE_TQDM, self.scan_num, self.MODEL_X_TRANSLATION)
        
        for i in tqdm(range(1, data.shape[0], 2),desc='X-motion warping',
                      disable=self.DISABLE_TQDM,ascii="░▖▘▝▗▚▞█", leave=False):
            data[i] = warp_image_affine(data[i], (tr_all[i, 0, 2], 0))
        return data

    def _save_data(self, data, suffix=''):
        """Save processed data to HDF5 file"""
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        folder_save = self.DATA_SAVE_DIR
        if not folder_save.endswith('/'):
            folder_save = folder_save + '/'
        if suffix == '_unregistered':
            folder_save = os.path.join(folder_save, 'unregistered/')
        elif suffix == '_registered':
            folder_save = os.path.join(folder_save, 'registered/')
        os.makedirs(folder_save, exist_ok=True)
        hdf5_filename = f'{folder_save}{self.scan_num}{suffix}.h5'
        with h5py.File(hdf5_filename, 'w') as hf:
            hf.create_dataset('volume', data=data, compression='gzip', compression_opts=5)
