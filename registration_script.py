import sys
import multiprocessing
multiprocessing.freeze_support()
import os
from natsort import natsorted
from tqdm import tqdm
from ultralytics import YOLO
from utils.util_funcs import resource_path, download_model
from registration_scripts.reg_worker import RegistrationWorker
import yaml
import torch
import time
import logging

class ProcessStdout:
    """A helper class to redirect stdout of a process to a multiprocessing.Queue."""
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

class RegistrationMaster:
    def __init__(self, config_path='datapaths.yaml', is_gui=False, ENABLE_MULTIPROC_SLURM = False,
                 DISABLE_TQDM = True, EXPECTED_SURFACES = 2, EXPECTED_CELLS = 2, BATCH_FLAG=False,
                 DATA_LOAD_DIR = None, DATA_SAVE_DIR = 'output/', USE_MODEL_LATERAL_TRANSLATION = False, 
                 SAVE_DETECTIONS = False, OUTPUT_QUEUE = None):
        """Initialize the registration pipeline with configuration and models"""
        self.is_gui_flag = is_gui
        self.output_queue = OUTPUT_QUEUE
        if not self.is_gui_flag:
            self.config = self._load_config(config_path)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            self.config = self._set_config(config_path, DATA_LOAD_DIR, DATA_SAVE_DIR,
                                            EXPECTED_SURFACES, EXPECTED_CELLS, BATCH_FLAG, DISABLE_TQDM,
                                            ENABLE_MULTIPROC_SLURM, USE_MODEL_LATERAL_TRANSLATION, SAVE_DETECTIONS)
            logging.basicConfig(level=logging.INFO, format='%(message)s', stream = ProcessStdout(self.output_queue))
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.BATCH_FLAG = self.config['FLAGS']['BATCH_FLAG']
        self.DATA_LOAD_DIR = self.config['PATHS']['DATA_LOAD_DIR']
        self.DATA_SAVE_DIR = self.config['PATHS']['DATA_SAVE_DIR']
        self.MODEL_FEATURE_DETECT_PATH = self.config['PATHS']['MODEL_FEATURE_DETECT_PATH']
        self.MODEL_X_TRANSLATION_PATH = self.config['PATHS']['MODEL_X_TRANSLATION_PATH']
        self.DISABLE_TQDM = self.config['FLAGS']['DISABLE_TQDM']
        self.USE_MODEL_LATERAL_TRANSLATION = self.config['FLAGS']['USE_MODEL_LATERAL_TRANSLATION']
        self.ENABLE_MULTIPROC_SLURM = self.config['FLAGS']['ENABLE_MULTIPROC_SLURM']
        self.save_detections = self.config['FLAGS']['SAVE_DETECTIONS']
        self.EXPECTED_SURFACES = self.config['VALUES']['EXPECTED_SURFACES']
        self.EXPECTED_CELLS = self.config['VALUES']['EXPECTED_CELLS']
        self.data_type = None
        self.models = self._load_models()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _set_config(self, config_path, DATA_LOAD_DIR, DATA_SAVE_DIR,
                    EXPECTED_SURFACES, EXPECTED_CELLS, BATCH_FLAG, DISABLE_TQDM,
                    ENABLE_MULTIPROC_SLURM, USE_MODEL_LATERAL_TRANSLATION, SAVE_DETECTIONS):
        """Set the configuration from YAML file, as we take inputs from GUI"""
        try:
            with open(resource_path(config_path), 'r') as f:
                config = yaml.safe_load(f)
        except:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        # Paths
        config['PATHS']['DATA_LOAD_DIR'] = DATA_LOAD_DIR
        config['PATHS']['DATA_SAVE_DIR'] = DATA_SAVE_DIR
        # Valuues
        config['VALUES']['EXPECTED_SURFACES'] = EXPECTED_SURFACES
        config['VALUES']['EXPECTED_SURFACES'] = EXPECTED_CELLS
        # Flags
        config['FLAGS']['BATCH_FLAG'] = BATCH_FLAG
        config['FLAGS']['DISABLE_TQDM'] = DISABLE_TQDM
        config['FLAGS']['ENABLE_MULTIPROC_SLURM'] = ENABLE_MULTIPROC_SLURM
        config['FLAGS']['USE_MODEL_LATERAL_TRANSLATION'] = USE_MODEL_LATERAL_TRANSLATION
        config['FLAGS']['SAVE_DETECTIONS'] = SAVE_DETECTIONS
        try:
            config['PATHS']['MODEL_FEATURE_DETECT_PATH'] = resource_path(config['PATHS']['MODEL_FEATURE_DETECT_PATH'])
            config['PATHS']['MODEL_X_TRANSLATION_PATH'] = resource_path(config['PATHS']['MODEL_X_TRANSLATION_PATH'])
        except:
            config['PATHS']['MODEL_FEATURE_DETECT_PATH'] = config['PATHS']['MODEL_FEATURE_DETECT_PATH']
            config['PATHS']['MODEL_X_TRANSLATION_PATH'] = config['PATHS']['MODEL_X_TRANSLATION_PATH']
        return config

    def _load_models(self):
        """Load models based on configuration"""
        model_x_translation_url = 'https://github.com/AKA2320/OCT_registration_framework/raw/refs/heads/main/models/transmorph_lateral_X_translation.pt'
        models = {}
        try:
            models['feature_yolo'] = YOLO(self.MODEL_FEATURE_DETECT_PATH)
            logging.info("YOLO Model Loaded Succesfully.")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}", exc_info=True)
            sys.exit("Failed to load YOLO model. Exiting.")
        if self.USE_MODEL_LATERAL_TRANSLATION:
            try:
                if not os.path.exists(self.MODEL_X_TRANSLATION_PATH):
                    logging.info("Model X not present in models.....Downloading the model.")  
                    download_model(model_x_translation_url,self.MODEL_X_TRANSLATION_PATH)
                    logging.info("Model Downloaded Succesfully.")
                # MODEL_X_TRANSLATION = torch.jit.load(self.MODEL_X_TRANSLATION_PATH, map_location=self.DEVICE)
                # MODEL_X_TRANSLATION.eval()
                # logging.info("Model X loaded successfully.")  
                MODEL_X_TRANSLATION = self.MODEL_X_TRANSLATION_PATH
            except Exception as e:
                logging.error(f"Error loading Model X: {e}", exc_info=True)
                logging.info("Proceeding without Model X translation.")  
                MODEL_X_TRANSLATION = None
        else:
            MODEL_X_TRANSLATION = None
        models['x_translation'] = MODEL_X_TRANSLATION
        return models
        
    def spawn_worker_and_run_pipeline(self):
        self.DATA_LOAD_DIR = self.DATA_LOAD_DIR
        if self.DATA_LOAD_DIR.endswith('/'):
            self.DATA_LOAD_DIR = self.DATA_LOAD_DIR[:-1]
        if not self.BATCH_FLAG:
            if self.DATA_LOAD_DIR.lower().endswith('.h5'):
                self.data_type = 'h5'
                scans = [self.DATA_LOAD_DIR.split('/')[-1].removesuffix('.h5')]
            elif os.listdir(self.DATA_LOAD_DIR)[0].endswith(('.dcm','.DCM')):
                self.data_type = 'dcm'
                scans = [self.DATA_LOAD_DIR.split('/')[-1]]
            else:
                if "binfiles" in os.listdir(self.DATA_LOAD_DIR):
                    self.data_type = 'bin'
                    scans = [self.DATA_LOAD_DIR.split('/')[-1]]
                else:
                    raise FileNotFoundError("Missing data, make sure its formatted according to README")
        elif self.BATCH_FLAG:
            scans = [i for i in os.listdir(self.DATA_LOAD_DIR) if (i.startswith('scan'))]
            scans = natsorted(scans)
            first_path = os.listdir(os.path.join(self.DATA_LOAD_DIR,scans[0]))[0]
            if first_path.endswith('.h5'):
                self.data_type = 'h5'
            elif first_path.endswith(('.dcm', '.DCM')):
                self.data_type = 'dcm'
            else:
                self.data_type = 'bin'

        pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█", disable = self.DISABLE_TQDM)
        if not self.ENABLE_MULTIPROC_SLURM:
            for scan_num in pbar:
                pbar.set_description(desc = f'Processing {scan_num}')
                start = time.time()
                self._launch_process_wrapper(scan_num, pbar)
                # scan_worker = RegistrationWorker()
                # scan_worker.process_scan()
                logging.info(f'Time taken: {time.time() - start:.2f} seconds')

        elif self.ENABLE_MULTIPROC_SLURM:
            self.DISABLE_TQDM = True
            # pbar = tqdm(scans, desc='Processing Scans',total = len(scans), ascii="░▖▘▝▗▚▞█", disable = self.DISABLE_TQDM)
            self._dask_slurm_spawner(scans)

    def init_dask_worker(self):
        self.MODEL_FEATURE_DETECT = self.models['feature_yolo']
        self.MODEL_X_TRANSLATION = self.models['x_translation']

    def _dask_slurm_spawner(self,scans):
        try:
            from dask_jobqueue import SLURMCluster
            from dask.distributed import Client
            from dask import delayed, compute
        except ImportError as e:
            raise ImportError("Dask and dask_jobqueue modules are required for multiprocessing with SLURM") from e
        multiproc_args_list = [(scan_num) for scan_num in scans]
        logging.info("Setting up Dask SLURM cluster...")
        cluster = SLURMCluster(
            queue='general',
            account='ACC_NUMBER',
            cores=2, 
            processes=1,
            memory='20GB',
            walltime='03:00:00',
            job_extra_directives=[
                "--nodes=1",
                "--job-name=oct_reg",
                "--output=my_job.out",
                "--error=my_job.err"
            ],
            python=sys.executable,
        )
        cluster.scale(jobs=len(scans))
        # Attach client
        client = Client(cluster)
        client.run(self.init_dask_worker)
        tasks = [delayed(self._launch_process_wrapper)(args) for args in multiproc_args_list]
        logging.info("Submitting tasks to the cluster...")
        results = compute(*tasks)
        logging.info("Jobs DONE")
        # results = client.gather(results)
        try:
            client.close()
            cluster.close()
        except:
            pass
    
    def _launch_process_wrapper(self, scan_num, pbar = tqdm([],total = 1,disable=True)):
        # pbar = tqdm([],total = 1,disable=True)
        scan_worker = RegistrationWorker(self.config, self.models, scan_num, pbar, self.DATA_LOAD_DIR, 
                                        self.data_type, self.save_detections, self.DEVICE, self.DISABLE_TQDM,
                                        self.EXPECTED_SURFACES, self.EXPECTED_CELLS)
        scan_worker.process_scan()

def start_reg_cli():
    registration_pipeline = RegistrationMaster(config_path = 'datapaths.yaml', is_gui = False)
    registration_pipeline.spawn_worker_and_run_pipeline()

def start_reg_gui(DATA_LOAD_DIR, DATA_SAVE_DIR, EXPECTED_SURFACES, EXPECTED_CELLS, BATCH_FLAG,
                 USE_MODEL_LATERAL_TRANSLATION, SAVE_DETECTIONS, OUTPUT_QUEUE):
    registration_pipeline = RegistrationMaster(config_path = 'datapaths.yaml', is_gui = True, DATA_LOAD_DIR = DATA_LOAD_DIR,
                            DATA_SAVE_DIR = DATA_SAVE_DIR, EXPECTED_SURFACES = EXPECTED_SURFACES, 
                            EXPECTED_CELLS = EXPECTED_CELLS, BATCH_FLAG = BATCH_FLAG, DISABLE_TQDM = True,
                            ENABLE_MULTIPROC_SLURM = False, USE_MODEL_LATERAL_TRANSLATION = USE_MODEL_LATERAL_TRANSLATION,
                            SAVE_DETECTIONS = SAVE_DETECTIONS, OUTPUT_QUEUE = OUTPUT_QUEUE) 
    registration_pipeline.spawn_worker_and_run_pipeline()

if __name__ == "__main__":
    # Initialize RegistrationMaster class
    p = multiprocessing.Process(target = start_reg_cli)
    p.start()
    p.join()
    