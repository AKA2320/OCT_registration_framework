import numpy as np
from scipy import interpolate
import os
from natsort import natsorted
import glob



# --- TARGET GEOMETRY ---
# DEFAULT_SHIFT_VAL = 83          # Registration Shift
# OUTPUT_WIDTH = 417      # Target Width
# OUTPUT_DEPTH = 1200     # Target Depth

# # --- CONTRAST ---
# # MAX_CONTRAST = np.iinfo(np.uint16).max
# MAX_CONTRAST = 255


def load_calibration(path):
    try:
        lamda = np.loadtxt(path, usecols=0)
    except ValueError:
        lamda = np.loadtxt(path, usecols=0, skiprows=1)
    return lamda

def prepare_k_linearization(lamda, SPECTROMETER_PIXELS):
    k_raw = (2 * np.pi) / lamda
    if k_raw[0] > k_raw[-1]:
        k_raw = k_raw[::-1]
        flip_spectrum = True
    else:
        flip_spectrum = False
    k_linear = np.linspace(np.min(k_raw), np.max(k_raw), SPECTROMETER_PIXELS)
    return k_raw, k_linear, flip_spectrum

def ncc(array1, array2):
    # Flatten views and Subtract means 
    a1 = array1.ravel()-array1.mean()
    a2 = array2.ravel()-array2.mean()
    # Compute normalized correlation efficiently
    numerator = np.dot(a1, a2)
    denominator = np.linalg.norm(a1) * np.linalg.norm(a2)
    return np.divide(numerator, denominator) if denominator != 0 else 0.0

def err_func_crop_val(crop_val, data):
    shifted_idata = data[int(crop_val):,100:]
    mid = shifted_idata.shape[0]//2 
    a, b = shifted_idata[:mid], shifted_idata[-mid:]
    b = np.flip(b,axis=0)
    return float(1 - ncc(a,b))

def process_file_binfiles(file_path, k_raw, k_linear, flip_spectrum,
                          BUFFER_LINES, SPECTROMETER_PIXELS, FFT_SIZE):
    # --- 1. Load Data ---
    file_size = os.path.getsize(file_path)
    expected_bytes = BUFFER_LINES * SPECTROMETER_PIXELS * 2
    header_size = file_size - expected_bytes
    
    with open(file_path, 'rb') as f:
        if header_size > 0:
            f.seek(header_size)
        raw_data = np.fromfile(f, dtype=np.uint16)
    spectrogram = raw_data.reshape((BUFFER_LINES, SPECTROMETER_PIXELS)).astype(np.float32)

    # --- 2. Signal Processing ---
    dc_vector = np.mean(spectrogram, axis=0)
    spectrogram = spectrogram - dc_vector

    # Spectral Flip
    if flip_spectrum:
        spectrogram = spectrogram[:, ::-1]

    # K-Space Linearization

    f_interp = interpolate.make_interp_spline(k_raw, spectrogram, k=1, axis=1)
    spectrogram_linear = f_interp(k_linear)

    # FFT with ZERO PADDING
    fft_res = np.fft.fft(spectrogram_linear, n=FFT_SIZE, axis=1)
    magnitude = np.abs(fft_res) # 1030, 4096


    # --- 3. Depth Cropping ---
    # We take the first 1200 points.
    # Since we padded to 4096, indices 0-2048 are positive frequencies.
    # Index 0 is DC (Left wall).
    crop_depth = magnitude.shape[1]//2
    image_data = magnitude[:, :crop_depth] # (1300,1200)
    # image_data = magnitude[:, :OUTPUT_DEPTH] 
    return image_data
    
def reconstruct_frames(image_data, DEFAULT_SHIFT_VAL):
    # --- 4. Contrast Scaling ---
    # image_data = np.clip(image_data, 0, MAX_CONTRAST)
    # image_data = (image_data / image_data.max()) * 255.0

    # --- 5. Registration & Cropping (The 417 Width Fix) ---
    
    # A. Split Raw Frames
    image_data = image_data[int(DEFAULT_SHIFT_VAL):]
    mid_crop = image_data.shape[0]//2
    frame_1_raw = image_data[:mid_crop, :]
    frame_2_raw = image_data[-mid_crop:, :]
    
    # B. Flip Backward Frame
    frame_2_flipped = np.flip(frame_2_raw, axis=0)
    
    # C. Transpose
    # # Frame 1: Crop the start (Shift) to align
    # f1_reg = frame_1_raw[DEFAULT_SHIFT_VAL : DEFAULT_SHIFT_VAL + OUTPUT_WIDTH, :]
    # # Frame 2: Crop the end (Shift) to align
    # f2_reg = frame_2_flipped[ : OUTPUT_WIDTH, :]
    f1_reg = np.flip(frame_1_raw.transpose(1,0), axis=0)
    f2_reg = np.flip(frame_2_flipped.transpose(1,0),axis=0)
        
    return f1_reg.astype(np.float32), f2_reg.astype(np.float32)

