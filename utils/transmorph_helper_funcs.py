import torch
from torchvision import transforms
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from utils.util_funcs import min_max, merge_intervals


## Misc Functions
def filter_list(result_list, expected_num):
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
    return coords.astype(np.uint32)

def preprocess_img(data):
    data = data.transpose(1,0)
    data = min_max(data)
    data = (data*255).astype(np.uint8)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20, 20))
    # data = clahe.apply(data)
    data = np.dstack([[data]*3]).transpose(1,2,0)
    data = np.ascontiguousarray(data)
    return data

def crop_data(data,surface_coords,cells_coords,max_crop_shape):
    uncroped_data = data
    merged_coords = []
    if surface_coords is not None:
        surface_coords[:,0],surface_coords[:,1] = surface_coords[:,0]-30, surface_coords[:,1]+30 # 30 is for padding, makes it atleast 60 pixels for transmorph to work
        surface_coords = np.where(surface_coords<0,0,surface_coords)
        surface_coords = np.where(surface_coords>max_crop_shape,max_crop_shape-1,surface_coords)
        merged_coords.extend([*surface_coords])
    if cells_coords is not None:
        cells_coords[:,0],cells_coords[:,1] = cells_coords[:,0]-30, cells_coords[:,1]+30 # 30 is for padding, makes it atleast 60 pixels for transmorph to work
        cells_coords = np.where(cells_coords<0,0,cells_coords)
        cells_coords = np.where(cells_coords>max_crop_shape,max_crop_shape-1,cells_coords)
        merged_coords.extend([*cells_coords])
    merged_coords = merge_intervals([*merged_coords])
    uncroped_data = uncroped_data[:, np.r_[tuple(np.r_[start:end] for start, end in merged_coords)], :]
    return uncroped_data

class CropOrPad():
    def __init__(self, target_shape: tuple):
        if not isinstance(target_shape, (tuple, list)) or len(target_shape) != 2:
            raise ValueError("target_shape must be a tuple or list of two integers (height, width).")
        self.target_height, self.target_width = target_shape

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        is_grayscale = False
        if img.dim() == 2: # (H, W) grayscale
            is_grayscale = True
            img = img.unsqueeze(0) # Add a channel dimension: (1, H, W)
        elif img.dim() == 3: # (C, H, W) color
            pass
        else:
            raise ValueError(f"Unsupported image tensor dimensions: {img.dim()}. Expected 2 or 3.")
        current_channels, current_height, current_width = img.shape
        # --- Padding Logic ---
        pad_top = max(0, (self.target_height - current_height) // 2)
        pad_bottom = max(0, self.target_height - current_height - pad_top)
        pad_left = max(0, (self.target_width - current_width) // 2)
        pad_right = max(0, self.target_width - current_width - pad_left)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            # F.pad expects padding in the order (left, right, top, bottom) for 2D spatial dims
            img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # --- Cropping Logic ---
        # Recalculate dimensions after potential padding
        _, current_height_padded, current_width_padded = img.shape

        if current_height_padded > self.target_height or current_width_padded > self.target_width:
            crop_start_h = max(0, (current_height_padded - self.target_height) // 2)
            crop_end_h = crop_start_h + self.target_height
            crop_start_w = max(0, (current_width_padded - self.target_width) // 2)
            crop_end_w = crop_start_w + self.target_width

            # Crop the image
            img = img[:, crop_start_h:crop_end_h, crop_start_w:crop_end_w]

        if is_grayscale:
            img = img.squeeze(0) # Remove the channel dimension if it was grayscale initially

        return img

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()

    # Prevent division by zero if all values are the same
    if max_val == min_val:
        return torch.zeros_like(tensor)

    return (tensor - min_val) / (max_val - min_val)

transform = transforms.Compose([
    transforms.ToTensor(),
    CropOrPad((64,416)),
])

def infer_x_translation(model_obj, static_np, moving_np, DEVICE = 'cpu'):
    static_np = transform(static_np)
    moving_np = transform(moving_np)
    
    # Add batch and channel dim: (1, 1, H, W)
    static_np = normalize(static_np.unsqueeze(0)).to(DEVICE)
    moving_np = normalize(moving_np.unsqueeze(0)).to(DEVICE)

    # Concat and infer
    with torch.no_grad():
        input_pair = torch.cat([static_np, moving_np], dim=1).double().to(DEVICE)  # shape: (1, 2, H, W)
        _, pred_translation = model_obj(input_pair)

        input_pair_rev = torch.cat([moving_np, static_np], dim=1).double().to(DEVICE)  # shape: (1, 2, H, W)
        _, pred_translation_rev = model_obj(input_pair_rev)
    return (pred_translation.squeeze().numpy()[0], pred_translation_rev.squeeze().numpy()[0])
