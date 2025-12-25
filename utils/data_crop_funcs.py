from collections import defaultdict
import numpy as np
from utils.util_funcs import min_max


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

def map_coords_after_crop(coords):
    temp_coords = coords.copy()
    for idx in range(temp_coords.shape[0]-1):
        if temp_coords[idx][1] > temp_coords[idx+1][0]:
            temp_coords[idx][1] = temp_coords[idx+1][0] = int(np.mean((temp_coords[idx][1], temp_coords[idx+1][0])))
    lengths = [s[1] - s[0] for s in temp_coords]
    new_coords = []
    offset = 0
    for coord_new in np.cumsum(lengths):
        new_coords.append([offset, coord_new])
        offset = coord_new
    return np.array(new_coords).astype(np.uint16)

def crop_data(data, surface_coords, cells_coords, max_crop_shape):
    merged_coords = []
    new_surface_coords = None
    new_cell_coords = None
    if surface_coords is not None:
        surface_coords[:,0],surface_coords[:,1] = surface_coords[:,0]-30, surface_coords[:,1]+30 # 30 is for padding, makes it atleast 60 pixels for transmorph to work
        surface_coords = np.where(surface_coords<0,0,surface_coords)
        surface_coords = np.where(surface_coords>max_crop_shape,max_crop_shape-1,surface_coords)
        merged_coords.extend([*surface_coords])
        new_surface_coords = map_coords_after_crop(surface_coords)
    if cells_coords is not None:
        cells_coords[:,0],cells_coords[:,1] = cells_coords[:,0]-30, cells_coords[:,1]+30 # 30 is for padding, makes it atleast 60 pixels for transmorph to work
        cells_coords = np.where(cells_coords<0,0,cells_coords)
        cells_coords = np.where(cells_coords>max_crop_shape,max_crop_shape-1,cells_coords)
        merged_coords.extend([*cells_coords])
        new_cell_coords = map_coords_after_crop(cells_coords)
    merged_coords = merge_intervals([*merged_coords])
    data = data[:, np.r_[tuple(np.r_[start:end] for start, end in merged_coords)], :]
    return data, new_surface_coords, new_cell_coords

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