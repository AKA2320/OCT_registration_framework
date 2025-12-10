import numpy as np
import os
import cv2
import requests
import sys


def ncc(array1, array2):
    # Flatten views and Subtract means
    a1 = array1.ravel() - array1.mean()
    a2 = array2.ravel() - array2.mean()
    # Compute normalized correlation efficiently
    numerator = np.dot(a1, a2)
    denominator = np.linalg.norm(a1) * np.linalg.norm(a2)
    return np.divide(numerator, denominator) if denominator != 0 else 0.0


def min_max(data1, global_min=None, global_max=None):
    min_val = np.min(data1) if global_min is None else global_min
    max_val = np.max(data1) if global_max is None else global_max
    if min_val == max_val:
        return data1
    return (data1 - min_val) / (max_val - min_val)


def non_zero_crop(a, b):
    mini = max(np.min(np.where(a[0] != 0)), np.min(np.where(b[0] != 0)))
    maxi = min(np.max(np.where(a[0] != 0)), np.max(np.where(b[0] != 0)))
    return mini, maxi


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


def warp_image_affine(image, shifts):
    mat = np.float32([[1, 0, -shifts[0]], [0, 1, -shifts[1]]])
    rows, cols = image.shape
    warped_image = cv2.warpAffine(
        image,
        mat,
        (cols, rows),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return warped_image


def resource_path(relative_path):
    """Get absolute path to resource for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def download_model(url, local_filename):
    """Download the model if absent"""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        raise ValueError(f"Model url not valid or failed model downloading {e}")
