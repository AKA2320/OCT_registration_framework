# -*- mode: python ; coding: utf-8 -*-

import sys
import os
import glob
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

# Get the site-packages path dynamically
site_path = [i for i in sys.path if "site-packages" in i][0]

# Construct paths using os.path.join for better cross-platform compatibility
napari_path = os.path.join(site_path, "napari")
vispy_path = os.path.join(site_path, "vispy")

rust_binary_source_path = glob.glob(os.path.join(site_path, 'rust_lib', 'rust_lib*'))[0]

datapaths_yaml_path = '../datapaths.yaml'
models_yolo_path = '../models/feature_detect_yolov12best.pt'
pyside_gui_py_path = '../pyside_gui.py'
icon_path = 'Tankam-Lab-Logo-2.png'

a = Analysis(
    [pyside_gui_py_path],
    pathex=[],
    binaries=[(rust_binary_source_path, '.')],
    datas=[
        (models_yolo_path, 'models/'),
        (datapaths_yaml_path, '.'),
        (napari_path, 'napari'),
        (vispy_path, 'vispy'),
        (icon_path, '.')
    ],
    hiddenimports=[
        'registration_scripts', 'utils', 'pydicom', 'skimage', 'scipy', 'napari',
        'napari.view_layers', 'torch', 'PySide6', 'ultralytics', 'h5py',
        'natsort', 'torchvision', 'tqdm', 'timm', 'ml_collections'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['ipykernel','jupyter_client','IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)


exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OCT_windows_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon = icon_path
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OCT_windows_app',
)