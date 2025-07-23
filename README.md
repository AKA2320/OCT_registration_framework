# OCT Registration Framework

## Overview

This project provides a comprehensive framework for performing registration of Optical Coherence Tomography (OCT) volumes. The framework focuses on providing both a user-friendly Graphical User Interface (GUI) and command-line tools for batch processing. It aims to correct for distortions and motion artifacts in OCT images, improving their quality and enabling more accurate analysis through advanced image processing techniques, deep learning models, and optimization algorithms.

## Key Features

*   **Feature Detection:** Employs state-of-the-art YOLO models for detecting anatomical features and structures in OCT images
*   **Multi-dimensional Motion Correction:** Corrects for motion artifacts in X, Y, and Z (flattening) directions
*   **Deep Learning Integration:** Utilizes Swin Transformer-based "TransMorph" models for advanced registration tasks
*   **Flexible Configuration:** Uses YAML configuration files for easy customization of parameters and file paths
*   **Dual Interface:** Provides both GUI (PySide6) and command-line interfaces for different use cases
*   **Multi-format Support:** Supports `.h5` and `.dcm` OCT data formats
*   **Batch Processing:** Includes multiprocessing capabilities for handling large datasets efficiently

## Installation

**Prerequisites:** This project requires Python 3.12. Please ensure you have Python 3.12 installed before proceeding.

### Quick Setup

1.  **Clone the repository:**
    ```shell
    git clone https://github.com/AKA2320/OCT_registration_framework.git
    cd OCT_registration_framework
    ```

2.  **Create and activate a virtual environment:**
    ```shell
    python3.12 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate   # On Windows
    ```

3.  **Install the package:**
    
    **Option A: Using pip (standard)**
    ```shell
    pip install .
    ```
    
    **Option B: Using uv (faster, recommended)**
    ```shell
    pip install uv
    uv pip install .
    ```
    * Before using `uv`, ensure that it is installed. Refer to the official [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

    **Option C: Using uv with lock file (most reproducible)**
    ```shell
    uv sync
    ```

## Usage

The framework can be used through multiple interfaces depending on your needs:

### Using the GUI (Recommended for Interactive Use)

1.  **Prepare your OCT data:**
    *   Ensure your `.h5` or `.dcm` files are organized in accessible directories

2.  **Launch the GUI:**
    ```shell
    python pyside_gui.py
    ```

3.  **Configure through the interface:**
    *   Select input data directory
    *   Specify output save directory
    *   Choose model parameters and processing options
    *   Monitor progress through the built-in interface

### Using Command-Line Scripts

#### Standard Registration Script
1. **Configure datapaths.yaml:**
   Edit `datapaths.yaml` to specify:
   - Input data directory (`DATA_LOAD_DIR`)
   - Output save directory (`DATA_SAVE_DIR`)
   - Model paths for feature detection and translation
   - Processing parameters (`USE_MODEL_X`, `EXPECTED_SURFACES`, `EXPECTED_CELLS`)

2. **Run the registration:**
   ```shell
   python registration_script.py
   ```

#### Multiprocessing Registration (for large datasets)
```shell
python registration_endo_multiproc.py
```

## Configuration

The framework uses `datapaths.yaml` for configuration:

```yaml
PATHS:
  DATA_LOAD_DIR: '/path/to/your/oct/data'
  DATA_SAVE_DIR: 'output_directory/'
  MODEL_FEATURE_DETECT_PATH: 'models/feature_detect_yolov12best.pt'
  MODEL_X_TRANSLATION_PATH: 'models/model_transmorph_LRNPOSEMBD_Large_x_translation.pt'
  USE_MODEL_X: True
  EXPECTED_SURFACES: 3
  EXPECTED_CELLS: 3
```

## Core Components

### Main Scripts
- **`pyside_gui.py`**: PySide6-based GUI application providing interactive registration workflow
- **`registration_script.py`**: Core registration backend for command-line usage
- **`registration_endo_multiproc.py`**: Multiprocessing-enabled registration for batch processing (Under Development)

### Key Modules
- **`utils/reg_util_funcs.py`**: Core registration utilities including motion correction, flattening, and feature detection
- **`utils/util_funcs.py`**: General-purpose utility functions for data handling and processing
- **`GUI_scripts/gui_registration_script.py`**: GUI-specific registration workflow management
- **`funcs_transmorph.py`**: TransMorph model implementation and integration
- **`config_transmorph.py`**: TransMorph model configuration parameters

### Models
The `models/` directory contains pre-trained models:

- **`feature_detect_yolov12best.pt`**: YOLO-based model for anatomical feature detection in OCT images
- **`model_transmorph_LRNPOSEMBD_Large_x_translation.pt`**: Advanced TransMorph model for X-axis motion correction using Swin Transformer architecture

## Dependencies

Key dependencies (see `pyproject.toml` for complete list):
- **Deep Learning**: PyTorch
- **Image Processing**: scikit-image, OpenCV
- **GUI**: PySide6, Napari (for visualization)
- **Data Handling**: h5py, pydicom, numpy, dask

