# OCT_registration_framework

## Overview

This project provides a framework for performing registration of Optical Coherence Tomography (OCT) volumes. It aims to correct for distortions and motion artifacts in OCT images, improving their quality and enabling more accurate analysis. The framework utilizes a combination of image processing techniques, deep learning models, and optimization algorithms.

## Key Features

*   **Feature Detection:** Employs a YOLO model for detecting anatomical features in OCT images.
*   **Motion Correction:** Corrects for motion artifacts in the X, Y, and Z (flattening) directions.
*   **Deep Learning Integration:** Integrates deep learning models, such as a Swin Transformer-based "TransMorph" model, for advanced registration tasks.
*   **Configuration-Driven:** Uses YAML configuration files for easy customization of parameters and file paths.

## Installation

You can install the `OCT_registration_framework` package using pip, uv, or directly from the Git repository.

**Using pip (recommended):**

1.  **Clone the repository:**
    ```shell
    git clone <repository_url>
    cd OCT_registration_framework
    ```
2.  **Create and activate a virtual environment:**
    ```shell
    python3 -m venv .venv
    source .venv/bin/activate # On Linux/macOS
    # .venv\Scripts\activate # On Windows
    ```
3.  **Install the package:**
    ```shell
    pip install .
    ```


**Using uv (recommended):**

1.  **Clone the repository:**
    ```shell
    git clone <repository_url>
    cd OCT_registration_framework
    ```
2.  **Create and activate a virtual environment:**
    ```shell
    python3 -m venv .venv
    source .venv/bin/activate # On Linux/macOS
    # .venv\Scripts\activate # On Windows
    ```
3.  **Install the package:**
    ```shell
    uv pip install .
    ```

**Installing directly from Git:**

You can install directly from the Git repository without cloning first. Replace `<repository_url>` with the actual URL.

```shell
pip install git+<repository_url>
# Or with uv:
# uv pip install git+<repository_url>
```


## Usage

1.  **Prepare your OCT data:**
    *   The framework supports `.h5` and `.dcm` OCT data formats.
    *   Place your data in the directory specified by the `DATA_LOAD_DIR` parameter in `datapaths.yaml`.

2.  **Configure the `datapaths.yaml` file:**
    *   Update the paths to the data directory, model files, and other configuration parameters according to your setup.

3.  **Run the registration script:**

    ```shell
    python registration_script.py
    ```
    *   The script will process the OCT data and save the registered volumes in the directory specified by the `DATA_SAVE_DIR` parameter in `datapaths.yaml`.
    *   **Note:** This script relies on the configuration in `datapaths.yaml`. To specify different input and output directories or choose whether to use Model X, use the GUI.

4. **Using the GUI:**
    *   Run the GUI script:
        ```shell
        python test_pyside.py
        ```
    *   Select the input and output directories and choose whether to use Model X.

## Configuration

The `datapaths.yaml` file is used to configure the project. It contains the following parameters:

*   `DATA_LOAD_DIR`: Default path to the directory containing the input OCT data.
*   `DATA_SAVE_DIR`: Default path to the directory where the registered OCT data will be saved.
*   `MODEL_FEATURE_DETECT_PATH`: Path to the YOLO model for feature detection.
*   `MODEL_X_TRANSLATION_PATH`: Path to the "TransMorph" model for X-motion correction.
*    `EXPECTED_SURFACES`: Expected number of surfaces.
*    `EXPECTED_CELLS`: Expected number of cells.
*   `USE_MODEL_X`: A flag to indicate whether to use MODEL_X_TRANSLATION (can be overridden by the GUI)

## Key Files

*   `registration_script.py`: The main script for performing OCT volume registration.
*   `test_pyside.py`: The PySide6 GUI application.
*   `funcs_transmorph.py`: Contains the implementation of the "TransMorph" model.
*   `utils/reg_util_funcs.py`: Provides utility functions for registration, including motion correction, flattening, and feature detection.
*   `datapaths.yaml`: Configuration file for specifying file paths and other parameters.
*   `pyproject.toml`: Project configuration and dependencies.
*   `requirements.txt`: (Deprecated - use pyproject.toml)

## Models

The `models/` directory contains the pre-trained models used by the framework:

*   `feature_detect_yolov12best.pt`: YOLO model for feature detection.
*   `model_transmorph_LRNPOSEMBD_Large_x_translation.pt`: "TransMorph" model for X-motion correction.
*   `model_transmorph_x_translation.pt`: Another "TransMorph" model variant.

