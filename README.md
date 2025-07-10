# OCT_registration_framework - feature_gui Branch

## Overview

This is the README for the **feature_gui branch** of the OCT_registration_framework.

This project provides a framework for performing registration of OCT volumes, with a focus on providing a user-friendly Graphical User Interface (GUI) for interaction. It aims to correct for distortions and motion artifacts in OCT images, improving their quality and enabling more accurate analysis. The framework utilizes a combination of image processing techniques, deep learning models, and optimization algorithms.

## Key Features

*   **Feature Detection:** Employs a YOLO model for detecting anatomical features in OCT images.
*   **Motion Correction:** Corrects for motion artifacts in the X, Y, and Z (flattening) directions.
*   **Deep Learning Integration:** Integrates deep learning models, such as a Swin Transformer-based "TransMorph" model, for advanced registration tasks.
*   **Configuration-Driven:** Uses YAML configuration files for easy customization of parameters and file paths.


## Installation

You can install the `OCT_registration_framework` package using pip or uv. It's highly recommended to create a virtual environment before installing.

**Note:** Before using `uv`, ensure that it is installed. Refer to the official [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Using pip (recommended):**

1.  **Clone the feature_gui branch:**
    ```shell
    git clone --branch feature_gui https://github.com/AKA2320/OCT_registration_framework.git
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

**Using uv (faster than pip):**

1.  **Clone the feature_gui branch:**
    ```shell
    git clone --branch feature_gui https://github.com/AKA2320/OCT_registration_framework.git
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

**Using uv with lock file (recommended for reproducible environments):**

1.  **Clone the feature_gui branch:**
    ```shell
    git clone --branch feature_gui https://github.com/AKA2320/OCT_registration_framework.git
    cd OCT_registration_framework
    ```
2.  **Create and activate a virtual environment:**
    ```shell
    python3 -m venv .venv
    source .venv/bin/activate # On Linux/macOS
    # .venv\Scripts\activate # On Windows
    ```
3.  **Install dependencies from the lock file:**
    ```shell
    uv sync
    ```

## Usage

For the feature_gui branch, the primary method of interaction is through the graphical interface (`GUI_scripts/pyside_gui.py`). The GUI scripts are located in the `GUI_scripts/` folder.

1.  **Prepare your OCT data:**
    *   The framework supports `.h5` and `.dcm` OCT data formats.
    *   Ensure your data is accessible.

2.  **Run the GUI application:**
    *   After installing the package and activating your virtual environment, run the GUI script:
        ```shell
        python GUI_scripts/pyside_gui.py
        ```
    *   Use the GUI to select your input data directory, specify the output save directory, and choose whether to use the Model X translation.

3.  **Configuration via `datapaths.yaml` (Optional for GUI):**
    *   While the GUI allows you to specify input/output paths, the `datapaths.yaml` file still contains default paths and configurations for models and other parameters. You may need to update this file for model paths or other settings not exposed in the GUI.

## Configuration

The `datapaths.yaml` file is used to configure the project, particularly for model paths and default settings. It contains the following parameters:

*   `DATA_LOAD_DIR`: Default path to the directory containing the input OCT data (can be overridden by GUI).
*   `DATA_SAVE_DIR`: Default path to the directory where the registered OCT data will be saved (can be overridden by GUI).
*   `MODEL_FEATURE_DETECT_PATH`: Path to the YOLO model for feature detection.
*   `MODEL_X_TRANSLATION_PATH`: Path to the "TransMorph" model for X-motion correction.
*    `EXPECTED_SURFACES`: Expected number of surfaces.
*    `EXPECTED_CELLS`: Expected number of cells.
*   `USE_MODEL_X`: A flag to indicate whether to use MODEL_X_TRANSLATION (can be overridden by the GUI).

## Key Files

*   `GUI_scripts/pyside_gui.py`: The main PySide6 GUI application.
*   `registration_script.py`: The backend script for performing OCT volume registration (typically run via the GUI).
*   `GUI_scripts/gui_reg_util_funcs.py`: Utility functions specific to the GUI registration process.
*   `GUI_scripts/gui_util_funcs.py`: General utility functions for the GUI.
*   `funcs_transmorph.py`: Contains the implementation of the "TransMorph" model.
*   `utils/reg_util_funcs.py`: Provides utility functions for registration, including motion correction, flattening, and feature detection.
*   `datapaths.yaml`: Configuration file for specifying file paths and other parameters.
*   `pyproject.toml`: Project configuration and dependencies.
*   `uv.lock`: Reproduce exact environment using UV.

## Models

The `models/` directory contains the pre-trained models used by the framework:

*   `feature_detect_yolov12best.pt`: YOLO model for feature detection.
*   `model_transmorph_LRNPOSEMBD_Large_x_translation.pt`: "TransMorph" model for X-motion correction.
*   `model_transmorph_x_translation.pt`: Another "TransMorph" model variant.
