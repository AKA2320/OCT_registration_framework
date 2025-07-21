# main_gui.py

import sys
import os
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QMessageBox,
    QLineEdit,
    QCheckBox,
    QTextEdit,
    QTabWidget,
)
from PySide6.QtCore import Qt, QProcess, QThread, Signal
import napari
import multiprocessing
from utils.util_funcs import GUI_load_h5, GUI_load_dcm
from GUI_scripts.gui_registration_script import gui_input

# Main Application Window
class PathLoaderApp(QWidget):
    registration_output_ready = Signal(str)
    registration_error_ready = Signal(str)
    registration_finished = Signal(int)
    registration_process_error = Signal(QProcess.ProcessError)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Loader & Processor")
        self.resize(600, 550)

        self.selected_load_path = None
        self.selected_register_path = None
        self.selected_save_path = None
        self.registration_thread = None

        overall_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        overall_layout.addWidget(self.tab_widget)

        # Tab 1: Data Loading & Visualization
        self.load_tab_widget = QWidget()
        self.load_layout = QVBoxLayout(self.load_tab_widget)

        self.browse_load_btn = QPushButton("Browse File/Directory...")
        self.browse_load_btn.setToolTip("Select an HDF5 file or any file within a DICOM directory.")
        self.load_layout.addWidget(self.browse_load_btn)

        self.path_display_load = QLineEdit("No file/directory selected for loading")
        self.path_display_load.setReadOnly(True)
        self.path_display_load.setStyleSheet("color: #777; font-style: italic;")
        self.load_layout.addWidget(self.path_display_load)

        self.visualize_checkbox = QCheckBox("Visualize with napari")
        self.visualize_checkbox.setChecked(True) 
        self.load_layout.addWidget(self.visualize_checkbox)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self.load_layout.addWidget(self.load_btn)

        self.status_label_load = QLabel("Please use 'Browse File/Directory...' to select data.")
        self.status_label_load.setStyleSheet("padding: 5px;")
        self.load_layout.addWidget(self.status_label_load)
        self.load_layout.addStretch()
        self.tab_widget.addTab(self.load_tab_widget, "Load & Visualize")

        # Tab 2: Data Registration
        self.register_tab_widget = QWidget()
        self.register_layout = QVBoxLayout(self.register_tab_widget)

        self.registration_output_ready.connect(self.append_registration_output)
        self.registration_error_ready.connect(self.append_registration_output)
        self.registration_finished.connect(self.process_finished)
        self.registration_process_error.connect(self.process_error)

        self.register_layout.addWidget(QLabel("Select Directory for Registration:"))
        self.browse_dir_btn = QPushButton("Browse Directory...")
        self.register_layout.addWidget(self.browse_dir_btn)

        self.registration_path_display = QLineEdit("No directory selected for registration")
        self.registration_path_display.setReadOnly(True)
        self.registration_path_display.setStyleSheet("color: #777; font-style: italic;")
        self.register_layout.addWidget(self.registration_path_display)

        # Add Save Directory selection
        self.register_layout.addWidget(QLabel("Select Directory for Saving Results (Default: 'output/'):"))
        self.browse_save_dir_btn = QPushButton("Browse Save Directory...")
        self.register_layout.addWidget(self.browse_save_dir_btn)

        self.save_path_display = QLineEdit("output/") 
        self.selected_save_path = "output/"
        self.save_path_display.setReadOnly(True)
        self.save_path_display.setStyleSheet("color: #777; font-style: italic;")
        self.register_layout.addWidget(self.save_path_display)

        # New: EXPECTED_CELLS input
        self.register_layout.addWidget(QLabel("Expected Cells (int, default: 2):"))
        self.expected_cells_input = QLineEdit("2") # Default value 2
        self.expected_cells_input.setValidator(QIntValidator()) # Only allow integer input
        self.register_layout.addWidget(self.expected_cells_input)

        # New: EXPECTED_SURFACES input
        self.register_layout.addWidget(QLabel("Expected Surfaces (int, default: 2):"))
        self.expected_surfaces_input = QLineEdit("2") # Default value 2
        self.expected_surfaces_input.setValidator(QIntValidator()) # Only allow integer input
        self.register_layout.addWidget(self.expected_surfaces_input)

        self.use_model_x_checkbox = QCheckBox("USE_MODEL_X")
        self.use_model_x_checkbox.setChecked(True)  # Set default to TRUE
        self.register_layout.addWidget(self.use_model_x_checkbox)

        self.disable_tqdm_checkbox = QCheckBox("DISABLE_TQDM")
        self.disable_tqdm_checkbox.setChecked(True)  # Set default to TRUE
        self.register_layout.addWidget(self.disable_tqdm_checkbox)

        self.register_btn = QPushButton("Register Data")
        self.register_btn.setToolTip("Runs an external script to register the selected data.")
        self.register_btn.setEnabled(False)
        self.register_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self.register_layout.addWidget(self.register_btn)

        self.cancel_register_btn = QPushButton("Cancel Registration")
        self.cancel_register_btn.setToolTip("Terminates the ongoing registration script.")
        self.cancel_register_btn.setEnabled(False)
        self.cancel_register_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self.register_layout.addWidget(self.cancel_register_btn)

        self.register_layout.addWidget(QLabel("--- Script Output ---"))
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setPlaceholderText("Script output will appear here...")
        self.output_log.setStyleSheet("background-color: black; color: white; border: 1px solid #ccc; padding: 5px;")
        self.register_layout.addWidget(self.output_log)
        self.register_layout.addStretch()

        self.tab_widget.addTab(self.register_tab_widget, "Register Data")

        self.browse_load_btn.clicked.connect(self.select_load_path)
        self.load_btn.clicked.connect(self.process_load_path)
        self.browse_dir_btn.clicked.connect(self.select_registration_directory)
        self.browse_save_dir_btn.clicked.connect(self.select_save_directory)
        self.register_btn.clicked.connect(self.register_data)
        self.cancel_register_btn.clicked.connect(self.cancel_registration)

    def append_registration_output(self, text):
        """Appends text to the output log."""
        self.output_log.append(text)
        self.output_log.verticalScrollBar().setValue(self.output_log.verticalScrollBar().maximum())

    def select_load_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File or a File in a DICOM Directory",
            "",
            "Supported Files (*.h5 *.hdf5 *.dcm);;All Files (*)"
        )
        if not file_path:
            return

        if file_path.lower().endswith(('.h5', '.hdf5')):
            self.update_load_path(file_path)
        elif file_path.lower().endswith('.dcm'):
            dir_path = os.path.dirname(file_path)
            self.update_load_path(dir_path)
        else:
            QMessageBox.warning(self, "Unsupported File", "Please select a .h5, .hdf5, or .dcm file.")
            self.load_btn.setEnabled(False)

    def update_load_path(self, path):
        self.selected_load_path = path
        self.path_display_load.setText(path)
        self.path_display_load.setStyleSheet("color: #fff; font-style: normal;")
        self.status_label_load.setText("Path selected for loading. Ready to load.")
        self.load_btn.setEnabled(True)

    def select_registration_directory(self):
        dir_path, _ = QFileDialog.getOpenFileName(self, "Select Directory for Registration")
        if not dir_path:
            self.register_btn.setEnabled(False)
            self.registration_path_display.setText("No directory selected for registration")
            self.registration_path_display.setStyleSheet("color: #777; font-style: italic;")
            return
        
        # if dir_path.lower().endswith(('.h5', '.hdf5')):
        #     # self.update_load_path(dir_path)
        if dir_path.lower().endswith('.dcm'):
            dir_path = os.path.dirname(dir_path)
            # self.update_load_path(dir_path)
        elif not dir_path.lower().endswith(('.h5', '.hdf5')):
            QMessageBox.warning(self, "Unsupported File", "Please select a .h5, .hdf5, or .dcm file.")
            self.register_btn.setEnabled(False)
            return

        self.selected_register_path = dir_path
        self.registration_path_display.setText(dir_path)
        self.registration_path_display.setStyleSheet("color: #fff; font-style: normal;")
        if self.selected_register_path and self.selected_save_path:
            self.register_btn.setEnabled(True)
        else:
            self.register_btn.setEnabled(False)
        self.output_log.clear()

    def select_save_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Saving Results")
        if not dir_path:
            dir_path = "output/"
            # self.save_path_display.setText("No directory selected for saving")
            # self.save_path_display.setStyleSheet("color: #777; font-style: italic;")
            # if self.selected_register_path:
            #     self.register_btn.setEnabled(False)
            # return

        self.selected_save_path = dir_path
        self.save_path_display.setText(dir_path)
        self.save_path_display.setStyleSheet("color: #fff; font-style: normal;")
        if self.selected_register_path and self.selected_save_path:
            self.register_btn.setEnabled(True)
        else:
            self.register_btn.setEnabled(False)

    def process_load_path(self):
        path = self.selected_load_path
        if not path:
            QMessageBox.warning(self, "Warning", "No path has been selected for loading.")
            return
        data = None

        if os.path.isdir(path):
            self.status_label_load.setText(f"Loading DICOM directory: {os.path.basename(path)}...")
            try:
                data = GUI_load_dcm(path)
                self.status_label_load.setText(f"Data loaded (Shape: {data.shape}). Ready for visualization.")
            except Exception as e:
                self.status_label_load.setText("An error occurred during loading.")
                QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{e}")

        elif os.path.isfile(path) and path.lower().endswith(('.h5', '.hdf5')):
            self.status_label_load.setText(f"Loading H5 file: {os.path.basename(path)}...")
            try:
                data = GUI_load_h5(path)
                self.status_label_load.setText(f"Data loaded (Shape: {data.shape}). Ready for visualization.")
            except Exception as e:
                self.status_label_load.setText("An error occurred during loading.")
                QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{e}")
        else:
            QMessageBox.critical(self, "Invalid Path", "The selected path is not a valid directory or HDF5 file.")
            return

        if data is not None and self.visualize_checkbox.isChecked():
            self.status_label_load.setText(f"Data loaded (Shape: {data.shape}). Visualizing with napari...")
            try:
                viewer = napari.view_image(data)
                self.status_label_load.setText(f"Visualization complete. Data Shape: {data.shape}")
            except Exception as e:
                QMessageBox.critical(self, "Napari Error", f"Failed to launch napari viewer:\n{e}")

    def register_data(self):
        path_to_register = self.selected_register_path
        path_to_save = self.selected_save_path

        if not path_to_register:
            QMessageBox.warning(self, "Warning", "No input directory has been selected for registration.")
            return

        if not path_to_save:
            QMessageBox.warning(self, "Warning", "No output directory has been selected for registration.")
            return
        
        try:
            expected_cells = int(self.expected_cells_input.text())
            expected_surfaces = int(self.expected_surfaces_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Expected Cells and Expected Surfaces must be valid integers.")
            return

        # registration_script = "GUI_scripts.gui_registration_script"
        registration_script = ""

        self.output_log.clear()
        self.status_label_load.setText(f"Registration process initiated for: {os.path.basename(path_to_register)}...")
        self.output_log.append(f"Starting registration for: {os.path.basename(path_to_register)}...")

        use_model_x = self.use_model_x_checkbox.isChecked()
        disable_tqdm = self.disable_tqdm_checkbox.isChecked()

        self.register_btn.setEnabled(False)
        self.browse_dir_btn.setEnabled(False)
        self.cancel_register_btn.setEnabled(True)

        self.registration_thread = RegistrationThread(path_to_register, path_to_save, registration_script
                                                      ,use_model_x, disable_tqdm, expected_cells, expected_surfaces)
        self.registration_thread.output_ready.connect(self.append_registration_output)
        self.registration_thread.error_ready.connect(self.append_registration_output)
        self.registration_thread.finished.connect(self.process_finished)
        self.registration_thread.process_error_occurred.connect(self.process_error)

        self.registration_thread.start()

    def process_finished(self):
        self.status_label_load.setText("Registration process finished.")
        self.output_log.append("Registration process finished.")
        self.register_btn.setEnabled(True)
        self.browse_dir_btn.setEnabled(True)
        self.cancel_register_btn.setEnabled(False)
        self.registration_thread = None

    def process_error(self, error_enum):
        error_message = f"QProcess Error: {error_enum.name}"
        self.status_label_load.setText(error_message)
        self.output_log.append(error_message)
        QMessageBox.critical(self, "QProcess Error", error_message)

    def cancel_registration(self):
        if self.registration_thread and self.registration_thread.isRunning():
            self.status_label_load.setText("Cancelling registration process...")
            self.output_log.append("User requested cancellation. Cancelling process...")
            self.registration_thread.terminate_process()
            self.register_btn.setEnabled(True)
            self.browse_dir_btn.setEnabled(True)
            self.cancel_register_btn.setEnabled(False)

# Registration Worker Thread
class RegistrationThread(QThread):
    output_ready = Signal(str)
    error_ready = Signal(str)
    process_error_occurred = Signal(QProcess.ProcessError)
    registration_finished = Signal(int)
    registration_cancelled = Signal()

    def __init__(self, directory_path, save_directory_path, script_path, use_model_x, disable_tqdm, expected_cells, expected_surfaces):
        super().__init__()
        self.directory_path = directory_path
        self.save_directory_path = save_directory_path
        # self.script_path = script_path
        self.use_model_x = use_model_x
        self.disable_tqdm = disable_tqdm
        self.expected_cells = expected_cells
        self.expected_surfaces = expected_surfaces
        self.process = None
        self._cancellation_flag = False

    def run(self):
        try:
            gui_input(
                dirname=self.directory_path,
                use_model_x=self.use_model_x,
                disable_tqdm=self.disable_tqdm,
                save_dirname=self.save_directory_path,
                expected_cells=self.expected_cells,
                expected_surfaces=self.expected_surfaces,
                cancellation_flag=lambda: self._cancellation_flag
            )
            if self._cancellation_flag:
                self.registration_cancelled.emit()
            else:
                self.registration_finished.emit(0)
        except Exception as e:
            self.error_ready.emit(f"Error in registration process: {e}")
            self.process_error_occurred.emit(QProcess.ProcessError.FailedToStart)

    def handle_stdout(self):
        output = self.process.readAllStandardOutput().data().decode(errors='ignore').strip()
        if output:
            self.output_ready.emit(output)

    def handle_stderr(self):
        error = self.process.readAllStandardError().data().decode(errors='ignore').strip()
        if error:
            self.error_ready.emit(error)

    def process_error(self, error_enum):
        self.process_error_occurred.emit(error_enum)

    def terminate_process(self):
        self._cancellation_flag = True
        self.registration_cancelled.emit()

    def handle_process_finished(self):
        self.registration_finished.emit(self.process.exitCode())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = PathLoaderApp()
    window.show()
    sys.exit(app.exec())