import multiprocessing
multiprocessing.freeze_support()
import sys
import os
# import logging
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
from PySide6.QtCore import QThread, Signal
from utils.load_data_funcs import GUI_load_h5, GUI_load_dcm, load_napari_viewer
from registration_scripts.gui_reg_process_wrapper import run_registration_process


# =============================================================================
# Worker Threads
# =============================================================================
class LoadThread(QThread):
    """Worker thread to load data without blocking the GUI."""
    data_ready = Signal(object)
    error_occurred = Signal(str)
    update_status = Signal(str)

    def __init__(self, path_to_load):
        super().__init__()
        self.path = path_to_load

    def run(self):
        """Loads data based on the path type."""
        try:
            if os.path.isdir(self.path):
                self.update_status.emit(f"Loading DICOM directory: {os.path.basename(self.path)}...")
                data = GUI_load_dcm(self.path)
            elif os.path.isfile(self.path) and self.path.lower().endswith(('.h5', '.hdf5')):
                self.update_status.emit(f"Loading H5 file: {os.path.basename(self.path)}...")
                data = GUI_load_h5(self.path)
            else:
                self.error_occurred.emit("The selected path is not a valid directory or HDF5 file.")
                return
            if data is not None:
                self.update_status.emit(f"Data loaded successfully (Shape: {data.shape}).")
                self.data_ready.emit(data)
            else:
                self.error_occurred.emit("Data loading returned None.")
        except Exception as e:
            self.error_occurred.emit(f"An error occurred during loading:\n{e}")

class RegistrationThread(QThread):
    """
    Manages a separate multiprocessing.Process to run the heavy computation,
    ensuring the GUI remains completely isolated and responsive.
    """
    output_ready = Signal(str)
    finished = Signal()
    registration_cancelled = Signal()

    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.process = None
        self.cancel_event = None

    def run(self):
        """
        Starts the external process and monitors its output queue.
        """
        try:
            output_queue = multiprocessing.Queue()
            self.cancel_event = multiprocessing.Event()
            
            self.process = multiprocessing.Process(
                target = run_registration_process,
                args = (output_queue, self.cancel_event) + self.args
            )
            self.process.start()

            # Monitor the queue for output from the process
            while self.process.is_alive():
                try:
                    # Wait for output with a timeout to keep the loop responsive
                    output = output_queue.get(timeout=0.1)
                    if output == "PROCESS_FINISHED":
                        break
                    self.output_ready.emit(output)
                except Exception: # queue.Empty
                    continue
            
            # Process remaining messages after the main loop exits
            while not output_queue.empty():
                output = output_queue.get()
                if output != "PROCESS_FINISHED":
                    self.output_ready.emit(output)

        finally:
            if self.process:
                self.process.join() # Wait for the process to terminate
            
            if self.cancel_event and self.cancel_event.is_set():
                self.registration_cancelled.emit()
            else:
                self.finished.emit()

    def terminate_process(self):
        if self.cancel_event:
            self.output_ready.emit("Cancellation signal sent. Waiting for process to acknowledge...\n")
            self.cancel_event.set()

# =============================================================================
# GUI Tab Widgets
# =============================================================================
class LoadTab(QWidget):
    """Encapsulated widget for the 'Load & Visualize' tab."""
    update_status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_load_path = None
        self.load_thread = None
        layout = QVBoxLayout(self)
        self.browse_load_btn = QPushButton("Browse File/Directory...")
        self.browse_load_btn.setToolTip("Select an HDF5 file or any file within a DICOM directory.")
        layout.addWidget(self.browse_load_btn)
        self.path_display_load = QLineEdit("No file/directory selected for loading")
        self.path_display_load.setReadOnly(True)
        self.path_display_load.setStyleSheet("background-color: white; color: black; font-style: italic;")
        layout.addWidget(self.path_display_load)
        self.visualize_checkbox = QCheckBox("Visualize with napari when loaded")
        self.visualize_checkbox.setChecked(True)
        layout.addWidget(self.visualize_checkbox)
        self.load_btn = QPushButton("Load Data")
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.load_btn)
        layout.addStretch()
        self.browse_load_btn.clicked.connect(self.select_load_path)
        self.load_btn.clicked.connect(self.start_loading_data)

    def select_load_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select HDF5 File or a File in a DICOM Directory", "", "Supported Files (*.h5 *.hdf5 *.dcm);;All Files (*)")
        if not file_path: return
        path_to_use = os.path.dirname(file_path) if file_path.lower().endswith('.dcm') else file_path
        if not (path_to_use.lower().endswith(('.h5', '.hdf5')) or os.path.isdir(path_to_use)):
            QMessageBox.warning(self, "Unsupported File", "Please select a .h5, .hdf5, or .dcm file.")
            self.load_btn.setEnabled(False)
            return
        self.selected_load_path = path_to_use
        self.path_display_load.setText(path_to_use)
        self.path_display_load.setStyleSheet("background-color: white; color: black; font-style: normal;")
        self.update_status.emit("Path selected. Ready to load.")
        self.load_btn.setEnabled(True)

    def start_loading_data(self):
        if not self.selected_load_path:
            QMessageBox.warning(self, "Warning", "No path has been selected for loading.")
            return
        self.load_btn.setEnabled(False)
        self.browse_load_btn.setEnabled(False)
        self.update_status.emit("Initializing data loading...")
        self.load_thread = LoadThread(self.selected_load_path)
        self.load_thread.data_ready.connect(self.on_data_loaded)
        self.load_thread.error_occurred.connect(self.on_load_error)
        self.load_thread.update_status.connect(self.update_status.emit)
        self.load_thread.finished.connect(self.on_load_finished)
        self.load_thread.start()

    def on_data_loaded(self, data):
        if self.visualize_checkbox.isChecked():
            self.update_status.emit("Visualizing with napari...")
            load_napari_viewer(data)
            # napari.view_image(data)
            self.update_status.emit("Visualization complete.")
        else:
            QMessageBox.information(self, "Load Complete", f"Data loaded successfully with shape: {data.shape}")


    def on_load_error(self, message):
        self.update_status.emit("An error occurred during loading.")
        QMessageBox.critical(self, "Processing Error", message)

    def on_load_finished(self):
        self.load_btn.setEnabled(True)
        self.browse_load_btn.setEnabled(True)
        self.load_thread = None

class RegistrationTab(QWidget):
    """Encapsulated widget for single or batch registration."""
    update_status = Signal(str)

    def __init__(self, mode='single', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.selected_register_path = None
        self.selected_save_path = "output/"
        self.registration_thread = None
        layout = QVBoxLayout(self)
        reg_label_text = f"Select {'Directory for Batch' if self.mode == 'batch' else 'Data for'} Processing:"
        browse_tooltip = f"Select a {'directory containing .h5 files' if self.mode == 'batch' else '.h5, .hdf5, or .dcm file'}."
        layout.addWidget(QLabel(reg_label_text))
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setToolTip(browse_tooltip)
        layout.addWidget(self.browse_btn)
        self.path_display = QLineEdit(f"No {self.mode} data selected")
        self.path_display.setReadOnly(True)
        self.path_display.setStyleSheet("background-color: white; color: black; font-style: italic;")
        layout.addWidget(self.path_display)
        layout.addWidget(QLabel("Select Directory for Saving Results (Default: 'output/'):"))
        self.browse_save_btn = QPushButton("Browse Save Directory...")
        layout.addWidget(self.browse_save_btn)
        self.save_path_display = QLineEdit(self.selected_save_path)
        self.save_path_display.setReadOnly(True)
        self.save_path_display.setStyleSheet("background-color: white; color: black; font-style: normal;")
        layout.addWidget(self.save_path_display)
        layout.addWidget(QLabel("Expected Cells (int, default: 2):"))
        self.expected_cells_input = QLineEdit("2")
        self.expected_cells_input.setValidator(QIntValidator())
        self.expected_cells_input.setStyleSheet("background-color: white; color: black;")
        layout.addWidget(self.expected_cells_input)
        layout.addWidget(QLabel("Expected Surfaces (int, default: 2):"))
        self.expected_surfaces_input = QLineEdit("2")
        self.expected_surfaces_input.setValidator(QIntValidator())
        self.expected_surfaces_input.setStyleSheet("background-color: white; color: black;")
        layout.addWidget(self.expected_surfaces_input)
        self.USE_MODEL_LATERAL_TRANSLATION_checkbox = QCheckBox("Use ML Model for Lateral(X) Motion Correction")
        self.USE_MODEL_LATERAL_TRANSLATION_checkbox.setChecked(True)
        layout.addWidget(self.USE_MODEL_LATERAL_TRANSLATION_checkbox)
        self.save_detections_checkbox = QCheckBox("Save Feature Detections")
        self.save_detections_checkbox.setChecked(True)
        layout.addWidget(self.save_detections_checkbox)
        self.register_btn = QPushButton("Process Data")
        self.register_btn.setEnabled(False)
        self.register_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.register_btn)
        self.cancel_btn = QPushButton("Cancel Registration")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.cancel_btn)
        layout.addWidget(QLabel("--- Script Output ---"))
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setPlaceholderText("Script output will appear here...")
        self.output_log.setStyleSheet("background-color: white; color: black; border: 1px solid #ccc; padding: 5px;")
        layout.addWidget(self.output_log)
        layout.addStretch()
        self.browse_btn.clicked.connect(self.select_registration_path)
        self.browse_save_btn.clicked.connect(self.select_save_directory)
        self.register_btn.clicked.connect(self.start_registration)
        self.cancel_btn.clicked.connect(self.cancel_registration)

    def _update_register_button_state(self):
        self.register_btn.setEnabled(bool(self.selected_register_path and self.selected_save_path))

    def select_registration_path(self):
        path = ""
        if self.mode == 'single':
            path, _ = QFileDialog.getOpenFileName(self, "Select Data for Registration", "", "Supported Files (*.h5 *.hdf5 *.dcm);;All Files (*)")
            if path and path.lower().endswith('.dcm'): path = os.path.dirname(path)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Batch Directory")
        if path:
            self.selected_register_path = path
            self.path_display.setText(path)
            self.path_display.setStyleSheet("background-color: white; color: black; font-style: normal;")
            self.output_log.clear()
        else:
            self.selected_register_path = None
            self.path_display.setText(f"No {self.mode} data selected")
            self.path_display.setStyleSheet("background-color: white; color: black; font-style: italic;")
        self._update_register_button_state()

    def select_save_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory for Saving Results")
        self.selected_save_path = path if path else "output/"
        self.save_path_display.setText(self.selected_save_path)
        self._update_register_button_state()

    def start_registration(self):
        try:
            expected_cells = int(self.expected_cells_input.text())
            expected_surfaces = int(self.expected_surfaces_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Expected Cells and Surfaces must be valid integers.")
            return

        self.output_log.clear()
        self.append_output(f"Starting {self.mode} registration for: {os.path.basename(self.selected_register_path)}...\n")
        self.update_status.emit(f"Registration started for: {os.path.basename(self.selected_register_path)}...")
        self._set_ui_running_state(True)
        
        args = (
            self.selected_register_path,
            self.USE_MODEL_LATERAL_TRANSLATION_checkbox.isChecked(),
            self.save_detections_checkbox.isChecked(),
            self.selected_save_path,
            expected_cells,
            expected_surfaces,
            (self.mode == 'batch')
        )
        self.registration_thread = RegistrationThread(*args)
        self.registration_thread.output_ready.connect(self.append_output)
        self.registration_thread.finished.connect(self.process_finished)
        self.registration_thread.registration_cancelled.connect(self.process_cancelled)
        self.registration_thread.start()

    def cancel_registration(self):
        if self.registration_thread and self.registration_thread.isRunning():
            self.update_status.emit("Cancelling registration process...")
            self.registration_thread.terminate_process()

    def process_finished(self):
        self.update_status.emit("Registration process finished.")
        self.append_output("\nRegistration process finished successfully.")
        self._set_ui_running_state(False)
        self.registration_thread = None

    def process_cancelled(self):
        self.update_status.emit("Registration process cancelled by user.")
        self.append_output("\nProcess successfully cancelled.")
        self._set_ui_running_state(False)
        self.registration_thread = None

    def _set_ui_running_state(self, is_running):
        self.register_btn.setEnabled(not is_running)
        self.browse_btn.setEnabled(not is_running)
        self.browse_save_btn.setEnabled(not is_running)
        self.cancel_btn.setEnabled(is_running)

    def append_output(self, text):
        cursor = self.output_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.output_log.setTextCursor(cursor)
        self.output_log.insertPlainText(text)
        self.output_log.ensureCursorVisible()

# =============================================================================
# Main Application Window
# =============================================================================
class PathLoaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Loader & Processor")
        self.resize(650, 700)
        overall_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        overall_layout.addWidget(self.tab_widget)
        self.load_tab = LoadTab()
        self.load_tab.update_status.connect(self.update_status_bar)
        self.tab_widget.addTab(self.load_tab, "Load & Visualize")
        self.single_register_tab = RegistrationTab(mode='single')
        self.single_register_tab.update_status.connect(self.update_status_bar)
        self.tab_widget.addTab(self.single_register_tab, "Process Data")
        self.batch_register_tab = RegistrationTab(mode='batch')
        self.batch_register_tab.update_status.connect(self.update_status_bar)
        self.tab_widget.addTab(self.batch_register_tab, "Batch Process Data")
        self.status_bar = QLabel("Welcome! Please select a file or directory to begin.")
        self.status_bar.setStyleSheet("QLabel { background-color: white; border: 1px solid #ccc; padding: 5px; font-weight: bold; color:black; }")
        overall_layout.addWidget(self.status_bar)

    def update_status_bar(self, message):
        self.status_bar.setText(message)
        QApplication.processEvents()

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PathLoaderApp()
    window.show()
    sys.exit(app.exec())
