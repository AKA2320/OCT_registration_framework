# main_gui.py

import sys
import os
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
from PySide6.QtGui import QTextCursor # Corrected import for QTextCursor

# --- Import the user's existing functions & napari ---
try:
    from GUI_scripts.gui_util_funcs import GUI_load_h5, GUI_load_dcm
    import napari
except ImportError as e:
    QMessageBox.critical(
        None,
        "Import Error",
        f"A required library is missing: {e}.\n\n"
        "Please ensure 'napari' is installed (`pip install napari[pyside6]`) "
        "and your project is structured correctly:\n"
        "/your_project/\n"
        "├── main_gui.py\n"
        "└── /utils/\n"
        "    ├── __init__.py\n"
        "    └─��� util_funcs.py",
        QMessageBox.StandardButton.Ok
    )
    sys.exit(1)


# --- Main Application Window ---
class PathLoaderApp(QWidget):
    """
    A GUI with multiple tabs for data loading, visualization, and registration.
    """
    # Define signals to update UI from thread
    registration_output_ready = Signal(str)
    registration_error_ready = Signal(str)
    registration_finished = Signal(int)
    registration_process_error = Signal(QProcess.ProcessError)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Loader & Processor")
        self.resize(600, 550) # Increased size for tabs and better layout

        # Path for Load/Visualize tab (file or directory)
        self.selected_load_path = None
        # Path for Register tab (directory only)
        self.selected_register_path = None
        self.registration_thread = None # Use a thread to run the process

        # Main layout for the entire application window
        overall_layout = QVBoxLayout(self)

        # Create a QTabWidget
        self.tab_widget = QTabWidget()
        overall_layout.addWidget(self.tab_widget)

        # --- Tab 1: Data Loading & Visualization ---
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
        self.visualize_checkbox.setToolTip("If checked, the loaded data will be opened in a napari viewer.")
        self.load_layout.addWidget(self.visualize_checkbox)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self.load_layout.addWidget(self.load_btn)

        self.status_label_load = QLabel("Please use 'Browse File/Directory...' to select data.")
        self.status_label_load.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label_load.setStyleSheet("padding: 5px;")
        self.load_layout.addWidget(self.status_label_load)
        self.load_layout.addStretch() # Adds flexible space at the bottom of load tab

        self.tab_widget.addTab(self.load_tab_widget, "Load & Visualize")

        # --- Tab 2: Data Registration ---
        self.register_tab_widget = QWidget()
        self.register_layout = QVBoxLayout(self.register_tab_widget)

        # Connect signals for updating UI from the registration thread
        self.registration_output_ready.connect(self.append_registration_output)
        self.registration_error_ready.connect(self.append_registration_output) # Appending errors to log
        self.registration_finished.connect(self.process_finished)
        self.registration_process_error.connect(self.process_error)
        self.register_layout.addWidget(QLabel("Select Directory for Registration:"))
        self.browse_dir_btn = QPushButton("Browse Directory...")
        self.browse_dir_btn.setToolTip("Select a directory containing data for registration.")
        self.register_layout.addWidget(self.browse_dir_btn)

        self.registration_path_display = QLineEdit("No directory selected for registration")
        self.registration_path_display.setReadOnly(True)
        self.registration_path_display.setStyleSheet("color: #777; font-style: italic;")
        self.register_layout.addWidget(self.registration_path_display)

        # Add Save Directory selection
        self.register_layout.addWidget(QLabel("Select Directory for Saving Results:"))
        self.browse_save_dir_btn = QPushButton("Browse Save Directory...")
        self.browse_save_dir_btn.setToolTip("Select a directory to save the registered data.")
        self.register_layout.addWidget(self.browse_save_dir_btn)

        self.save_path_display = QLineEdit("No directory selected for saving")
        self.save_path_display.setReadOnly(True)
        self.save_path_display.setStyleSheet("color: #777; font-style: italic;")
        self.register_layout.addWidget(self.save_path_display)

        # Add USE_MODEL_X checkbox
        self.use_model_x_checkbox = QCheckBox("USE_MODEL_X")
        self.use_model_x_checkbox.setToolTip("If checked, the registration script will use Model X.")
        self.register_layout.addWidget(self.use_model_x_checkbox)

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
        self.register_layout.addStretch() # Adds flexible space at the bottom of register tab

        self.tab_widget.addTab(self.register_tab_widget, "Register Data")


        # --- Connect signals/slots for both tabs ---
        self.browse_load_btn.clicked.connect(self.select_load_path)
        self.load_btn.clicked.connect(self.process_load_path)
        self.browse_dir_btn.clicked.connect(self.select_registration_directory)
        self.browse_save_dir_btn.clicked.connect(self.select_save_directory) # Connect new save button
        self.register_btn.clicked.connect(self.register_data)
        self.cancel_register_btn.clicked.connect(self.cancel_registration)

    # --- Slot method for Registration Thread signals ---
    # This method updates the UI based on signals from the RegistrationThread

    def append_registration_output(self, text):
         """Appends text to the output log."""
         self.output_log.append(text)
         self.output_log.verticalScrollBar().setValue(self.output_log.verticalScrollBar().maximum())


    def select_load_path(self):
        """
        Opens a single file dialog for the Load & Visualize tab. It then determines if the user intends to select
        an H5 file or a DICOM directory (by selecting a file within it).
        """
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
            QMessageBox.warning(
                self, "Unsupported File", "Please select a .h5, .hdf5, or .dcm file."
            )
            self.load_btn.setEnabled(False)

    def update_load_path(self, path):
        """Updates the UI and stores the selected path for the Load & Visualize tab."""
        self.selected_load_path = path
        self.path_display_load.setText(path)
        self.path_display_load.setStyleSheet("color: #000; font-style: normal;")
        self.status_label_load.setText("Path selected for loading. Ready to load.")
        self.load_btn.setEnabled(True)

    def select_registration_directory(self):
        """
        Opens a directory dialog for the Register Data tab.
        """
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory for Registration",
            ""
        )

        if not dir_path:
            self.register_btn.setEnabled(False)
            self.registration_path_display.setText("No directory selected for registration")
            self.registration_path_display.setStyleSheet("color: #777; font-style: italic;")
            return

        self.selected_register_path = dir_path
        self.registration_path_display.setText(dir_path)
        self.registration_path_display.setStyleSheet("color: #000; font-style: normal;")
        # Enable register button only if both input and save paths are selected
        if self.selected_register_path and hasattr(self, 'selected_save_path') and self.selected_save_path:
             self.register_btn.setEnabled(True)
        else:
             self.register_btn.setEnabled(False)
        self.output_log.clear() # Clear log when a new registration path is selected

    def select_save_directory(self):
        """
        Opens a directory dialog for the Save Data path in the Register Data tab.
        """
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory for Saving Results",
            ""
        )

        if not dir_path:
            # self.register_btn.setEnabled(False) # Don't disable if only save path is missing
            self.save_path_display.setText("No directory selected for saving")
            self.save_path_display.setStyleSheet("color: #777; font-style: italic;")
            # Disable register button if save path becomes empty
            if self.selected_register_path:
                 self.register_btn.setEnabled(False)
            return

        self.selected_save_path = dir_path
        self.save_path_display.setText(dir_path)
        self.save_path_display.setStyleSheet("color: #000; font-style: normal;")
        # Enable register button only if both input and save paths are selected
        if self.selected_register_path and self.selected_save_path:
             self.register_btn.setEnabled(True)
        else:
             self.register_btn.setEnabled(False)


    def process_load_path(self):
        """Validates the stored path and calls the appropriate external load function."""
        path = self.selected_load_path
        if not path:
            QMessageBox.warning(self, "Warning", "No path has been selected for loading.")
            return

        data = None

        if os.path.isdir(path):
            self.status_label_load.setText(f"Loading DICOM directory: {os.path.basename(path)}...")
            print(f"GUI: Calling 'GUI_load_dcm' with directory: {path}")
            try:
                data = GUI_load_dcm(path)
                self.status_label_load.setText(f"Data loaded (Shape: {data.shape}). Ready for visualization or next steps.")
            except Exception as e:
                self.status_label_load.setText("An error occurred during loading.")
                QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{e}")

        elif os.path.isfile(path) and path.lower().endswith(('.h5', '.hdf5')):
            self.status_label_load.setText(f"Loading H5 file: {os.path.basename(path)}...")
            print(f"GUI: Calling 'GUI_load_h5' with file: {path}")
            try:
                data = GUI_load_h5(path)
                self.status_label_load.setText(f"Data loaded (Shape: {data.shape}). Ready for visualization or next steps.")
            except Exception as e:
                self.status_label_load.setText("An error occurred during loading.")
                QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{e}")
        else:
            QMessageBox.critical(self, "Invalid Path", "The selected path is not a valid directory or HDF5 file.")
            return

        if data is not None and self.visualize_checkbox.isChecked():
            self.status_label_load.setText(f"Data loaded (Shape: {data.shape}). Visualizing with napari...")
            print("GUI: Visualizing data with napari...")
            try:
                viewer = napari.view_image(data)
                self.status_label_load.setText(f"Visualization complete. Data Shape: {data.shape}")
            except Exception as e:
                QMessageBox.critical(self, "Napari Error", f"Failed to launch napari viewer:\n{e}")

    def register_data(self):
        """
        Runs an external Python script for data registration, passing the selected path and save path.
        """
        path_to_register = self.selected_register_path
        path_to_save = getattr(self, 'selected_save_path', None) # Get save path, default to None if not set

        if not path_to_register:
            QMessageBox.warning(self, "Warning", "No input directory has been selected for registration.")
            return

        if not path_to_save:
             QMessageBox.warning(self, "Warning", "No output directory has been selected for registration.")
             return

        registration_script = "GUI_scripts.gui_registration_script"
        # if not os.path.exists(registration_script):
        #     QMessageBox.critical(self, "Error", f"Registration script not found: {registration_script}\n")
        #     return

        self.output_log.clear()
        self.status_label_load.setText(f"Registration process initiated in 'Register Data' tab for: {os.path.basename(path_to_register)}...")
        self.output_log.append(f"Starting registration for: {os.path.basename(path_to_register)}...")
        print(f"GUI: Starting registration thread for path: {path_to_register}")

        # Get the state of the USE_MODEL_X checkbox
        use_model_x = self.use_model_x_checkbox.isChecked()
        print(f"GUI: USE_MODEL_X is set to: {use_model_x}")

        self.register_btn.setEnabled(False)
        self.browse_dir_btn.setEnabled(False)
        self.cancel_register_btn.setEnabled(True)

        # Create and start the registration thread, passing paths and checkbox state
        self.registration_thread = RegistrationThread(path_to_register, path_to_save, registration_script, use_model_x)
        self.registration_thread.output_ready.connect(self.append_registration_output)
        self.registration_thread.error_ready.connect(self.append_registration_output) # Append stderr to log
        self.registration_thread.finished.connect(self.process_finished)
        self.registration_thread.process_error_occurred.connect(self.process_error)

        self.registration_thread.start()
        self.output_log.append(f"Registration thread started. Waiting for process PID...")

    def process_finished(self, exit_code):
        """Handles the cleanup and status update when the process (run by thread) finishes."""
        self.status_label_load.setText(f"Registration process finished (exit code: {exit_code}).")
        self.output_log.append(f"Registration process finished with exit code: {exit_code}")

        self.register_btn.setEnabled(True)
        self.browse_dir_btn.setEnabled(True)
        self.cancel_register_btn.setEnabled(False)
        self.registration_thread = None # Clean up the thread reference

    def process_error(self, error_enum):
        """Handles errors encountered by QProcess itself within the thread."""
        error_message = f"QProcess Error: {error_enum.name}"
        self.status_label_load.setText(error_message)
        self.output_log.append(f"{error_message}")
        QMessageBox.critical(self, "QProcess Error", error_message)
        # The thread's finished signal should still fire, triggering process_finished

    def cancel_registration(self):
        """Terminates the ongoing registration process running in the thread."""
        if self.registration_thread and self.registration_thread.isRunning():
            self.status_label_load.setText("Terminating registration process...")
            self.output_log.append("User requested termination. Terminating process...")
            self.registration_thread.terminate_process() # Call method on the thread


# --- Registration Worker Thread ---
class RegistrationThread(QThread):
    """
    A QThread that runs the external registration script using QProcess.
    """
    output_ready = Signal(str)
    error_ready = Signal(str)
    process_error_occurred = Signal(QProcess.ProcessError)

    def __init__(self, directory_path, save_directory_path, script_path, use_model_x):
        super().__init__()
        self.directory_path = directory_path
        self.save_directory_path = save_directory_path # Store the save path
        self.script_path = script_path
        self.use_model_x = use_model_x # Store the state of the checkbox
        self.process = None # QProcess instance

    def run(self):
        """Main method of the thread, executed when start() is called."""
        self.process = QProcess(self)

        # Connect process signals to thread signals
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.errorOccurred.connect(self.process_error)

        # Start the process
        try:
            # Build command arguments, including --use-model-x and --save-dirname
            command_args = [sys.executable, "-m", self.script_path, "--dirname", self.directory_path, "--save-dirname", self.save_directory_path, "--disable-tqdm", False]
            if self.use_model_x:
                command_args.extend(["--use-model-x", True])
            else:
                command_args.extend(["--use-model-x", False])

            print(f"GUI Thread: Starting process with command: {command_args}")

            # self.process.start(sys.executable, command_args) # Modified below
            self.process.start(command_args[0], command_args[1:]) # Start process correctly with args list

            # The line below seems to be a duplicate start call and is likely the cause of issues.
            # self.process.start(sys.executable, command_args)
            # Wait for the process to finish (this is the blocking part for the thread, not the GUI)
            self.process.waitForFinished(-1) # -1 means wait indefinitely

        except Exception as e:
            # This catches errors in starting the process itself
            self.error_ready.emit(f"Error starting registration process: {e}")
            self.process_error_occurred.emit(QProcess.ProcessError.FailedToStart) # Emit a specific error type

    def handle_stdout(self):
        """Reads stdout from QProcess and emits a signal."""
        output = self.process.readAllStandardOutput().data().decode().strip()
        if output:
            self.output_ready.emit(output)

    def handle_stderr(self):
        """Reads stderr from QProcess and emits a signal."""
        error = self.process.readAllStandardError().data().decode().strip()
        if error:
            self.error_ready.emit(error)

    def process_error(self, error_enum):
        """Handles errors from QProcess itself and emits a signal."""
        self.process_error_occurred.emit(error_enum)

    def terminate_process(self):
        """Terminates the ongoing registration process running in the thread."""
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            self.wait(5000) # Wait up to 5 seconds for process to terminate gracefully
            if self.process and self.process.state() == QProcess.ProcessState.Running:
                 self.process.kill() # If it didn't terminate, kill it


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PathLoaderApp()
    window.show()
    sys.exit(app.exec())

