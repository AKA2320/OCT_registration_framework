from registration_script import start_reg_gui
import multiprocessing

multiprocessing.freeze_support()


def run_registration_process(output_queue, cancel_event, *args):
    """
    This function is the target for the multiprocessing.Process.
    It sets up stdout redirection and calls the main processing function.

    """

    # Unpack arguments for gui_input
    (
        DATA_LOAD_DIR,
        USE_MODEL_LATERAL_TRANSLATION,
        SAVE_DETECTIONS,
        DATA_SAVE_DIR,
        expected_cells,
        expected_surfaces,
        batch_data_flag,
    ) = args

    args_dict = {
        "DATA_LOAD_DIR": DATA_LOAD_DIR,
        "USE_MODEL_LATERAL_TRANSLATION": USE_MODEL_LATERAL_TRANSLATION,
        "DATA_SAVE_DIR": DATA_SAVE_DIR,
        "EXPECTED_CELLS": expected_cells,
        "EXPECTED_SURFACES": expected_surfaces,
        "BATCH_FLAG": batch_data_flag,
        "SAVE_DETECTIONS": SAVE_DETECTIONS,
        "OUTPUT_QUEUE": output_queue,
    }
    try:
        gui_start_process = multiprocessing.Process(
            target=start_reg_gui, kwargs=args_dict
        )
        gui_start_process.start()
        while gui_start_process.is_alive():
            try:
                if cancel_event.is_set():
                    gui_start_process.terminate()
                    break
            except Exception:  # queue.Empty
                continue
        gui_start_process.join()
    except Exception as e:
        # Ensure exceptions in the child process are reported to the GUI
        output_queue.put("\n--- A CRITICAL ERROR OCCURRED IN THE PROCESS ---\n")
        output_queue.put(str(e))
    finally:
        # Signal that the process is done
        if cancel_event.is_set():
            output_queue.put("PROCESS_CANCELLED")
        else:
            output_queue.put("PROCESS_FINISHED")
