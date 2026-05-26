import os
import sys
from importlib.util import find_spec
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
_ADDED_DLL_DIRECTORIES = set()


def configure_torch_dll_path():
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    torch_spec = find_spec("torch")
    if torch_spec is None or torch_spec.submodule_search_locations is None:
        return

    torch_package_dir = Path(next(iter(torch_spec.submodule_search_locations)))
    torch_lib_dir = torch_package_dir / "lib"
    if not torch_lib_dir.is_dir():
        return

    torch_lib_path = str(torch_lib_dir)
    if torch_lib_path in _ADDED_DLL_DIRECTORIES:
        return

    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(torch_lib_path))
    _ADDED_DLL_DIRECTORIES.add(torch_lib_path)
