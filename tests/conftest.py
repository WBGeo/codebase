import os

import pytest


def _register_each_dll_directory_once() -> None:
    """
    Make ``os.add_dll_directory`` register each directory only once (Windows only).

    Every ``gempy.compute_model`` makes GemPy build a fresh copy of the numpy
    module, and the Windows build of numpy registers its DLL directory again on
    each import. Over the many GemPy calls of a full test session the process's
    DLL search list overflows and imports fail with ``WinError 206``. Returning
    the existing handle for an already registered directory keeps the list
    short. Linux (CI, Docker) has no ``os.add_dll_directory`` and is unaffected.
    """
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    handles = {}

    def add_once(path):
        key = os.path.normcase(os.path.abspath(os.fspath(path)))
        if key not in handles:
            handles[key] = add_dll_directory(path)
        return handles[key]

    os.add_dll_directory = add_once


_register_each_dll_directory_once()


def pytest_addoption(parser):
    parser.addoption("--show-plots", action="store_true", default=False,
                     help="Show plots when a test fails")
    parser.addoption("--show-plots-always", action="store_true", default=False,
                     help="Always show plots (even on pass)")

@pytest.fixture
def plot_mode(request):
    return {
        "on_fail": request.config.getoption("--show-plots"),
        "always": request.config.getoption("--show-plots-always"),
    }