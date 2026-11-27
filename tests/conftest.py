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


@pytest.fixture
def fake_inspector():
    """Stand-in for the Workbench's InspectorHelper (the ``_inspector`` argument of inspectors).

    ``fake_inspector(value)`` returns an object whose ``await (await trace(SomeType)).get_value()``
    gives ``value``; pass a dict ``{SomeType: value, ...}`` to answer per traced type.
    """
    class _Trace:
        def __init__(self, value):
            self._value = value

        async def get_value(self):
            return self._value

    class _Inspector:
        def __init__(self, traced):
            self._traced = traced

        async def trace(self, traced_type):
            if isinstance(self._traced, dict):
                return _Trace(self._traced[traced_type])
            return _Trace(self._traced)

    return _Inspector
