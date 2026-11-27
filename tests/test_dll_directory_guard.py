"""
The test session registers each DLL directory once (see conftest.py), so the
many GemPy calls of a full run don't overflow Windows' DLL search list.
"""

import os
import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="os.add_dll_directory exists only on Windows")
def test_a_dll_directory_is_registered_only_once(tmp_path):
    first = os.add_dll_directory(str(tmp_path))
    second = os.add_dll_directory(str(tmp_path))
    assert first is second


@pytest.mark.skipif(sys.platform != "win32", reason="os.add_dll_directory exists only on Windows")
def test_repeated_gempy_calls_do_not_exhaust_the_dll_search_list():
    """600 GemPy evaluations in one process; without the guard this failed at call 434 (WinError 206)."""
    pytest.importorskip("gempy")
    import warnings

    import numpy as np
    import pandas as pd

    from core.structural_modeling_components.interpolator_functions.universal_cokriging import (
        fit_fault_universal_cokriging,
    )

    zs = np.linspace(0.1, 0.9, 4)
    points = pd.DataFrame([{"X": 0.5, "Y": y, "Z": z, "formation": "F"} for y in zs for z in zs])
    orientations = points.assign(G_x=1.0, G_y=0.0, G_z=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = fit_fault_universal_cokriging(
            fault_name="F", extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            fault_surface_points_df=points, fault_orientations_points_df=orientations,
        )
        for _ in range(600):
            field.evaluate(np.array([[0.25, 0.5, 0.5]]))
