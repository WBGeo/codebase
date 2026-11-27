"""
Input data objects (core/object_components.py): they must not change the caller's DataFrames.
"""
from __future__ import annotations

import pandas as pd

from core.object_components import InputData_FaultElements, InputData_StructuralElements


def _points() -> tuple[pd.DataFrame, pd.DataFrame]:
    sp = pd.DataFrame({"X": [0.0, 1.0, 2.0, 3.0], "Y": [0.0] * 4, "Z": [0.0, 1.0, 2.0, 3.0],
                       "formation": ["rock2", "rock1", "fault", "rock1"]})
    ori = pd.DataFrame({"X": [0.0, 2.0], "Y": [0.0, 0.0], "Z": [0.0, 2.0], "G_x": [0.0, 1.0], "G_y": [0.0, 0.0],
                        "G_z": [1.0, 0.0], "formation": ["rock1", "fault"]})
    return sp, ori


def test_structural_input_leaves_the_callers_dataframes_unchanged():
    sp, ori = _points()
    sp_before, ori_before = sp.copy(), ori.copy()
    InputData_StructuralElements(name="m", mapping_object={"Series": ("rock2", "rock1")}, surface_points=sp,
                                 orientations=ori)
    pd.testing.assert_frame_equal(sp, sp_before)
    pd.testing.assert_frame_equal(ori, ori_before)


def test_one_dataframe_can_feed_the_structural_and_the_fault_input():
    """The structural input used to turn unmapped formations (the faults) into NaN in the caller's DataFrame."""
    sp, ori = _points()
    structural = InputData_StructuralElements(name="m", mapping_object={"Series": ("rock2", "rock1")},
                                              surface_points=sp, orientations=ori)
    faults = InputData_FaultElements(name="f", fault_surface_points=sp, fault_orientations=ori,
                                     fault_names=["fault"])
    assert (faults.fault_surface_points["formation"] == "fault").sum() == 1
    # The structural input itself is still sorted in mapping order.
    assert list(structural.surface_points["formation"][:3]) == ["rock2", "rock1", "rock1"]
