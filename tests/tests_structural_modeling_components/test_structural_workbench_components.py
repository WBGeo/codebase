"""Tests of the Workbench pre-check for interpolation input data (Compute Structural Model)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components.interpolator_functions.interpolator_parameters import (
    InterpolationMethod, RBFParams)
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_workbench_components import (
    StructuralFrameInputOptions, StructuralFrameInputOptions_Frame, StructuralFrameInputOptions_Root,
    check_interpolation_inputs)


def _elements(n_points: int) -> InputData_StructuralElements:
    rng = np.random.default_rng(0)
    return InputData_StructuralElements(
        name="test",
        mapping_object={"Series": ("upper", "lower")},
        surface_points=pd.DataFrame({
            "X": rng.uniform(100, 900, n_points),
            "Y": rng.uniform(100, 700, n_points),
            "Z": rng.uniform(100, 500, n_points),
            "formation": (["upper", "lower"] * n_points)[:n_points],
        }),
    )


def _grid() -> RegularGrid:
    return RegularGrid(extent=(0, 1000, 0, 800, 0, 600), resolution=(5, 4, 3))


def _rbf_options(kernel: str) -> StructuralFrameInputOptions:
    return StructuralFrameInputOptions(root=StructuralFrameInputOptions_Root(frame={
        "Series": StructuralFrameInputOptions_Frame(method=InterpolationMethod.RADIAL_BASIS_FUNCTION,
                                                    rbf=RBFParams(kernel=kernel)),
    }))


def test_pre_check_passes_with_enough_points():
    check_interpolation_inputs(_elements(8), _grid())  # must not raise


def test_pre_check_reports_too_few_points_for_default_kernel():
    with pytest.raises(ValueError, match=r"Group 'Series'.*thin_plate_spline.*at least 4 surface points"):
        check_interpolation_inputs(_elements(3), _grid())


def test_pre_check_uses_kernel_from_options():
    check_interpolation_inputs(_elements(3), _grid(), options=_rbf_options("linear"))  # must not raise
    with pytest.raises(ValueError, match="at least 10 surface points"):
        check_interpolation_inputs(_elements(8), _grid(), options=_rbf_options("quintic"))
