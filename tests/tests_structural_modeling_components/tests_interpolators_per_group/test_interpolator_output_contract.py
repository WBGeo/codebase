"""
Output contract tests for all interpolators.

Verifies that every interpolator, when called through
run_interpolation_with_fault_domains on a simple layercake, produces:
  1. A scalar field with the correct shape (nx, ny, nz) after XYZ transpose
  2. All-finite values (no NaN / Inf)
  3. Spatial variation (field is not constant)
  4. Scalar values set on every element (not None, finite)

An asymmetric grid resolution (5, 4, 3) is used deliberately so that any
axis-order bug (e.g. returning ZYX instead of XYZ) produces the wrong shape
and is caught immediately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general  # type: ignore

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod


# ---------------------------------------------------------------------------
# Shared input data
# ---------------------------------------------------------------------------

def _layercake_input() -> InputData_StructuralElements:
    """
    Simple horizontal 2-layer model: rock2 (young) above rock1 (old).
    9 surface points per formation (3×3 XY grid) for numerical stability.
    """
    mapping_object = {"Series": ("rock2", "rock1")}
    xs = np.linspace(0.1, 0.9, 3)
    ys = np.linspace(0.1, 0.9, 3)
    z_rock1, z_rock2 = 0.3, 0.6

    sp_rows, ori_rows = [], []
    for x in xs:
        for y in ys:
            for name, z in [("rock1", z_rock1), ("rock2", z_rock2)]:
                sp_rows.append({"X": float(x), "Y": float(y), "Z": z, "formation": name})
                ori_rows.append({"X": float(x), "Y": float(y), "Z": z,
                                 "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "formation": name})

    return InputData_StructuralElements(
        name="contract_layercake",
        mapping_object=mapping_object,
        surface_points=pd.DataFrame(sp_rows),
        orientations=pd.DataFrame(ori_rows),
    )


# ---------------------------------------------------------------------------
# Parametrized contract test
# ---------------------------------------------------------------------------

ALL_METHODS = [m for m in InterpolationMethod if m != InterpolationMethod.GEOINR]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_interpolator_output_contract(method):
    """
    For every interpolator:
      - scalar field has shape == grid.resolution (nx, ny, nz)
      - scalar field contains only finite values
      - scalar field varies spatially (not degenerate/constant)
      - every element has a finite scalar value assigned
    """
    if method == InterpolationMethod.UNIVERSAL_COKRIGING:
        pytest.importorskip("gempy")

    # Asymmetric resolution: (5, 4, 3) ensures shape (5, 4, 3) ≠ (3, 4, 5),
    # so an axis-order bug would produce a wrong shape and fail assertion 1.
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(5, 4, 3))

    data = _layercake_input()
    frame = general.build_structural_frame(input_data_elements=data, grid=grid)

    for g in frame.structural_groups:
        g.set_interpolation_method(method)

    general.run_interpolation_with_fault_domains(frame=frame, fault_frame=None)

    for group in frame.structural_groups:
        sf = group.get_scalar_field()

        # 1. Shape: must be (nx, ny, nz) = grid.resolution after XYZ transpose
        assert sf is not None, f"[{method.name}] Group '{group.name}' has no scalar field"
        assert sf.shape == tuple(grid.resolution), (
            f"[{method.name}] Group '{group.name}': expected shape {tuple(grid.resolution)}, "
            f"got {sf.shape}. Possible axis-order bug."
        )

        # 2. Dtype and finiteness
        assert np.issubdtype(sf.dtype, np.floating), (
            f"[{method.name}] Scalar field dtype is {sf.dtype}, expected float"
        )
        assert np.isfinite(sf).all(), (
            f"[{method.name}] Group '{group.name}': scalar field contains NaN or Inf"
        )

        # 3. Spatial variation — a constant field means the interpolation is degenerate
        assert sf.max() > sf.min(), (
            f"[{method.name}] Group '{group.name}': scalar field is spatially constant "
            f"(min == max == {sf.min():.4f})"
        )

        # 4. Every element must have a finite scalar value assigned
        for elem in group.structural_elements:
            sval = elem.get_scalar_value()
            assert sval is not None, (
                f"[{method.name}] Element '{elem.name}' has no scalar value"
            )
            assert np.isfinite(sval), (
                f"[{method.name}] Element '{elem.name}' has non-finite scalar value: {sval}"
            )
