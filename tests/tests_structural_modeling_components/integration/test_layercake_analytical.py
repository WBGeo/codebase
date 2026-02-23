from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general  # type: ignore

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod


def _layercake_input(z_top: float = 0.6, z_bottom: float = 0.3) -> InputData_StructuralElements:
    """
    2-layer layercake with basement below oldest:

      basement: z < z_bottom        -> id 0
      rock1   : z_bottom <= z < z_top -> id 1
      rock2   : z >= z_top            -> id 2

    Mapping order: ("rock2","rock1") (youngest -> oldest).
    """
    mapping_object = {"Strat_Series1": ("rock2", "rock1")}

    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]

    sp_rows = []
    for (x, y) in corners:
        sp_rows.append({"X": x, "Y": y, "Z": z_top, "formation": "rock2"})
        sp_rows.append({"X": x, "Y": y, "Z": z_bottom, "formation": "rock1"})
    surface_points = pd.DataFrame(sp_rows)

    ori_rows = []
    for (x, y) in corners:
        ori_rows.append({"X": x, "Y": y, "Z": z_top, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "formation": "rock2"})
        ori_rows.append({"X": x, "Y": y, "Z": z_bottom, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "formation": "rock1"})
    orientations = pd.DataFrame(ori_rows)

    return InputData_StructuralElements(
        name="toy_layercake",
        mapping_object=mapping_object,
        surface_points=surface_points,
        orientations=orientations,
    )


def _expected_lith_block(grid: RegularGrid, z_top: float, z_bottom: float) -> np.ndarray:
    """
    Analytical lithology block (XYZ order) including basement:

      z < z_bottom           -> basement id 0
      z_bottom <= z < z_top  -> rock1 id 1
      z >= z_top             -> rock2 id 2

    Your pipeline assigns rock IDs:
      rock1 -> 1, rock2 -> 2
    basement is 0 by framework rule.
    """
    nx, ny, nz = map(int, grid.resolution)
    zc = grid.gridz

    zvol = np.broadcast_to(zc.reshape(1, 1, nz), (nx, ny, nz))

    expected = np.zeros((nx, ny, nz), dtype=int)  # basement everywhere
    expected[(zvol >= z_bottom) & (zvol < z_top)] = 1  # rock1
    expected[zvol >= z_top] = 2  # rock2
    return expected


ALL_GROUP_METHODS = [
    m for m in InterpolationMethod
    if m != InterpolationMethod.GEOINR  # non-deterministic
]

# Minimum fraction of interior voxels (1-voxel boundary margin excluded) that
# must match the analytical solution.
_MIN_INTERIOR_MATCH: dict[InterpolationMethod, float] = {
    InterpolationMethod.RADIAL_BASIS_FUNCTION: 0.95,
    InterpolationMethod.FINITE_DIFFERENCES:    0.95,
    InterpolationMethod.UNIVERSAL_COKRIGING:   0.90,
    InterpolationMethod.UNIVERSAL_KRIGING:     0.90,
    InterpolationMethod.PIECEWISE_LINEAR:      0.80,
    InterpolationMethod.ORDINARY_KRIGING:      0.75,
}


def _near_boundary_mask(lith: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """True for voxels within `radius` of any lithology boundary (6-neighbourhood)."""
    b = np.zeros_like(lith, dtype=bool)
    b[1:,  :,  :] |= lith[1:,  :,  :] != lith[:-1, :,  :]
    b[:-1, :,  :] |= lith[:-1, :,  :] != lith[1:,  :,  :]
    b[:,  1:,  :] |= lith[:,  1:,  :] != lith[:,  :-1, :]
    b[:, :-1,  :] |= lith[:, :-1,  :] != lith[:,  1:,  :]
    b[:, :,   1:] |= lith[:, :,   1:] != lith[:, :,  :-1]
    b[:, :,  :-1] |= lith[:, :,  :-1] != lith[:, :,   1:]
    near = b.copy()
    for _ in range(radius):
        exp = near.copy()
        exp[1:,  :,  :] |= near[:-1, :,  :]
        exp[:-1, :,  :] |= near[1:,  :,  :]
        exp[:,  1:,  :] |= near[:,  :-1, :]
        exp[:, :-1,  :] |= near[:,  1:,  :]
        exp[:, :,   1:] |= near[:, :,  :-1]
        exp[:, :,  :-1] |= near[:, :,   1:]
        near = exp
    return near


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("method", ALL_GROUP_METHODS)
def test_layercake_analytical_lithology_block(method, plot_mode):
    if method == InterpolationMethod.UNIVERSAL_COKRIGING:
        pytest.importorskip("gempy")

    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(10, 10, 10))

    z_bottom = 0.3
    z_top = 0.6
    data_elements = _layercake_input(z_top=z_top, z_bottom=z_bottom)

    frame = general.build_structural_frame(input_data_elements=data_elements, grid=grid)

    for g in frame.structural_groups:
        g.set_interpolation_method(method)

    res = general.compute_structural_model(frame, extract_meshes=False, verbose=False)

    out_frame = res.structural_frame
    lith = out_frame._lith_block
    expected = _expected_lith_block(grid, z_top=z_top, z_bottom=z_bottom)

    from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import plot_structural_model_2D  # type: ignore

    if plot_mode["always"]:
        plot_structural_model_2D(res.structural_frame)

    interior = ~_near_boundary_mask(expected, radius=1)
    interior_match = float(np.mean((lith == expected)[interior]))
    min_match = _MIN_INTERIOR_MATCH.get(method, 0.85)

    try:
        assert interior_match >= min_match, (
            f"[{method.name}] Interior match {interior_match:.1%} < required {min_match:.0%}"
        )
    except AssertionError:
        if plot_mode["on_fail"]:
            plot_structural_model_2D(res.structural_frame)
        raise
