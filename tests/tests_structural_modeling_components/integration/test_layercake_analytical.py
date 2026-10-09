from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general  # type: ignore

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod


def _layercake_input(
        z_top: float = 0.6, z_bottom: float = 0.3,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> InputData_StructuralElements:
    """
    2-layer layercake with basement below oldest:

      basement: z < z_bottom        -> id 0
      rock1   : z_bottom <= z < z_top -> id 1
      rock2   : z >= z_top            -> id 2

    Mapping order: ("rock2","rock1") (youngest -> oldest).

    `offset` shifts every coordinate by (ox, oy, oz) -- used to build an
    otherwise-identical scenario whose grid does not start at (0, 0, 0), to
    catch any interpolator path that (like piecewise_linear.py's past
    evaluate_feature_value(..., scale=False) bug) only works correctly when
    the model happens to already sit at the origin.
    """
    ox, oy, oz = offset
    mapping_object = {"Strat_Series1": ("rock2", "rock1")}

    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]

    sp_rows = []
    for (x, y) in corners:
        sp_rows.append({"X": x + ox, "Y": y + oy, "Z": z_top + oz, "formation": "rock2"})
        sp_rows.append({"X": x + ox, "Y": y + oy, "Z": z_bottom + oz, "formation": "rock1"})
    surface_points = pd.DataFrame(sp_rows)

    ori_rows = []
    for (x, y) in corners:
        ori_rows.append({"X": x + ox, "Y": y + oy, "Z": z_top + oz, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "formation": "rock2"})
        ori_rows.append({"X": x + ox, "Y": y + oy, "Z": z_bottom + oz, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "formation": "rock1"})
    orientations = pd.DataFrame(ori_rows)

    return InputData_StructuralElements(
        name="toy_layercake",
        mapping_object=mapping_object,
        surface_points=surface_points,
        orientations=orientations,
    )


def _expected_lith_block(
        grid: RegularGrid, z_top: float, z_bottom: float,
) -> np.ndarray:
    """
    Analytical lithology block (XYZ order) including basement:

      z < z_bottom           -> basement id 0
      z_bottom <= z < z_top  -> rock1 id 1
      z >= z_top             -> rock2 id 2

    Your pipeline assigns rock IDs:
      rock1 -> 1, rock2 -> 2
    basement is 0 by framework rule.

    z_top/z_bottom must already include any coordinate offset applied to
    the input data (grid.gridz is always in absolute/world coordinates).
    """
    nx, ny, nz = map(int, grid.resolution)
    zc = grid.gridz

    zvol = np.broadcast_to(zc.reshape(1, 1, nz), (nx, ny, nz))

    expected = np.zeros((nx, ny, nz), dtype=int)  # basement everywhere
    expected[(zvol >= z_bottom) & (zvol < z_top)] = 1  # rock1
    expected[zvol >= z_top] = 2  # rock2
    return expected


ALL_GROUP_METHODS = list(InterpolationMethod)

# Minimum fraction of interior voxels (1-voxel boundary margin excluded) that
# must match the analytical solution.
_MIN_INTERIOR_MATCH: dict[InterpolationMethod, float] = {
    InterpolationMethod.RADIAL_BASIS_FUNCTION: 0.95,
    InterpolationMethod.FINITE_DIFFERENCES:    0.95,
    InterpolationMethod.UNIVERSAL_COKRIGING:   0.90,
    InterpolationMethod.UNIVERSAL_KRIGING:     0.90,

    InterpolationMethod.ORDINARY_KRIGING:      0.90,
    # GeoINR is non-deterministic (unseeded random init); 0.90 gives a safe margin below
    # the 1.000 interior match observed on this scenario across 15 unseeded trials, so a
    # normal run-to-run variation doesn't make this test flaky.
    InterpolationMethod.GEOINR:                0.90,
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
        if method == InterpolationMethod.GEOINR:
            # default is 4000 epochs; 1000 passed 6/6 unseeded trials here and keeps the suite fast
            g.configure_interpolation_params(epochs=1000)

    res = general.compute_structural_model(frame, extract_meshes=False, verbose=False)

    out_frame = res.structural_frame
    lith = out_frame.lith_block
    expected = _expected_lith_block(grid, z_top=z_top, z_bottom=z_bottom)

    from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import plot_structural_model_2D  # type: ignore

    if plot_mode["always"]:
        plot_structural_model_2D(res.structural_frame, title_suffix=method.value)

    interior = ~_near_boundary_mask(expected, radius=1)
    interior_match = float(np.mean((lith == expected)[interior]))
    min_match = _MIN_INTERIOR_MATCH.get(method, 0.85)

    try:
        assert interior_match >= min_match, (
            f"[{method.name}] Interior match {interior_match:.1%} < required {min_match:.0%}"
        )
    except AssertionError:
        if plot_mode["on_fail"]:
            plot_structural_model_2D(res.structural_frame, title_suffix=method.value)
        raise


# Realistic real-world-scale offset (comparable to UTM easting/northing/elevation)
# -- deliberately large relative to the model's own 0..1 extent, so any
# interpolator path that silently assumes (or only works correctly at) a
# zero origin would be caught here.
_NONZERO_ORIGIN_OFFSET = (523_000.0, 5_810_000.0, 1_200.0)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("method", ALL_GROUP_METHODS)
def test_layercake_analytical_lithology_block_nonzero_origin(method, plot_mode):
    """
    Same scenario as test_layercake_analytical_lithology_block, but translated
    to a non-zero, real-world-scale origin. Every interpolation method should
    be translation-invariant: the analytical answer is identical, just shifted.
    Regression guard for bugs like piecewise_linear.py's past
    evaluate_feature_value(..., scale=False), which only matched the (0,0,0)
    version of this test and silently diverged once the grid moved.
    """
    if method == InterpolationMethod.UNIVERSAL_COKRIGING:
        pytest.importorskip("gempy")

    ox, oy, oz = _NONZERO_ORIGIN_OFFSET
    extent = (0.0 + ox, 1.0 + ox, 0.0 + oy, 1.0 + oy, 0.0 + oz, 1.0 + oz)
    grid = RegularGrid(extent=extent, resolution=(10, 10, 10))

    z_bottom = 0.3 + oz
    z_top = 0.6 + oz
    data_elements = _layercake_input(z_top=0.6, z_bottom=0.3, offset=_NONZERO_ORIGIN_OFFSET)

    frame = general.build_structural_frame(input_data_elements=data_elements, grid=grid)

    for g in frame.structural_groups:
        g.set_interpolation_method(method)

    res = general.compute_structural_model(frame, extract_meshes=False, verbose=False)

    out_frame = res.structural_frame
    lith = out_frame.lith_block
    expected = _expected_lith_block(grid, z_top=z_top, z_bottom=z_bottom)

    from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import plot_structural_model_2D  # type: ignore

    if plot_mode["always"]:
        plot_structural_model_2D(res.structural_frame, title_suffix=method.value)

    interior = ~_near_boundary_mask(expected, radius=1)
    interior_match = float(np.mean((lith == expected)[interior]))
    min_match = _MIN_INTERIOR_MATCH.get(method, 0.85)

    try:
        assert interior_match >= min_match, (
            f"[{method.name}] Interior match {interior_match:.1%} < required {min_match:.0%} "
            f"at non-zero origin {_NONZERO_ORIGIN_OFFSET}"
        )
    except AssertionError:
        if plot_mode["on_fail"]:
            plot_structural_model_2D(res.structural_frame, title_suffix=method.value)
        raise
