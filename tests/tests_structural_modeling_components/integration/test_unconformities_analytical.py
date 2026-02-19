from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general  # type: ignore

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod


# -----------------------------------------------------------------------------
# Geometry (analytical)
# -----------------------------------------------------------------------------

def _plane_45_x(x: np.ndarray, z0: float) -> np.ndarray:
    """45° in x-z on [0,1]: slope 1 => dz/dx = 1."""
    return z0 + (x - 0.5)


def _expected_solution(grid: RegularGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IDs:
      0 basement
      1 rock1 (oldest of lower group)
      2 rock2
      3 rock3 (oldest of upper group, base of upper group / unconformity)
      4 rock4
    Masks:
      - upper mask True where z >= z_rock3 (horizontal)
      - lower mask all True
    """
    nx, ny, nz = map(int, grid.resolution)

    # cell centers
    xc = np.asarray(grid.gridx)
    yc = np.asarray(grid.gridy)
    zc = np.asarray(grid.gridz)

    X, Y = np.meshgrid(xc, yc, indexing="ij")          # (nx, ny)
    Z = np.broadcast_to(zc.reshape(1, 1, nz), (nx, ny, nz))

    # Lower group (45° inclined)
    z_rock1 = _plane_45_x(X, z0=0.20)                  # basement/rock1 boundary
    z_rock2 = _plane_45_x(X, z0=0.40)                  # rock1/rock2 boundary

    # Upper group (horizontal + erosive)
    z_rock3 = 0.62                                     # unconformity/base of upper group
    z_rock4 = 0.82                                     # rock3/rock4 boundary

    # broadcast to 3D
    z1 = np.broadcast_to(z_rock1[:, :, None], (nx, ny, nz))
    z2 = np.broadcast_to(z_rock2[:, :, None], (nx, ny, nz))

    expected = np.zeros((nx, ny, nz), dtype=int)        # basement

    # Lower units only below unconformity (erosion)
    rock1 = (Z >= z1) & (Z < z2) & (Z < z_rock3)
    expected[rock1] = 1

    rock2 = (Z >= z2) & (Z < z_rock3)
    expected[rock2] = 2

    # Upper units overwrite above unconformity
    rock3 = (Z >= z_rock3) & (Z < z_rock4)
    expected[rock3] = 3

    rock4 = Z >= z_rock4
    expected[rock4] = 4

    upper_mask = Z >= z_rock3
    lower_mask = np.ones((nx, ny, nz), dtype=bool)
    return expected, upper_mask, lower_mask


def _input_unconformity_more_points() -> InputData_StructuralElements:
    """
    Two groups (youngest->oldest):
      UpperSeries: rock4, rock3  (both horizontal)
      LowerSeries: rock2, rock1  (both 45° in x)
    """
    mapping_object = {
        "UpperSeries": ("rock4", "rock3"),
        "LowerSeries": ("rock2", "rock1"),
    }

    # Dense XY sampling for strong constraint (5x5 = 25 points per surface)
    xs = np.linspace(0.05, 0.95, 5)
    ys = np.linspace(0.05, 0.95, 5)
    samples = [(float(x), float(y)) for x in xs for y in ys]

    sp_rows = []
    ori_rows = []

    # Lower (45°)
    def z_rock1(x): return float(0.20 + (x - 0.5))
    def z_rock2(x): return float(0.40 + (x - 0.5))

    # Upper (horizontal)
    z_rock3 = 0.62
    z_rock4 = 0.82

    def add_surface(name: str, z_value_fn):
        for (x, y) in samples:
            z = float(z_value_fn(x))
            sp_rows.append({"X": x, "Y": y, "Z": z, "formation": name})
            # Simple vertical orientation; enough for many workflows (UCK included).
            ori_rows.append({"X": x, "Y": y, "Z": z, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "formation": name})

    add_surface("rock1", z_rock1)
    add_surface("rock2", z_rock2)
    add_surface("rock3", lambda x: z_rock3)
    add_surface("rock4", lambda x: z_rock4)

    return InputData_StructuralElements(
        name="toy_unconformity",
        mapping_object=mapping_object,
        surface_points=pd.DataFrame(sp_rows),
        orientations=pd.DataFrame(ori_rows),
    )


def _match_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


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

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("method", ALL_GROUP_METHODS)
def test_unconformity_lithology_and_masks(method, plot_mode):
    if method == InterpolationMethod.UNIVERSAL_COKRIGING:
        pytest.importorskip("gempy")

    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(10, 10, 10))

    data = _input_unconformity_more_points()
    frame = general.build_structural_frame(input_data_elements=data, grid=grid)

    for g in frame.structural_groups:
        g.set_interpolation_method(method)

    res = general.compute_structural_model(frame, extract_meshes=False, verbose=False)
    out_frame = res.structural_frame

    lith = out_frame._lith_block
    assert lith is not None
    assert lith.shape == tuple(grid.resolution)

    expected_lith, expected_upper_mask, expected_lower_mask = _expected_solution(grid)

    # group order: UpperSeries then LowerSeries
    upper = out_frame.structural_groups[0]
    lower = out_frame.structural_groups[1]

    assert np.array_equal(lower.mask, expected_lower_mask)
    assert np.array_equal(upper.mask, expected_upper_mask)

    from core.visualization_components import plot_structural_model_2D  # type: ignore
    if plot_mode["always"]:
        plot_structural_model_2D(out_frame)

    interior = ~_near_boundary_mask(expected_lith, radius=1)
    interior_match = float(np.mean((lith == expected_lith)[interior]))
    min_match = _MIN_INTERIOR_MATCH.get(method, 0.85)

    try:
        assert interior_match >= min_match, (
            f"[{method.name}] Interior match {interior_match:.1%} < required {min_match:.0%}. "
            f"({int((lith != expected_lith)[interior].sum())} interior voxels wrong)"
        )
    except AssertionError:
        if plot_mode["on_fail"]:
            plot_structural_model_2D(out_frame)
        raise

    # Extra safety: basement must exist and be below the lowest horizon in many cells
    assert (lith == 0).any()


def _near_boundary_mask(expected_lith: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """
    Returns True for voxels within `radius` voxels of any lithology boundary
    in the expected block (6-neighborhood).
    """
    # boundary where any 6-neighbor has different lith ID
    b = np.zeros_like(expected_lith, dtype=bool)

    b[1:, :, :] |= expected_lith[1:, :, :] != expected_lith[:-1, :, :]
    b[:-1, :, :] |= expected_lith[:-1, :, :] != expected_lith[1:, :, :]
    b[:, 1:, :] |= expected_lith[:, 1:, :] != expected_lith[:, :-1, :]
    b[:, :-1, :] |= expected_lith[:, :-1, :] != expected_lith[:, 1:, :]
    b[:, :, 1:] |= expected_lith[:, :, 1:] != expected_lith[:, :, :-1]
    b[:, :, :-1] |= expected_lith[:, :, :-1] != expected_lith[:, :, 1:]

    # dilate by `radius` using simple shifting (no SciPy dependency)
    near = b.copy()
    for _ in range(radius):
        expanded = near.copy()
        expanded[1:, :, :] |= near[:-1, :, :]
        expanded[:-1, :, :] |= near[1:, :, :]
        expanded[:, 1:, :] |= near[:, :-1, :]
        expanded[:, :-1, :] |= near[:, 1:, :]
        expanded[:, :, 1:] |= near[:, :, :-1]
        expanded[:, :, :-1] |= near[:, :, 1:]
        near = expanded
    return near

