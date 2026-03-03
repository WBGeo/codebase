"""
Integration test inspired by Synthetic Model 2:
  - 1 vertical fault splitting the domain at x = 0.5
  - Lower group (inclined surfaces): rock1 (oldest), rock2
  - Upper group (horizontal, unconformity): rock3 (oldest of upper), rock4

All input data is generated analytically — no CSV files required.
The expected lithology is computed from the same geometric parameters,
giving an exact analytical reference to compare against.

Acceptance is per-method: interpolators known to be accurate must
achieve a higher interior match ratio than methods that are less precise
for dipping or faulted geometries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general  # type: ignore
import core.structural_modeling_components.general_faults as gf  # type: ignore

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
_FAULT_X = 0.5    # fault at x = 0.5
_THROW   = 0.08   # vertical offset; right side (x >= 0.5) is upthrown
_TILT    = 0.25   # dz/dx for lower group horizons (dips in x direction)

_Z_ROCK1 = 0.20   # base of rock1 at x = FAULT_X (left, no throw)
_Z_ROCK2 = 0.38   # base of rock2 at x = FAULT_X (left, no throw)
_Z_UNC   = 0.68   # unconformity (base of upper group)
_Z_ROCK4 = 0.84   # base of rock4


# ---------------------------------------------------------------------------
# Analytical input data (no CSV)
# ---------------------------------------------------------------------------

def _build_input_data() -> tuple[InputData_StructuralElements, InputData_FaultElements]:
    """
    Generate surface points and orientations analytically for the model2-inspired geometry.

    Lower group surfaces are inclined (dip in x-z). The fault introduces
    a vertical throw of _THROW on the right-hand side. Upper group surfaces
    are horizontal and unaffected by the fault offset.
    """
    # Surface normal for inclined lower horizons:
    #   F(x,z) = z - z0 - TILT*(x - FAULT_X) = 0
    #   grad F = (-TILT, 0, 1), normalised:
    _norm = np.sqrt(_TILT ** 2 + 1.0)
    GX_LOWER = float(-_TILT / _norm)
    GZ_LOWER = float(1.0 / _norm)

    xs_left  = np.linspace(0.05, 0.45, 5)
    xs_right = np.linspace(0.55, 0.95, 5)
    ys       = np.linspace(0.10, 0.90, 5)

    sp_rows: list[dict] = []
    ori_rows: list[dict] = []

    def _z_lower(x: float, right: bool, z_at_fault: float) -> float:
        return z_at_fault + _TILT * (x - _FAULT_X) + (_THROW if right else 0.0)

    def _add_lower(name: str, z_at_fault: float, xs: np.ndarray, right: bool) -> None:
        for x in xs:
            for y in ys:
                z = _z_lower(float(x), right, z_at_fault)
                sp_rows.append({"X": float(x), "Y": float(y), "Z": z, "formation": name})
                ori_rows.append({"X": float(x), "Y": float(y), "Z": z,
                                 "G_x": GX_LOWER, "G_y": 0.0, "G_z": GZ_LOWER,
                                 "formation": name})

    # Lower group: left and right domains sampled separately to encode the fault throw
    _add_lower("rock1", _Z_ROCK1, xs_left,  right=False)
    _add_lower("rock1", _Z_ROCK1, xs_right, right=True)
    _add_lower("rock2", _Z_ROCK2, xs_left,  right=False)
    _add_lower("rock2", _Z_ROCK2, xs_right, right=True)

    # Upper group: horizontal, sampled across full x range
    xs_upper = np.linspace(0.05, 0.95, 6)
    for x in xs_upper:
        for y in ys:
            for name, z in [("rock3", _Z_UNC), ("rock4", _Z_ROCK4)]:
                sp_rows.append({"X": float(x), "Y": float(y), "Z": z, "formation": name})
                ori_rows.append({"X": float(x), "Y": float(y), "Z": z,
                                 "G_x": 0.0, "G_y": 0.0, "G_z": 1.0,
                                 "formation": name})

    data_elements = InputData_StructuralElements(
        name="model2_inspired",
        mapping_object={
            "UpperSeries": ("rock4", "rock3"),   # youngest → oldest within group
            "LowerSeries": ("rock2", "rock1"),
        },
        surface_points=pd.DataFrame(sp_rows),
        orientations=pd.DataFrame(ori_rows),
    )

    # Fault: vertical plane at x = FAULT_X, normal along +x
    zs_fault = np.linspace(0.05, 0.95, 7)
    ys_fault = np.linspace(0.05, 0.95, 5)
    f_sp, f_ori = [], []
    for y in ys_fault:
        for z in zs_fault:
            f_sp.append({"X": _FAULT_X, "Y": float(y), "Z": float(z), "formation": "F1"})
            f_ori.append({"X": _FAULT_X, "Y": float(y), "Z": float(z),
                          "G_x": 1.0, "G_y": 0.0, "G_z": 0.0, "formation": "F1"})

    data_faults = InputData_FaultElements(
        name="model2_faults",
        fault_names=["F1"],
        fault_surface_points=pd.DataFrame(f_sp),
        fault_orientations=pd.DataFrame(f_ori),
    )

    return data_elements, data_faults


# ---------------------------------------------------------------------------
# Analytical expected lithology
# ---------------------------------------------------------------------------

def _expected_lithology(grid: RegularGrid) -> np.ndarray:
    """
    Compute the ground-truth lithology block from the exact geometric parameters.

    IDs (matching framework assignment order):
      0  basement   — below rock1
      1  rock1      — oldest of lower group (inclined + faulted)
      2  rock2      — upper lower group (inclined + faulted, eroded at unconformity)
      3  rock3      — base of upper group (horizontal, unconformity surface)
      4  rock4      — top of upper group (horizontal)
    """
    nx, ny, nz = map(int, grid.resolution)
    xc = np.asarray(grid.gridx)
    zc = np.asarray(grid.gridz)

    X = np.broadcast_to(xc.reshape(nx, 1, 1), (nx, ny, nz))
    Z = np.broadcast_to(zc.reshape(1, 1, nz), (nx, ny, nz))

    right = X >= _FAULT_X
    throw = np.where(right, _THROW, 0.0)

    z1 = _Z_ROCK1 + _TILT * (X - _FAULT_X) + throw   # base of rock1
    z2 = _Z_ROCK2 + _TILT * (X - _FAULT_X) + throw   # base of rock2

    expected = np.zeros((nx, ny, nz), dtype=int)
    expected[(Z >= z1) & (Z < z2) & (Z < _Z_UNC)] = 1   # rock1 (eroded at unconformity)
    expected[(Z >= z2) & (Z < _Z_UNC)]            = 2   # rock2
    expected[(Z >= _Z_UNC) & (Z < _Z_ROCK4)]      = 3   # rock3
    expected[Z >= _Z_ROCK4]                        = 4   # rock4
    return expected


# ---------------------------------------------------------------------------
# Acceptance thresholds and method list
# ---------------------------------------------------------------------------

# Minimum fraction of interior voxels (excluding 1-voxel boundary margin) that
# must match the analytical solution. Higher = stricter.
_MIN_INTERIOR_MATCH: dict[InterpolationMethod, float] = {
    InterpolationMethod.RADIAL_BASIS_FUNCTION: 0.95,
    InterpolationMethod.FINITE_DIFFERENCES:    0.95,
    InterpolationMethod.UNIVERSAL_COKRIGING:   0.90,
    InterpolationMethod.UNIVERSAL_KRIGING:     0.90,

    InterpolationMethod.ORDINARY_KRIGING:      0.75,
}

# GeoINR is non-deterministic and cannot be compared against an analytical solution
ALL_METHODS = [
    m for m in InterpolationMethod
    if m not in {
        InterpolationMethod.GEOINR,            # non-deterministic
    }
]


# ---------------------------------------------------------------------------
# Boundary mask helper (shared with other integration tests)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("method", ALL_METHODS)
def test_model2_fault_with_unconformity(method, plot_mode):
    """
    Model-2-inspired test: inclined lower group + horizontal upper group separated by
    an unconformity, cut by a single vertical fault with horizontal throw.

    Checks:
      1. Fault domains correctly split the model into left/right.
      2. Interior voxels match the analytical lithology above the per-method threshold.
      3. Structural sanity: basement, upper group, and fault offset are all present.
    """
    if method == InterpolationMethod.UNIVERSAL_COKRIGING:
        pytest.importorskip("gempy")
    pytest.importorskip("gempy")  # fault UCK interpolation also requires gempy

    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(20, 15, 10))

    data_elements, data_faults = _build_input_data()

    # --- Build and compute fault frame ---
    fault_frame = gf.build_fault_frame(data_faults, grid)
    fault_model_result = gf.compute_fault_domains(fault_frame)

    dm = fault_frame.domain_map
    assert dm is not None, "Fault domain map not set after compute_fault_domains"
    assert dm.shape == tuple(grid.resolution)

    domain_ids = np.unique(dm)
    assert len(domain_ids) == 2, f"Expected 2 fault domains, got {domain_ids}"

    # Verify the split is roughly at x = 0.5
    nx = int(grid.resolution[0])
    left_domain  = int(np.bincount(dm[: nx // 2].ravel()).argmax())
    right_domain = int(np.bincount(dm[nx // 2 :].ravel()).argmax())
    assert left_domain != right_domain, "Left and right halves should be in different domains"

    # --- Build structural frame and run full pipeline ---
    frame = general.build_structural_frame(
        input_data_elements=data_elements,
        grid=grid,
        fault_model_results=fault_model_result,
    )
    for g in frame.structural_groups:
        g.set_interpolation_method(method)

    res = general.compute_structural_model(frame, extract_meshes=False, verbose=False)
    lith = res.structural_frame._lith_block

    assert lith is not None
    assert lith.shape == tuple(grid.resolution)

    # --- Compute analytical reference ---
    expected = _expected_lithology(grid)

    # --- Interior match ratio (exclude 1-voxel boundary margin) ---
    interior = ~_near_boundary_mask(expected, radius=1)
    interior_match = float(np.mean((lith == expected)[interior]))
    min_match = _MIN_INTERIOR_MATCH.get(method, 0.85)

    from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import plot_structural_model_2D  # type: ignore
    if plot_mode["always"]:
        plot_structural_model_2D(res.structural_frame, title_suffix=method.value)

    try:
        assert interior_match >= min_match, (
            f"[{method.name}] Interior match {interior_match:.1%} < required {min_match:.0%}. "
            f"({int((lith != expected)[interior].sum())} interior voxels wrong)"
        )
    except AssertionError:
        if plot_mode["on_fail"]:
            plot_structural_model_2D(res.structural_frame, title_suffix=method.value)
        raise

    # --- Structural sanity checks ---
    assert (lith == 0).any(), "Basement (ID 0) should exist below the oldest horizon"
    assert (lith == 3).any(), "rock3 (unconformity base) should be present"
    assert (lith == 4).any(), "rock4 (topmost unit) should be present"

    # Fault throw check: rock2 should appear at a higher z-index on the right (upthrown) side
    ny = int(grid.resolution[1])
    mid_y  = ny // 2
    left_x = nx // 4          # well inside left domain
    right_x = (3 * nx) // 4   # well inside right domain

    col_left  = lith[left_x,  mid_y, :]
    col_right = lith[right_x, mid_y, :]

    def _first_idx(col: np.ndarray, val: int):
        idx = np.where(col == val)[0]
        return int(idx[0]) if idx.size else None

    z_rock2_left  = _first_idx(col_left,  2)
    z_rock2_right = _first_idx(col_right, 2)

    if z_rock2_left is not None and z_rock2_right is not None:
        assert z_rock2_right > z_rock2_left, (
            f"[{method.name}] Fault throw not detected: rock2 starts at z-index "
            f"{z_rock2_right} (right) vs {z_rock2_left} (left); right should be higher."
        )
