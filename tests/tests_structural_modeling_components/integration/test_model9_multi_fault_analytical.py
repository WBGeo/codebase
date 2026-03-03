"""
Integration test inspired by Synthetic Model 9 (simplified):
  - 2 vertical faults creating 3 fault domains
  - 3 groups (C youngest, B middle, A oldest) separated by unconformities
  - F1 at x=1/3 affects only Group A (oldest); F2 at x=2/3 affects Groups A and B

Fault-activity table:
  Group C (idx=0): both faults inactive → all domains merged → no offset
  Group B (idx=1): F1 inactive, F2 active → left+mid merged, right separate → +throw2 in right
  Group A (idx=2): both faults active → 3 separate domains → +throw1 in mid, +throw1+throw2 in right

All input data is generated analytically — no CSV files required.
The expected lithology is computed from the same geometric parameters,
giving an exact analytical reference to compare against.
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
_F1_X = 1.0 / 3.0   # fault 1 at x = 1/3; affects Group A only
_F2_X = 2.0 / 3.0   # fault 2 at x = 2/3; affects Groups A and B
_THROW1 = 0.10       # vertical throw for F1 (right side upthrown)
_THROW2 = 0.08       # vertical throw for F2 (right side upthrown)

# Z-levels (horizontal surfaces, none displaced by faults)
_Z_ROCK1  = 0.08     # base of rock1 in left domain (below = basement)
_Z_ROCK2  = 0.20     # base of rock2 in left domain (within Group A)
_Z_UNC_AB = 0.45     # unconformity A/B (fixed, not displaced)
_Z_ROCK4  = 0.54     # base of rock4 in left+mid domain (within Group B)
_Z_UNC_BC = 0.70     # unconformity B/C (fixed, not displaced)
_Z_ROCK6  = 0.83     # base of rock6 (within Group C)


# ---------------------------------------------------------------------------
# Analytical input data (no CSV)
# ---------------------------------------------------------------------------

def _build_input_data() -> tuple[InputData_StructuralElements, InputData_FaultElements]:
    """
    Generate surface points and orientations analytically.

    Group A (oldest, idx=2): both faults active → 3 x-bands
    Group B (middle, idx=1): only F2 active → 2 x-bands (left+mid merged, right separate)
    Group C (youngest, idx=0): no faults active → full x range
    """
    ys = np.linspace(0.10, 0.90, 5)

    # x-ranges for the 3 fault domains
    xs_left     = np.linspace(0.05, 0.28, 4)   # x < F1_X
    xs_mid      = np.linspace(0.39, 0.61, 4)   # F1_X <= x < F2_X
    xs_right    = np.linspace(0.72, 0.95, 4)   # x >= F2_X
    xs_left_mid = np.linspace(0.05, 0.62, 6)   # left+mid merged (for group B / group C)
    xs_full     = np.linspace(0.05, 0.95, 6)   # full range (for group C)

    sp_rows: list[dict] = []
    ori_rows: list[dict] = []

    def _add_surface(name: str, z: float, xvals: np.ndarray) -> None:
        for x in xvals:
            for y in ys:
                sp_rows.append({"X": float(x), "Y": float(y), "Z": float(z), "formation": name})
                ori_rows.append({"X": float(x), "Y": float(y), "Z": float(z),
                                 "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "formation": name})

    # Group A (oldest): both F1 and F2 active → 3 separate domains
    _add_surface("rock1", _Z_ROCK1,                       xs_left)
    _add_surface("rock1", _Z_ROCK1 + _THROW1,             xs_mid)
    _add_surface("rock1", _Z_ROCK1 + _THROW1 + _THROW2,   xs_right)

    _add_surface("rock2", _Z_ROCK2,                       xs_left)
    _add_surface("rock2", _Z_ROCK2 + _THROW1,             xs_mid)
    _add_surface("rock2", _Z_ROCK2 + _THROW1 + _THROW2,   xs_right)

    # Group B (middle): F1 inactive → left+mid merged; F2 active → right separate
    # rock3 is at the unconformity (not displaced)
    _add_surface("rock3", _Z_UNC_AB,              xs_full)
    _add_surface("rock4", _Z_ROCK4,               xs_left_mid)
    _add_surface("rock4", _Z_ROCK4 + _THROW2,     xs_right)

    # Group C (youngest): both faults inactive → single merged domain
    _add_surface("rock5", _Z_UNC_BC, xs_full)
    _add_surface("rock6", _Z_ROCK6,  xs_full)

    data_elements = InputData_StructuralElements(
        name="model9_mini",
        mapping_object={
            "C": ("rock6", "rock5"),   # youngest group; rock6 youngest, rock5 oldest
            "B": ("rock4", "rock3"),
            "A": ("rock2", "rock1"),   # oldest group; rock2 youngest, rock1 oldest
        },
        surface_points=pd.DataFrame(sp_rows),
        orientations=pd.DataFrame(ori_rows),
    )

    # Two vertical fault planes (normal along +x)
    zs_fault = np.linspace(0.05, 0.95, 7)
    ys_fault = np.linspace(0.05, 0.95, 5)
    f_sp: list[dict] = []
    f_ori: list[dict] = []
    for y in ys_fault:
        for z in zs_fault:
            f_sp.append({"X": _F1_X, "Y": float(y), "Z": float(z), "formation": "F1"})
            f_ori.append({"X": _F1_X, "Y": float(y), "Z": float(z),
                          "G_x": 1.0, "G_y": 0.0, "G_z": 0.0, "formation": "F1"})
            f_sp.append({"X": _F2_X, "Y": float(y), "Z": float(z), "formation": "F2"})
            f_ori.append({"X": _F2_X, "Y": float(y), "Z": float(z),
                          "G_x": 1.0, "G_y": 0.0, "G_z": 0.0, "formation": "F2"})

    data_faults = InputData_FaultElements(
        name="model9_mini_faults",
        fault_names=["F1", "F2"],
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
      0  basement  — below rock1 horizon
      1  rock1     — bottom of Group A (oldest, faulted by both F1 and F2)
      2  rock2     — top of Group A (eroded at unconformity AB)
      3  rock3     — bottom of Group B (anchored at unconformity AB, not displaced)
      4  rock4     — top of Group B (faulted by F2 only)
      5  rock5     — bottom of Group C (anchored at unconformity BC, not displaced)
      6  rock6     — top of Group C (no faults)
    """
    nx, ny, nz = map(int, grid.resolution)
    xc = np.asarray(grid.gridx)
    zc = np.asarray(grid.gridz)

    X = np.broadcast_to(xc.reshape(nx, 1, 1), (nx, ny, nz))
    Z = np.broadcast_to(zc.reshape(1, 1, nz), (nx, ny, nz))

    # Cumulative throw per group
    throw_A = np.where(X >= _F1_X, _THROW1, 0.0) + np.where(X >= _F2_X, _THROW2, 0.0)
    throw_B = np.where(X >= _F2_X, _THROW2, 0.0)

    z1 = _Z_ROCK1 + throw_A   # base of rock1 in Group A zone
    z2 = _Z_ROCK2 + throw_A   # base of rock2 in Group A zone
    z4 = _Z_ROCK4 + throw_B   # base of rock4 in Group B zone

    expected = np.zeros((nx, ny, nz), dtype=int)   # 0 = basement

    # Group A zone: 0 <= Z < _Z_UNC_AB
    expected[(Z >= z1) & (Z < z2) & (Z < _Z_UNC_AB)] = 1   # rock1
    expected[(Z >= z2) & (Z < _Z_UNC_AB)]             = 2   # rock2

    # Group B zone: _Z_UNC_AB <= Z < _Z_UNC_BC
    expected[(Z >= _Z_UNC_AB) & (Z < z4)]             = 3   # rock3
    expected[(Z >= z4) & (Z < _Z_UNC_BC)]             = 4   # rock4

    # Group C zone: _Z_UNC_BC <= Z
    expected[(Z >= _Z_UNC_BC) & (Z < _Z_ROCK6)]       = 5   # rock5
    expected[Z >= _Z_ROCK6]                            = 6   # rock6

    return expected


# ---------------------------------------------------------------------------
# Acceptance thresholds and method list
# ---------------------------------------------------------------------------

_MIN_INTERIOR_MATCH: dict[InterpolationMethod, float] = {
    InterpolationMethod.RADIAL_BASIS_FUNCTION: 0.95,
    InterpolationMethod.FINITE_DIFFERENCES:    0.95,
    InterpolationMethod.UNIVERSAL_COKRIGING:   0.90,
    InterpolationMethod.UNIVERSAL_KRIGING:     0.90,

    InterpolationMethod.ORDINARY_KRIGING:      0.75,
}

# GeoINR is non-deterministic; OK excluded due to singular matrix risk with
# small cropped domains when 2 faults are active simultaneously
ALL_METHODS = [
    m for m in InterpolationMethod
    if m not in {
        InterpolationMethod.GEOINR,            # non-deterministic
        InterpolationMethod.ORDINARY_KRIGING,  # singular matrix risk with small cropped domains
    }
]


# ---------------------------------------------------------------------------
# Boundary mask helper
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
def test_multi_fault_with_differential_activity(method, plot_mode):
    """
    Model-9-inspired test: 2 faults with differential fault activity across 3 groups.

      F1 (x=1/3, throw=0.10): youngest_affected = Group A (idx=2) — oldest group only
      F2 (x=2/3, throw=0.08): youngest_affected = Group B (idx=1) — A and B affected

    Fault-activity logic:
      Group C (idx=0): F1 inactive (0<2), F2 inactive (0<1) → all domains merged → no offset
      Group B (idx=1): F1 inactive (1<2), F2 active (1>=1) → left+mid merged, right separate
      Group A (idx=2): F1 active  (2>=2), F2 active (2>=1) → full 3-domain splitting

    Checks:
      1. Domain map has exactly 3 domains (2 faults → 3 regions).
      2. Interior voxels match analytical lithology above the per-method threshold.
      3. All 7 lithology IDs (0–6) are present.
      4. Fault-activity sanity: offsets are present/absent as dictated by fault activity.
    """
    pytest.importorskip("gempy")

    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(24, 18, 20))

    data_elements, data_faults = _build_input_data()

    # --- Build and compute fault frame ---
    fault_frame = gf.build_fault_frame(data_faults, grid)
    fault_model_result = gf.compute_fault_domains(fault_frame)

    dm = fault_frame.domain_map
    assert dm is not None, "Fault domain map not set after compute_fault_domains"
    assert dm.shape == tuple(grid.resolution)

    domain_ids = np.unique(dm)
    assert len(domain_ids) == 3, f"Expected 3 fault domains (2 faults), got {domain_ids}"

    # Verify each x-third has a distinct majority domain ID
    nx = int(grid.resolution[0])
    d_left  = int(np.bincount(dm[: nx // 3].ravel()).argmax())
    d_mid   = int(np.bincount(dm[nx // 3: 2 * nx // 3].ravel()).argmax())
    d_right = int(np.bincount(dm[2 * nx // 3 :].ravel()).argmax())
    assert d_left != d_mid,   "Left and mid thirds should be in different fault domains"
    assert d_mid  != d_right, "Mid and right thirds should be in different fault domains"
    assert d_left != d_right, "Left and right thirds should be in different fault domains"

    # --- Build structural frame and set differential fault activity ---
    frame = general.build_structural_frame(
        input_data_elements=data_elements,
        grid=grid,
        fault_model_results=fault_model_result,
    )
    frame.set_fault_activity_by_group("F1", "A")   # F1 youngest_affected = Group A (idx=2)
    frame.set_fault_activity_by_group("F2", "B")   # F2 youngest_affected = Group B (idx=1)

    for g in frame.structural_groups:
        g.set_interpolation_method(method)

    # --- Run full pipeline ---
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

    # --- Presence check: all 7 lithology IDs must appear ---
    for lith_id in range(7):   # 0=basement, 1=rock1, ..., 6=rock6
        assert (lith == lith_id).any(), f"Lithology ID {lith_id} not found in model output"

    # --- Fault-activity sanity checks ---
    # Sample one x-column per zone at mid-y to verify offset presence/absence
    ny = int(grid.resolution[1])
    mid_y   = ny // 2
    x_left  = nx // 6       # well inside x < F1 zone  (~0.08)
    x_mid   = nx // 2       # well inside F1 <= x < F2 (~0.50)
    x_right = 5 * nx // 6   # well inside x >= F2 zone (~0.83)

    def _first_idx(col: np.ndarray, val: int):
        idx = np.where(col == val)[0]
        return int(idx[0]) if idx.size else None

    col_left  = lith[x_left,  mid_y, :]
    col_mid   = lith[x_mid,   mid_y, :]
    col_right = lith[x_right, mid_y, :]

    # Group C (rock6, id=6): both faults inactive → no offset in any column
    rock6_left  = _first_idx(col_left,  6)
    rock6_mid   = _first_idx(col_mid,   6)
    rock6_right = _first_idx(col_right, 6)
    if rock6_left is not None and rock6_mid is not None:
        assert abs(rock6_left - rock6_mid) <= 1, (
            f"[{method.name}] Group C (F1+F2 inactive): unexpected offset between "
            f"left ({rock6_left}) and mid ({rock6_mid}) for rock6"
        )
    if rock6_mid is not None and rock6_right is not None:
        assert abs(rock6_mid - rock6_right) <= 1, (
            f"[{method.name}] Group C (F1+F2 inactive): unexpected offset between "
            f"mid ({rock6_mid}) and right ({rock6_right}) for rock6"
        )

    # Group B (rock4, id=4): F1 inactive → left ≈ mid; F2 active → right > mid
    rock4_left  = _first_idx(col_left,  4)
    rock4_mid   = _first_idx(col_mid,   4)
    rock4_right = _first_idx(col_right, 4)
    if rock4_left is not None and rock4_mid is not None:
        assert abs(rock4_left - rock4_mid) <= 1, (
            f"[{method.name}] Group B (F1 inactive): unexpected offset between "
            f"left ({rock4_left}) and mid ({rock4_mid}) for rock4"
        )
    if rock4_mid is not None and rock4_right is not None:
        assert rock4_right > rock4_mid, (
            f"[{method.name}] Group B (F2 active): rock4 should appear at higher z-index "
            f"on right ({rock4_right}) vs mid ({rock4_mid})"
        )

    # Group A (rock2, id=2): F1 active → mid > left; F2 active → right > mid
    rock2_left  = _first_idx(col_left,  2)
    rock2_mid   = _first_idx(col_mid,   2)
    rock2_right = _first_idx(col_right, 2)
    if rock2_left is not None and rock2_mid is not None:
        assert rock2_mid > rock2_left, (
            f"[{method.name}] Group A (F1 active): rock2 should appear at higher z-index "
            f"in mid ({rock2_mid}) vs left ({rock2_left})"
        )
    if rock2_mid is not None and rock2_right is not None:
        assert rock2_right > rock2_mid, (
            f"[{method.name}] Group A (F2 active): rock2 should appear at higher z-index "
            f"on right ({rock2_right}) vs mid ({rock2_mid})"
        )
