from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general  # type: ignore
import core.structural_modeling_components.general_faults as gf  # type: ignore

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod


def _near_boundary_mask(expected_lith: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """Voxels within `radius` of any lith boundary (6-neighborhood)."""
    b = np.zeros_like(expected_lith, dtype=bool)

    b[1:, :, :] |= expected_lith[1:, :, :] != expected_lith[:-1, :, :]
    b[:-1, :, :] |= expected_lith[:-1, :, :] != expected_lith[1:, :, :]  # same but explicit
    b[:, 1:, :] |= expected_lith[:, 1:, :] != expected_lith[:, :-1, :]
    b[:, :-1, :] |= expected_lith[:, :-1, :] != expected_lith[:, 1:, :]
    b[:, :, 1:] |= expected_lith[:, :, 1:] != expected_lith[:, :, :-1]
    b[:, :, :-1] |= expected_lith[:, :, :-1] != expected_lith[:, :, 1:]

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


def _make_simple_fault_inputs(
        *,
        throw: float = 0.10,
        z_rock1_left: float = 0.30,
        z_rock2_left: float = 0.60,
) -> tuple[InputData_StructuralElements, InputData_FaultElements, dict[str, float]]:
    """
    One group: Series -> (rock2 youngest, rock1 oldest).
    One vertical fault at x=0.5 splitting into 2 domains.

    We encode throw by providing surface points on both sides:
      left side: rock1 at z_rock1_left, rock2 at z_rock2_left
      right side: shifted up by `throw`
    """
    mapping_object = {"Series": ("rock2", "rock1")}

    # --- group surface points (provide both sides explicitly) ---
    # Dense points help many interpolators.
    ys = np.linspace(0.1, 0.9, 5)
    xs_left = [0.25]
    xs_right = [0.75]

    sp_rows = []
    ori_rows = []

    def add_group_surface(name: str, z: float, xvals):
        for x in xvals:
            for y in ys:
                sp_rows.append({"X": float(x), "Y": float(y), "Z": float(z), "formation": name})
                # simple upward gradient
                ori_rows.append({"X": float(x), "Y": float(y), "Z": float(z),
                                 "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "formation": name})

    # left domain horizons
    add_group_surface("rock1", z_rock1_left, xs_left)
    add_group_surface("rock2", z_rock2_left, xs_left)

    # right domain horizons (shifted)
    add_group_surface("rock1", z_rock1_left + throw, xs_right)
    add_group_surface("rock2", z_rock2_left + throw, xs_right)

    data_elements = InputData_StructuralElements(
        name="toy_faulted_layers",
        mapping_object=mapping_object,
        surface_points=pd.DataFrame(sp_rows),
        orientations=pd.DataFrame(ori_rows),
    )

    # --- fault points: vertical plane x=0.5 ---
    # surface points along y-z (line of points on plane)
    zs = np.linspace(0.1, 0.9, 5)
    f_sp = []
    f_ori = []

    for y in ys:
        for z in zs:
            f_sp.append({"X": 0.5, "Y": float(y), "Z": float(z), "formation": "F1"})
            # normal along +x for a vertical plane
            f_ori.append({"X": 0.5, "Y": float(y), "Z": float(z),
                          "G_x": 1.0, "G_y": 0.0, "G_z": 0.0, "formation": "F1"})

    fault_elements = InputData_FaultElements(
        name="toy_faults",
        fault_names=["F1"],
        fault_surface_points=pd.DataFrame(f_sp),
        fault_orientations=pd.DataFrame(f_ori),
    )

    params = {"throw": throw, "z1_left": z_rock1_left, "z2_left": z_rock2_left}
    return data_elements, fault_elements, params


def _expected_lithology(grid: RegularGrid, *, throw: float, z1_left: float, z2_left: float) -> np.ndarray:
    """
    Analytical expected lithology with basement (0) below rock1.

    IDs (your framework):
      0 basement
      1 rock1
      2 rock2

    Fault at x=0.5:
      left side uses (z1_left, z2_left)
      right side uses (z1_left+throw, z2_left+throw)
    """
    nx, ny, nz = map(int, grid.resolution)
    xc = np.asarray(grid.gridx)
    zc = np.asarray(grid.gridz)

    X = np.broadcast_to(xc.reshape(nx, 1, 1), (nx, ny, nz))
    Z = np.broadcast_to(zc.reshape(1, 1, nz), (nx, ny, nz))

    z1 = np.where(X < 0.5, z1_left, z1_left + throw)
    z2 = np.where(X < 0.5, z2_left, z2_left + throw)

    expected = np.zeros((nx, ny, nz), dtype=int)  # basement
    expected[(Z >= z1) & (Z < z2)] = 1  # rock1
    expected[Z >= z2] = 2  # rock2
    return expected


def _majority_domain(domain_map: np.ndarray, mask: np.ndarray) -> int:
    vals, counts = np.unique(domain_map[mask], return_counts=True)
    return int(vals[np.argmax(counts)])


ALL_GROUP_METHODS = [
    m for m in InterpolationMethod
    if m not in {
        InterpolationMethod.GEOINR,  # not deterministic
        InterpolationMethod.ORDINARY_KRIGING  # singular matrix
    }
]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("method", ALL_GROUP_METHODS)
def test_single_vertical_fault_two_layers_with_offset(method, plot_mode):
    # UCK uses GemPy in your project
    if method == InterpolationMethod.UNIVERSAL_COKRIGING:
        pytest.importorskip("gempy")
    # Fault UCK path also needs GemPy
    pytest.importorskip("gempy")

    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(20, 15, 10))

    data_elements, fault_elements, p = _make_simple_fault_inputs(throw=0.10)

    # Build fault frame + compute domains
    fault_frame = gf.build_fault_frame(fault_elements, grid)
    gf.compute_fault_domains(fault_frame)

    fault_frame.plot_fault_domain_section()

    dm = fault_frame.domain_map

    print(dm.shape)

    assert dm is not None
    assert dm.shape == tuple(grid.resolution)

    # Domain map should have two domains (left/right of x=0.5)
    domain_ids = np.unique(dm)
    assert len(domain_ids) == 2, f"Expected 2 fault domains, got {domain_ids}"

    # Rough geometric validation: majority domain differs on left vs right halves
    nx, ny, nz = map(int, grid.resolution)
    left_mask = np.zeros(dm.shape, dtype=bool);
    left_mask[: nx // 2, :, :] = True
    right_mask = ~left_mask
    left_dom = _majority_domain(dm, left_mask)
    right_dom = _majority_domain(dm, right_mask)
    assert left_dom != right_dom, "Expected different domain IDs on left and right halves"

    # Build structural frame
    frame = general.build_structural_frame(input_data_elements=data_elements, grid=grid)

    # Make sure the group method is the one under test
    for g in frame.structural_groups:
        g.set_interpolation_method(method)

    # CRITICAL: ensure domains are not merged away.
    # For a single group at idx=0, set fault active at 0 so interpolation stays split.
    # frame.fault_activity = {"F1": 0}

    # Run the pipeline steps explicitly (avoids any signature ambiguity)
    general.run_interpolation_with_fault_domains(frame=frame, fault_frame=fault_frame, crop_to_domain=True)
    general.set_scalar_masks_per_domain(frame)
    lith = general.compute_lithology_block_with_domains(frame)

    assert lith is not None
    assert lith.shape == tuple(grid.resolution)
    assert (lith == 0).any(), "Basement (0) should exist below the oldest horizon"

    expected = _expected_lithology(grid, throw=p["throw"], z1_left=p["z1_left"], z2_left=p["z2_left"])

    # Optional plotting for debugging
    from core.visualization_components import plot_structural_model_2D  # type: ignore
    if plot_mode["always"]:
        plot_structural_model_2D(frame)
        # fault section view (optional)
        # fault_frame.plot_fault_domain_section(axis="y", index=grid.resolution[1] // 2)

    # Allow 1-voxel offset near boundaries (interfaces + fault plane)
    near = _near_boundary_mask(expected, radius=1)
    ok_region = ~near
    bad_far = (lith != expected) & ok_region

    try:
        assert not bad_far.any(), f"{int(bad_far.sum())} mismatched voxels >1 voxel away from boundaries"
    except AssertionError:
        if plot_mode["on_fail"]:
            plot_structural_model_2D(frame)
        raise

    # Extra “offset exists” sanity check:
    # Compare the z-index of the rock1->rock2 transition on left vs right at a mid Y.
    mid_y = ny // 2
    left_x = nx // 4
    right_x = (3 * nx) // 4

    col_left = lith[left_x, mid_y, :]
    col_right = lith[right_x, mid_y, :]

    # transition index where lith becomes rock2 (2); if none, fail
    def first_idx(arr, val):
        idx = np.where(arr == val)[0]
        return int(idx[0]) if idx.size else None

    idx_left = first_idx(col_left, 2)
    idx_right = first_idx(col_right, 2)
    assert idx_left is not None and idx_right is not None

    # Right side should be "upthrown": transition happens at higher z-index (larger index)
    assert idx_right > idx_left
