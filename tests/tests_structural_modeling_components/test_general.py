from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as mod  # type: ignore

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import (
    StructuralElement,
    StructuralGroup,
    StructuralFrame,
    FaultElement,
    FaultFrame,
    InterpolationMethod,
)


# -----------------------------------------------------------------------------
# Helpers to build minimal objects
# -----------------------------------------------------------------------------

def make_grid(resolution=(4, 3, 2)) -> RegularGrid:
    return RegularGrid(extent=(0.0, 4.0, 0.0, 3.0, 0.0, 2.0), resolution=resolution)


def make_frame_one_group(grid: RegularGrid) -> StructuralFrame:
    e1 = StructuralElement(name="e1")
    e2 = StructuralElement(name="e2")
    g = StructuralGroup(name="G1", structural_elements=[e1, e2])
    frame = StructuralFrame(structural_groups=[g])

    frame._grid = grid
    frame._surface_points = pd.DataFrame(
        {"X": [0.5, 1.5], "Y": [0.5, 1.5], "Z": [0.5, 0.5], "formation": ["e1", "e2"]}
    )
    frame._orientations = None

    # Make sure interpolation context is initialized before set_interpolation_method
    pts = frame._surface_points[["X", "Y", "Z"]].to_numpy()
    g.update_interpolation_context(pts)
    g.set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION)

    # Preallocate scalar field (same convention as your code)
    g._scalar_field = np.zeros(grid.resolution, dtype=float)

    return frame


# -----------------------------------------------------------------------------
# compute_domain_bbox_indices
# -----------------------------------------------------------------------------

def test_compute_domain_bbox_indices_none_when_domain_absent():
    domain_map = np.zeros((3, 3, 3), dtype=int)
    assert mod.compute_domain_bbox_indices(domain_map, domain_id=5) is None


def test_compute_domain_bbox_indices_returns_tight_bbox_xyz():
    domain_map = np.zeros((5, 4, 3), dtype=int)
    domain_map[1:3, 2:4, 0:2] = 7  # x=1..2, y=2..3, z=0..1

    bbox = mod.compute_domain_bbox_indices(domain_map, domain_id=7)
    assert bbox == (1, 2, 2, 3, 0, 1)


# -----------------------------------------------------------------------------
# build_subgrid_from_bbox
# -----------------------------------------------------------------------------

def test_build_subgrid_from_bbox_extent_and_resolution():
    grid = RegularGrid(extent=(0.0, 10.0, 0.0, 20.0, 0.0, 30.0), resolution=(10, 10, 10))
    # spacing = (1,2,3)
    bbox = (2, 4, 1, 1, 0, 2)  # x=2..4 (3 cells), y=1..1 (1 cell), z=0..2 (3 cells)

    sub = mod.build_subgrid_from_bbox(grid, bbox)
    assert sub.resolution == (3, 1, 3)
    # extent should span exactly those cells
    assert sub.extent == (
        0.0 + 2 * 1.0,
        0.0 + (4 + 1) * 1.0,
        0.0 + 1 * 2.0,
        0.0 + (1 + 1) * 2.0,
        0.0 + 0 * 3.0,
        0.0 + (2 + 1) * 3.0,
    )


# -----------------------------------------------------------------------------
# assign_domain_ids_to_points
# -----------------------------------------------------------------------------

def test_assign_domain_ids_to_points_empty_df():
    grid = make_grid()
    dm = np.zeros(grid.resolution, dtype=int)
    df = pd.DataFrame(columns=["X", "Y", "Z", "formation"])
    out = mod.assign_domain_ids_to_points(grid, dm, df)
    assert "domain_id" in out.columns
    assert out.empty


def test_assign_domain_ids_to_points_clamps_indices_and_assigns():
    grid = RegularGrid(extent=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0), resolution=(2, 2, 2))
    dm = np.zeros(grid.resolution, dtype=int)
    dm[1, 1, 1] = 9

    # point outside max -> should clamp to last voxel (1,1,1) -> domain 9
    df = pd.DataFrame({"X": [999.0], "Y": [999.0], "Z": [999.0]})
    out = mod.assign_domain_ids_to_points(grid, dm, df)
    assert int(out["domain_id"].iloc[0]) == 9


# -----------------------------------------------------------------------------
# effective_domain_components_for_group
# -----------------------------------------------------------------------------

class FakeFault:
    def __init__(self, name: str, pairs: set[tuple[int, int]]):
        self.name = name
        self._pairs = pairs

    def get_domain_pairs(self):
        return self._pairs


def test_effective_domain_components_merges_across_inactive_faults_only():
    domain_ids = np.array([0, 1, 2, 3], dtype=int)

    # Fault F1 connects (0,1); Fault F2 connects (2,3)
    faults = [
        FakeFault("F1", {(0, 1)}),
        FakeFault("F2", {(2, 3)}),
    ]

    # If group_idx < youngest_idx => inactive => merge across
    fault_activity = {"F1": 5, "F2": 0}

    comps = mod.effective_domain_components_for_group(domain_ids, faults, fault_activity, group_idx=1)
    # group_idx=1:
    # F1: 1>=5? no -> inactive -> merge 0&1
    # F2: 1>=0? yes -> active -> do NOT merge 2&3
    comps_sorted = sorted([sorted(c) for c in comps])
    assert comps_sorted == [[0, 1], [2], [3]]


# -----------------------------------------------------------------------------
# run_interpolation_with_fault_domains (mock-heavy wiring test)
# -----------------------------------------------------------------------------

def test_run_interpolation_with_fault_domains_calls_interpolator_and_sets_scalar_values(monkeypatch):
    grid = make_grid((3, 2, 2))
    frame = make_frame_one_group(grid)
    group = frame.structural_groups[0]
    group.set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION)

    # Domain map: single domain only -> no cropping
    class TmpFF:
        domain_map = np.zeros(grid.resolution, dtype=int)

    captured = {}

    def fake_interp(*, group, grid, group_surface_points_df, group_orientations_points_df):
        # Return field in (Z, Y, X) so transpose(2,1,0) -> (X, Y, Z)
        z, y, x = grid.resolution[2], grid.resolution[1], grid.resolution[0]
        sf = np.arange(x * y * z, dtype=float).reshape((z, y, x))
        captured["grid_res"] = grid.resolution
        captured["n_sp"] = len(group_surface_points_df)
        return sf, {"e1": 1.0, "e2": 2.0}

    monkeypatch.setitem(mod.interpolate_dispatch, InterpolationMethod.RADIAL_BASIS_FUNCTION, fake_interp)

    # Pre-existing scalar field used as "background"
    group._scalar_field = np.zeros(grid.resolution, dtype=float)

    mod.run_interpolation_with_fault_domains(frame=frame, fault_frame=TmpFF(), crop_to_domain=False)

    # scalar values pushed to elements
    assert frame.structural_groups[0].structural_elements[0].scalar_value == 1.0
    assert frame.structural_groups[0].structural_elements[1].scalar_value == 2.0

    # interpolator was called with filtered points
    assert captured["n_sp"] == 2


# -----------------------------------------------------------------------------
# compute_structural_model (mock the big steps; pure orchestration)
# -----------------------------------------------------------------------------

def test_compute_structural_model_orchestrates_steps(monkeypatch):
    grid = make_grid((2, 2, 2))
    frame = make_frame_one_group(grid)

    calls = []

    monkeypatch.setattr(mod, "run_interpolation_with_fault_domains", lambda **kw: calls.append("interp"))
    monkeypatch.setattr(mod, "set_scalar_masks_per_domain", lambda fr: calls.append("masks"))
    monkeypatch.setattr(mod, "compute_lithology_block_with_domains", lambda fr: calls.append("lith") or np.zeros(grid.resolution, dtype=int))
    monkeypatch.setattr(mod, "extract_all_meshes_per_domain", lambda frame: calls.append("meshes"))

    res = mod.compute_structural_model(frame, extract_meshes=True, verbose=False)

    assert calls == ["interp", "masks", "lith", "meshes"]
    assert hasattr(res, "structural_frame")


# -----------------------------------------------------------------------------
# set_scalar_masks_per_domain
# -----------------------------------------------------------------------------

def _make_1d_grid(n: int = 5) -> RegularGrid:
    """1-D grid: n voxels along X, 1 in Y and Z."""
    return RegularGrid(extent=(0.0, float(n), 0.0, 1.0, 0.0, 1.0), resolution=(n, 1, 1))


def _lin_sf(n: int = 5) -> np.ndarray:
    """Scalar field [1, 2, ..., n] shaped (n, 1, 1)."""
    return np.arange(1.0, n + 1.0).reshape(n, 1, 1)


def test_set_scalar_masks_raises_when_no_groups():
    grid = _make_1d_grid()
    frame = StructuralFrame(structural_groups=[])
    frame._grid = grid
    with pytest.raises(ValueError, match="no groups"):
        mod.set_scalar_masks_per_domain(frame)


def test_set_scalar_masks_raises_when_group_has_no_scalar_field():
    grid = _make_1d_grid()
    e1 = StructuralElement(name="e1")
    g = StructuralGroup(name="G1", structural_elements=[e1])
    frame = StructuralFrame(structural_groups=[g])
    frame._grid = grid
    # scalar field intentionally NOT set
    with pytest.raises(ValueError, match="no scalar field"):
        mod.set_scalar_masks_per_domain(frame)


def test_set_scalar_masks_raises_when_oldest_element_has_no_scalar_value():
    grid = _make_1d_grid()
    e_young = StructuralElement(name="e_young")
    e_old = StructuralElement(name="e_old")
    # G0 is NOT the last group, so its oldest element scalar value is required
    g0 = StructuralGroup(name="G0", structural_elements=[e_young, e_old])
    g1 = StructuralGroup(name="G1", structural_elements=[StructuralElement(name="base")])
    frame = StructuralFrame(structural_groups=[g0, g1])
    frame._grid = grid
    g0.set_scalar_field(_lin_sf())
    g1.set_scalar_field(_lin_sf())
    # e_old (structural_elements[-1]) has no scalar value
    with pytest.raises(ValueError, match="no scalar value"):
        mod.set_scalar_masks_per_domain(frame)


def test_set_scalar_masks_single_group_gets_all_true_mask():
    """The only (= last) group always receives an all-True mask."""
    grid = _make_1d_grid()
    e1 = StructuralElement(name="e1")
    g = StructuralGroup(name="G1", structural_elements=[e1])
    frame = StructuralFrame(structural_groups=[g])
    frame._grid = grid
    g.set_scalar_field(_lin_sf())

    mod.set_scalar_masks_per_domain(frame)

    mask = g.get_mask()
    assert mask is not None
    assert mask.shape == (5, 1, 1)
    assert mask.all()


def test_set_scalar_masks_non_last_group_thresholds_at_oldest_element():
    """Non-last group mask = sf >= oldest_element.scalar_value."""
    grid = _make_1d_grid()
    e_young = StructuralElement(name="e_young")
    e_old = StructuralElement(name="e_old")
    # elements ordered youngest → oldest (convention); oldest = structural_elements[-1]
    g0 = StructuralGroup(name="G0", structural_elements=[e_young, e_old])
    g1 = StructuralGroup(name="G1", structural_elements=[StructuralElement(name="base")])
    frame = StructuralFrame(structural_groups=[g0, g1])
    frame._grid = grid

    sf = _lin_sf()  # [1, 2, 3, 4, 5]
    g0.set_scalar_field(sf)
    g1.set_scalar_field(sf.copy())
    e_old.set_scalar_value(3.0)

    mod.set_scalar_masks_per_domain(frame)

    mask_g0 = g0.get_mask()
    expected = np.array([False, False, True, True, True]).reshape(5, 1, 1)
    np.testing.assert_array_equal(mask_g0, expected)
    assert g1.get_mask().all()


# -----------------------------------------------------------------------------
# compute_lithology_block_with_domains
# -----------------------------------------------------------------------------

def test_compute_lithology_returns_correct_shape():
    grid = make_grid((4, 3, 2))
    e1 = StructuralElement(name="e1")
    g = StructuralGroup(name="G1", structural_elements=[e1])
    frame = StructuralFrame(structural_groups=[g])
    frame._grid = grid
    g.set_scalar_field(np.ones((4, 3, 2)))
    g.set_mask(np.ones((4, 3, 2), dtype=bool))
    e1.set_scalar_value(0.5)

    lith = mod.compute_lithology_block_with_domains(frame)
    assert lith.shape == (4, 3, 2)


def test_compute_lithology_assigns_element_ids_automatically():
    grid = _make_1d_grid()
    e1 = StructuralElement(name="e1")
    g = StructuralGroup(name="G1", structural_elements=[e1])
    frame = StructuralFrame(structural_groups=[g])
    frame._grid = grid
    g.set_scalar_field(np.ones((5, 1, 1)))
    g.set_mask(np.ones((5, 1, 1), dtype=bool))
    e1.set_scalar_value(0.5)

    assert e1.id is None
    mod.compute_lithology_block_with_domains(frame)
    assert e1.id is not None


def test_compute_lithology_single_element_fills_above_threshold():
    """Voxels with sf >= sval get the element ID; others stay 0."""
    grid = _make_1d_grid()
    e1 = StructuralElement(name="e1")
    g = StructuralGroup(name="G1", structural_elements=[e1])
    frame = StructuralFrame(structural_groups=[g])
    frame._grid = grid
    g.set_scalar_field(_lin_sf())          # [1, 2, 3, 4, 5]
    g.set_mask(np.ones((5, 1, 1), dtype=bool))
    e1.set_scalar_value(3.0)

    lith = mod.compute_lithology_block_with_domains(frame)
    eid = e1.id
    expected = np.array([0, 0, eid, eid, eid]).reshape(5, 1, 1)
    np.testing.assert_array_equal(lith, expected)


def test_compute_lithology_two_elements_partition_correctly():
    """Younger element (higher threshold) fills first; older fills remaining."""
    grid = _make_1d_grid()
    # Convention: elements listed youngest → oldest
    e_young = StructuralElement(name="e_young")  # sval=4 → fills sf >= 4
    e_old = StructuralElement(name="e_old")       # sval=2 → fills remaining sf >= 2
    g = StructuralGroup(name="G1", structural_elements=[e_young, e_old])
    frame = StructuralFrame(structural_groups=[g])
    frame._grid = grid
    g.set_scalar_field(_lin_sf())          # [1, 2, 3, 4, 5]
    g.set_mask(np.ones((5, 1, 1), dtype=bool))
    e_young.set_scalar_value(4.0)
    e_old.set_scalar_value(2.0)

    lith = mod.compute_lithology_block_with_domains(frame)
    # sf=1 → below e_old threshold → 0
    # sf=2,3 → e_old (not yet claimed by e_young)
    # sf=4,5 → e_young
    expected = np.array([0, e_old.id, e_old.id, e_young.id, e_young.id]).reshape(5, 1, 1)
    np.testing.assert_array_equal(lith, expected)


def test_compute_lithology_younger_group_overwrites_older():
    """groups[0] is youngest and must overwrite groups[1] (oldest) where both are active."""
    grid = _make_1d_grid()
    e_young = StructuralElement(name="e_young")
    e_old = StructuralElement(name="e_old")
    # groups[0] = youngest, groups[1] = oldest
    g_youngest = StructuralGroup(name="G_youngest", structural_elements=[e_young])
    g_oldest = StructuralGroup(name="G_oldest", structural_elements=[e_old])
    frame = StructuralFrame(structural_groups=[g_youngest, g_oldest])
    frame._grid = grid

    sf = _lin_sf()  # [1, 2, 3, 4, 5]
    g_oldest.set_scalar_field(sf)
    g_oldest.set_mask(np.ones((5, 1, 1), dtype=bool))
    e_old.set_scalar_value(1.0)   # fills all voxels (sf >= 1)

    g_youngest.set_scalar_field(sf)
    g_youngest.set_mask(np.ones((5, 1, 1), dtype=bool))
    e_young.set_scalar_value(3.0)  # fills only sf >= 3 (voxels 2, 3, 4)

    lith = mod.compute_lithology_block_with_domains(frame)
    # voxels 0,1 → e_old (oldest fills, youngest doesn't reach)
    # voxels 2,3,4 → e_young (youngest overwrites oldest)
    assert lith.flat[0] == e_old.id
    assert lith.flat[1] == e_old.id
    assert lith.flat[2] == e_young.id
    assert lith.flat[3] == e_young.id
    assert lith.flat[4] == e_young.id


def test_compute_lithology_age_mask_limits_fill():
    """Only voxels inside the group mask receive an element ID."""
    grid = _make_1d_grid()
    e1 = StructuralElement(name="e1")
    g = StructuralGroup(name="G1", structural_elements=[e1])
    frame = StructuralFrame(structural_groups=[g])
    frame._grid = grid
    g.set_scalar_field(_lin_sf())
    e1.set_scalar_value(1.0)  # would fill entire grid without mask
    # Only first 3 voxels are inside the group's age mask
    g.set_mask(np.array([True, True, True, False, False]).reshape(5, 1, 1))

    lith = mod.compute_lithology_block_with_domains(frame)
    eid = e1.id
    expected = np.array([eid, eid, eid, 0, 0]).reshape(5, 1, 1)
    np.testing.assert_array_equal(lith, expected)


def test_compute_lithology_skips_group_with_no_scalar_field():
    """A group with no scalar field is silently skipped; other groups fill normally."""
    grid = _make_1d_grid()
    e_skip = StructuralElement(name="e_skip")
    e_fill = StructuralElement(name="e_fill")
    # groups[0] = youngest (no sf → skipped), groups[1] = oldest (has sf)
    g_skip = StructuralGroup(name="G_skip", structural_elements=[e_skip])
    g_fill = StructuralGroup(name="G_fill", structural_elements=[e_fill])
    frame = StructuralFrame(structural_groups=[g_skip, g_fill])
    frame._grid = grid
    # g_skip: no scalar field set
    g_fill.set_scalar_field(np.ones((5, 1, 1)))
    g_fill.set_mask(np.ones((5, 1, 1), dtype=bool))
    e_fill.set_scalar_value(0.5)

    lith = mod.compute_lithology_block_with_domains(frame)
    assert (lith == e_fill.id).all()
