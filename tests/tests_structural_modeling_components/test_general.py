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

    frame.grid = grid
    frame.surface_points = pd.DataFrame(
        {"X": [0.5, 1.5], "Y": [0.5, 1.5], "Z": [0.5, 0.5], "formation": ["e1", "e2"]}
    )
    frame.orientations = None

    # Make sure interpolation context is initialized before set_interpolation_method
    pts = frame.surface_points[["X", "Y", "Z"]].to_numpy()
    g.update_interpolation_context(pts)
    g.set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION)

    # Preallocate scalar field (same convention as your code)
    g.scalar_field = np.zeros(grid.resolution, dtype=float)

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


def test_effective_domain_components_all_faults_active_keeps_domains_separate():
    """When every fault is active for the group, no domains are merged."""
    domain_ids = np.array([0, 1, 2, 3], dtype=int)
    faults = [FakeFault("F1", {(0, 1)}), FakeFault("F2", {(2, 3)})]
    # youngest_idx=0 → is_active = group_idx >= 0 → True for any group_idx >= 0
    fault_activity = {"F1": 0, "F2": 0}

    comps = mod.effective_domain_components_for_group(domain_ids, faults, fault_activity, group_idx=0)
    comps_sorted = sorted([sorted(c) for c in comps])
    assert comps_sorted == [[0], [1], [2], [3]]


def test_effective_domain_components_all_faults_inactive_merges_all():
    """When every fault is inactive for the group, all connected domains collapse into one."""
    domain_ids = np.array([0, 1, 2, 3], dtype=int)
    # F1 links 0-1 and 2-3; F2 bridges 1-2 → transitively connects 0,1,2,3
    faults = [FakeFault("F1", {(0, 1), (2, 3)}), FakeFault("F2", {(1, 2)})]
    fault_activity = {"F1": 5, "F2": 5}

    comps = mod.effective_domain_components_for_group(domain_ids, faults, fault_activity, group_idx=0)
    comps_sorted = sorted([sorted(c) for c in comps])
    assert comps_sorted == [[0, 1, 2, 3]]


def test_effective_domain_components_transitive_merge_via_shared_domain():
    """F1 merges (0,1) and F2 merges (1,2); transitively 0,1,2 form one component."""
    domain_ids = np.array([0, 1, 2], dtype=int)
    faults = [FakeFault("F1", {(0, 1)}), FakeFault("F2", {(1, 2)})]
    fault_activity = {"F1": 5, "F2": 5}  # both inactive

    comps = mod.effective_domain_components_for_group(domain_ids, faults, fault_activity, group_idx=0)
    comps_sorted = sorted([sorted(c) for c in comps])
    assert comps_sorted == [[0, 1, 2]]


def test_effective_domain_components_fault_with_no_pairs_has_no_effect():
    """A fault with an empty domain-pair set must not change the components."""
    domain_ids = np.array([0, 1], dtype=int)
    faults = [FakeFault("F_empty", set())]
    fault_activity = {"F_empty": 5}  # inactive, but no pairs to merge

    comps = mod.effective_domain_components_for_group(domain_ids, faults, fault_activity, group_idx=0)
    comps_sorted = sorted([sorted(c) for c in comps])
    assert comps_sorted == [[0], [1]]


def test_effective_domain_components_single_domain_returns_one_component():
    """With a single domain there is nothing to merge; result is always one component."""
    domain_ids = np.array([0], dtype=int)
    faults = []  # no faults — nothing to merge

    comps = mod.effective_domain_components_for_group(domain_ids, faults=faults, fault_activity={}, group_idx=0)
    assert len(comps) == 1
    assert 0 in comps[0]


def test_effective_domain_components_no_faults_returns_all_separate():
    """With no faults every domain is its own component."""
    domain_ids = np.array([0, 1, 2], dtype=int)

    comps = mod.effective_domain_components_for_group(domain_ids, faults=[], fault_activity={}, group_idx=0)
    comps_sorted = sorted([sorted(c) for c in comps])
    assert comps_sorted == [[0], [1], [2]]


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
    group.scalar_field = np.zeros(grid.resolution, dtype=float)

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

    monkeypatch.setattr(mod, "validate_interpolation_inputs", lambda frame, fault_frame: calls.append("validate"))
    monkeypatch.setattr(mod, "run_interpolation_with_fault_domains", lambda **kw: calls.append("interp"))
    monkeypatch.setattr(mod, "set_scalar_masks_per_domain", lambda fr: calls.append("masks"))
    monkeypatch.setattr(mod, "compute_lithology_block_with_domains", lambda fr: calls.append("lith") or np.zeros(grid.resolution, dtype=int))
    monkeypatch.setattr(mod, "extract_all_meshes_per_domain", lambda frame: calls.append("meshes"))

    res = mod.compute_structural_model(frame, extract_meshes=True, verbose=False)

    assert calls == ["validate", "interp", "masks", "lith", "meshes"]
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
    frame.grid = grid
    with pytest.raises(ValueError, match="no groups"):
        mod.set_scalar_masks_per_domain(frame)


def test_set_scalar_masks_raises_when_group_has_no_scalar_field():
    grid = _make_1d_grid()
    e1 = StructuralElement(name="e1")
    g = StructuralGroup(name="G1", structural_elements=[e1])
    frame = StructuralFrame(structural_groups=[g])
    frame.grid = grid
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
    frame.grid = grid
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
    frame.grid = grid
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
    frame.grid = grid

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
    frame.grid = grid
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
    frame.grid = grid
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
    frame.grid = grid
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
    frame.grid = grid
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
    frame.grid = grid

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
    frame.grid = grid
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
    frame.grid = grid
    # g_skip: no scalar field set
    g_fill.set_scalar_field(np.ones((5, 1, 1)))
    g_fill.set_mask(np.ones((5, 1, 1), dtype=bool))
    e_fill.set_scalar_value(0.5)

    lith = mod.compute_lithology_block_with_domains(frame)
    assert (lith == e_fill.id).all()


# -----------------------------------------------------------------------------
# extract_all_meshes_per_domain
# -----------------------------------------------------------------------------

# Helpers shared across mesh tests

def _make_mesh_frame(grid: RegularGrid, n_groups: int = 1):
    """
    Build a minimal StructuralFrame with `n_groups` groups (each with one element),
    all scalar fields set to all-ones and scalar values set to 0.5.
    Groups are ordered youngest (index 0) → oldest (index n-1).
    """
    groups = []
    elems = []
    for i in range(n_groups):
        e = StructuralElement(name=f"elem{i}")
        e.set_scalar_value(0.5)
        g = StructuralGroup(name=f"G{i}", structural_elements=[e])
        g.set_scalar_field(np.ones(grid.resolution, dtype=float))
        g.set_mask(np.ones(grid.resolution, dtype=bool))
        groups.append(g)
        elems.append(e)
    frame = StructuralFrame(structural_groups=groups)
    frame.grid = grid
    return frame, groups, elems


def _stub_mc_per_element(verts=None, faces=None):
    """Return a monkeypatch replacement for marching_cubes_per_element that records calls."""
    if verts is None:
        verts = np.zeros((1, 3), dtype=float)
    if faces is None:
        faces = np.zeros((1, 3), dtype=int)
    calls = []

    def _fake(sf, sval, spacing, extent, mask=None):
        calls.append({"mask": mask, "sval": sval})
        return verts.copy(), faces.copy()

    return _fake, calls


class _FakeFaultFrame:
    """Minimal fault frame stub with a domain_map."""
    def __init__(self, domain_map: np.ndarray):
        self.domain_map = domain_map
        self.fault_elements = []  # no faults → skip fault-mesh section


def test_extract_meshes_youngest_group_masked_with_all_true_erosion(monkeypatch):
    """Youngest group (i=0): erosion_mask is all-True → mc_mask == domain_mask (all-True for no fault)."""
    grid = make_grid((3, 2, 2))
    frame, groups, elems = _make_mesh_frame(grid, n_groups=1)

    captured_masks = []

    def fake_mc(sf, sval, spacing, extent, mask=None):
        captured_masks.append(mask)
        return np.zeros((1, 3), dtype=float), np.zeros((1, 3), dtype=int)

    monkeypatch.setattr(mod, "marching_cubes_per_element", fake_mc)
    monkeypatch.setattr(mod, "marching_cubes", lambda *a, **kw: (
        [np.zeros((1, 3))], [np.zeros((1, 3), dtype=int)]
    ))

    mod.extract_all_meshes_per_domain(frame)

    # First call is the masked pass; mask should be all True (no erosion, single full domain)
    masked_mask = captured_masks[0]
    assert masked_mask is not None
    assert masked_mask.all(), "Youngest group should receive an all-True combined mask"


def test_extract_meshes_older_group_erosion_mask_is_inverted_prev_mask(monkeypatch):
    """For the second (oldest) group, erosion_mask = ~prev_mask. Verify mc receives ~G0.mask & domain."""
    grid = make_grid((4, 1, 1))
    frame, groups, elems = _make_mesh_frame(grid, n_groups=2)

    # Give G0 a partial mask: first 2 voxels True
    partial_mask = np.array([True, True, False, False]).reshape(4, 1, 1)
    groups[0].set_mask(partial_mask)

    captured = []

    def fake_mc(sf, sval, spacing, extent, mask=None):
        captured.append(mask.copy() if mask is not None else None)
        return np.zeros((1, 3), dtype=float), np.zeros((1, 3), dtype=int)

    monkeypatch.setattr(mod, "marching_cubes_per_element", fake_mc)
    monkeypatch.setattr(mod, "marching_cubes", lambda *a, **kw: (
        [np.zeros((1, 3))], [np.zeros((1, 3), dtype=int)]
    ))

    mod.extract_all_meshes_per_domain(frame)

    # Masked pass: 2 calls (one per group). Index 0=G0, index 1=G1.
    # G1's mask = ~partial_mask & domain_mask(all True) = [False, False, True, True]
    g1_masked_call = captured[1]
    expected = np.array([False, False, True, True]).reshape(4, 1, 1)
    np.testing.assert_array_equal(g1_masked_call, expected)


def test_extract_meshes_group_without_scalar_field_is_skipped(monkeypatch):
    """Groups with no scalar field must be silently skipped (no mc call for that group)."""
    grid = make_grid((3, 2, 2))
    e0 = StructuralElement(name="e0")
    e0.set_scalar_value(0.5)
    g_no_sf = StructuralGroup(name="G_noSF", structural_elements=[e0])
    # intentionally no scalar field set on g_no_sf

    e1 = StructuralElement(name="e1")
    e1.set_scalar_value(0.5)
    g_ok = StructuralGroup(name="G_ok", structural_elements=[e1])
    g_ok.set_scalar_field(np.ones(grid.resolution, dtype=float))
    g_ok.set_mask(np.ones(grid.resolution, dtype=bool))

    frame = StructuralFrame(structural_groups=[g_no_sf, g_ok])
    frame.grid = grid

    call_count = [0]

    def fake_mc(sf, sval, spacing, extent, mask=None):
        call_count[0] += 1
        return np.zeros((1, 3), dtype=float), np.zeros((1, 3), dtype=int)

    monkeypatch.setattr(mod, "marching_cubes_per_element", fake_mc)
    monkeypatch.setattr(mod, "marching_cubes", lambda *a, **kw: (
        [np.zeros((1, 3))], [np.zeros((1, 3), dtype=int)]
    ))

    mod.extract_all_meshes_per_domain(frame)

    # Only g_ok contributes: 1 masked call + 1 unmasked call = 2 total
    assert call_count[0] == 2, f"Expected 2 mc calls (skip g_no_sf), got {call_count[0]}"


def test_extract_meshes_element_without_scalar_value_is_skipped(monkeypatch):
    """Elements with no scalar value must be skipped in the masked pass."""
    grid = make_grid((3, 2, 2))
    e_no_sval = StructuralElement(name="e_no_sval")   # scalar_value intentionally None
    e_ok = StructuralElement(name="e_ok")
    e_ok.set_scalar_value(0.5)

    g = StructuralGroup(name="G1", structural_elements=[e_no_sval, e_ok])
    g.set_scalar_field(np.ones(grid.resolution, dtype=float))
    g.set_mask(np.ones(grid.resolution, dtype=bool))
    frame = StructuralFrame(structural_groups=[g])
    frame.grid = grid

    called_with_svals = []

    def fake_mc(sf, sval, spacing, extent, mask=None):
        called_with_svals.append(sval)
        return np.zeros((1, 3), dtype=float), np.zeros((1, 3), dtype=int)

    monkeypatch.setattr(mod, "marching_cubes_per_element", fake_mc)
    monkeypatch.setattr(mod, "marching_cubes", lambda *a, **kw: (
        [np.zeros((1, 3))], [np.zeros((1, 3), dtype=int)]
    ))

    mod.extract_all_meshes_per_domain(frame)

    # Only e_ok should be processed: 1 masked + 1 unmasked = 2 calls, both with sval=0.5
    assert all(sv == 0.5 for sv in called_with_svals)
    assert len(called_with_svals) == 2


def test_extract_meshes_unmasked_calls_mc_with_none_mask(monkeypatch):
    """Unmasked meshes must be extracted by calling marching_cubes_per_element with mask=None."""
    grid = make_grid((3, 2, 2))
    frame, groups, elems = _make_mesh_frame(grid, n_groups=1)

    none_mask_calls = []

    def fake_mc(sf, sval, spacing, extent, mask=None):
        if mask is None:
            none_mask_calls.append(sval)
        return np.zeros((1, 3), dtype=float), np.zeros((1, 3), dtype=int)

    monkeypatch.setattr(mod, "marching_cubes_per_element", fake_mc)
    monkeypatch.setattr(mod, "marching_cubes", lambda *a, **kw: (
        [np.zeros((1, 3))], [np.zeros((1, 3), dtype=int)]
    ))

    mod.extract_all_meshes_per_domain(frame)

    assert len(none_mask_calls) == 1, "Expected exactly one unmasked mc call"
    verts, faces = elems[0].get_mesh("unmasked")
    assert verts.shape[1] == 3


def test_extract_meshes_multidomain_vertices_are_offset(monkeypatch):
    """When two fault domains exist, domain-1 vertex indices are offset by domain-0's vertex count."""
    grid = make_grid((4, 2, 2))
    frame, groups, elems = _make_mesh_frame(grid, n_groups=1)

    # Two-domain map: left half = domain 0, right half = domain 1
    dom_map = np.zeros(grid.resolution, dtype=int)
    dom_map[2:, :, :] = 1
    frame.fault_frame = _FakeFaultFrame(dom_map)

    domain_call_idx = [0]
    V0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])   # 2 verts for domain 0
    V1 = np.array([[5.0, 5.0, 5.0]])                      # 1 vert for domain 1
    F0 = np.array([[0, 1, 0]], dtype=int)
    F1 = np.array([[0, 0, 0]], dtype=int)

    def fake_mc(sf, sval, spacing, extent, mask=None):
        idx = domain_call_idx[0]
        domain_call_idx[0] += 1
        return (V0.copy(), F0.copy()) if idx == 0 else (V1.copy(), F1.copy())

    monkeypatch.setattr(mod, "marching_cubes_per_element", fake_mc)
    monkeypatch.setattr(mod, "marching_cubes", lambda *a, **kw: (
        [np.zeros((1, 3))], [np.zeros((1, 3), dtype=int)]
    ))

    mod.extract_all_meshes_per_domain(frame)

    combined_verts, combined_faces = elems[0].get_mesh("masked")

    # Vertices: V0 rows then V1 rows
    assert combined_verts.shape == (3, 3)
    np.testing.assert_allclose(combined_verts[:2], V0)
    np.testing.assert_allclose(combined_verts[2], V1[0])

    # Faces: F0 unchanged (offset=0); F1 shifted by +2 (V0 had 2 verts)
    np.testing.assert_array_equal(combined_faces[:1], F0)
    np.testing.assert_array_equal(combined_faces[1:], F1 + 2)


def test_extract_meshes_combined_not_set_without_lith_block(monkeypatch):
    """Combined meshes must not be set when the frame has no lith block."""
    grid = make_grid((3, 2, 2))
    frame, groups, elems = _make_mesh_frame(grid, n_groups=1)
    # frame.lith_block is None by default

    monkeypatch.setattr(mod, "marching_cubes_per_element", lambda *a, **kw: (
        np.zeros((1, 3), dtype=float), np.zeros((1, 3), dtype=int)
    ))
    monkeypatch.setattr(mod, "marching_cubes", lambda *a, **kw: (
        [np.zeros((1, 3))], [np.zeros((1, 3), dtype=int)]
    ))

    mod.extract_all_meshes_per_domain(frame)

    with pytest.raises(KeyError):
        elems[0].get_mesh("combined")


def test_extract_meshes_combined_set_when_lith_block_present(monkeypatch):
    """Combined meshes are stored from lith block when frame.lith_block is available."""
    grid = make_grid((3, 2, 2))
    frame, groups, elems = _make_mesh_frame(grid, n_groups=1)

    elems[0].set_id(1)
    frame.lith_block = np.ones(grid.resolution, dtype=int)

    sentinel_verts = np.array([[9.0, 9.0, 9.0]])
    sentinel_faces = np.array([[0, 0, 0]], dtype=int)

    monkeypatch.setattr(mod, "marching_cubes_per_element", lambda *a, **kw: (
        np.zeros((1, 3), dtype=float), np.zeros((1, 3), dtype=int)
    ))
    monkeypatch.setattr(mod, "marching_cubes", lambda *a, **kw: (
        [sentinel_verts], [sentinel_faces]
    ))

    mod.extract_all_meshes_per_domain(frame)

    combined_verts, combined_faces = elems[0].get_mesh("combined")
    np.testing.assert_array_equal(combined_verts, sentinel_verts)
    np.testing.assert_array_equal(combined_faces, sentinel_faces)


# -----------------------------------------------------------------------------
# validate_interpolation_inputs
# -----------------------------------------------------------------------------

# Duck-typed helpers used only in this section.

class _FakeFaultElem:
    def __init__(self, name: str, pairs: set):
        self.name = name
        self._pairs = pairs

    def get_domain_pairs(self):
        return self._pairs


class _FakeFaultFrameWithFaults:
    """Fault frame stub that exposes both domain_map and fault_elements."""
    def __init__(self, domain_map: np.ndarray, fault_elements: list):
        self.domain_map = domain_map
        self.fault_elements = fault_elements


def _make_validate_frame(
        grid: RegularGrid,
        method: InterpolationMethod,
        *,
        n_elements: int = 2,
        has_orientations: bool = True,
) -> StructuralFrame:
    """Minimal frame wired up for validate_interpolation_inputs."""
    elements = [StructuralElement(name=f"e{i}") for i in range(n_elements)]
    g = StructuralGroup(name="G1", structural_elements=elements)
    frame = StructuralFrame(structural_groups=[g])
    frame.grid = grid

    sp_rows = max(n_elements, 1)
    formations = [e.name for e in elements] if n_elements > 0 else ["orphan"]
    frame.surface_points = pd.DataFrame({
        "X": np.linspace(0.5, 1.5, sp_rows),
        "Y": [0.5] * sp_rows,
        "Z": [0.5] * sp_rows,
        "formation": formations,
    })

    if has_orientations and n_elements > 0:
        frame.orientations = pd.DataFrame({
            "X": [1.0], "Y": [0.5], "Z": [0.5],
            "G_x": [0.0], "G_y": [0.0], "G_z": [1.0],
            "formation": [elements[0].name],
        })
    else:
        frame.orientations = None

    # update_interpolation_context requires ≥2 points; provide dummy points when needed
    ctx_pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    g.update_interpolation_context(ctx_pts)
    g.set_interpolation_method(method)
    g.scalar_field = np.zeros(grid.resolution, dtype=float)
    return frame


def test_validate_passes_for_valid_rbf_group():
    """RBF with 2 elements and surface points — no fault frame — should not raise."""
    grid = make_grid((4, 3, 2))
    frame = _make_validate_frame(grid, InterpolationMethod.RADIAL_BASIS_FUNCTION, has_orientations=False)
    mod.validate_interpolation_inputs(frame=frame, fault_frame=None)  # must not raise


def test_validate_passes_for_valid_fdi_group():
    """FDI with 1+ elements, surface points, and orientations should not raise."""
    grid = make_grid((4, 3, 2))
    frame = _make_validate_frame(grid, InterpolationMethod.FINITE_DIFFERENCES, has_orientations=True)
    mod.validate_interpolation_inputs(frame=frame, fault_frame=None)  # must not raise


def test_validate_fails_when_orientations_required_but_absent():
    """FDI/PLI/UCK/GeoINR group without any orientation data must raise."""
    grid = make_grid((4, 3, 2))
    for method in (
        InterpolationMethod.FINITE_DIFFERENCES,
        # InterpolationMethod.PIECEWISE_LINEAR,  # excluded pending parameter tuning
        InterpolationMethod.UNIVERSAL_COKRIGING,
    ):
        frame = _make_validate_frame(grid, method, has_orientations=False)
        with pytest.raises(ValueError, match="orientation"):
            mod.validate_interpolation_inputs(frame=frame, fault_frame=None)


def test_validate_fails_for_zero_elements():
    """A group with zero structural elements must raise regardless of interpolator."""
    grid = make_grid((4, 3, 2))
    frame = _make_validate_frame(grid, InterpolationMethod.RADIAL_BASIS_FUNCTION, n_elements=0)
    with pytest.raises(ValueError, match="structural element"):
        mod.validate_interpolation_inputs(frame=frame, fault_frame=None)


def test_validate_collects_multiple_violations():
    """All violations across groups are reported together in a single ValueError."""
    grid = make_grid((4, 3, 2))

    _ctx = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])  # ≥2 points required

    # Group 1: FDI, no orientations
    e1 = StructuralElement(name="a")
    g1 = StructuralGroup(name="G1", structural_elements=[e1])
    g1.update_interpolation_context(_ctx)
    g1.set_interpolation_method(InterpolationMethod.FINITE_DIFFERENCES)
    g1.scalar_field = np.zeros(grid.resolution, dtype=float)

    # Group 2: FDI, no orientations (PLI excluded pending parameter tuning)
    e2 = StructuralElement(name="b")
    g2 = StructuralGroup(name="G2", structural_elements=[e2])
    g2.update_interpolation_context(_ctx)
    g2.set_interpolation_method(InterpolationMethod.FINITE_DIFFERENCES)
    g2.scalar_field = np.zeros(grid.resolution, dtype=float)

    frame = StructuralFrame(structural_groups=[g1, g2])
    frame.grid = grid
    frame.surface_points = pd.DataFrame({
        "X": [0.5, 1.5], "Y": [0.5, 0.5], "Z": [0.5, 0.5],
        "formation": ["a", "b"],
    })
    frame.orientations = None

    with pytest.raises(ValueError) as exc_info:
        mod.validate_interpolation_inputs(frame=frame, fault_frame=None)

    msg = str(exc_info.value)
    assert "2 issue" in msg
    assert "G1" in msg
    assert "G2" in msg


def test_validate_fails_when_orientations_absent_in_active_fault_domain():
    """
    FDI group whose orientations exist globally but all land in the wrong fault
    domain component should fail validation (domain-aware check).
    """
    # 4×1×1 grid: voxels 0,1 → domain 0; voxels 2,3 → domain 1
    grid = RegularGrid(extent=(0.0, 4.0, 0.0, 1.0, 0.0, 1.0), resolution=(4, 1, 1))
    dom_map = np.array([0, 0, 1, 1]).reshape(4, 1, 1)

    e = StructuralElement(name="layer")
    g = StructuralGroup(name="G", structural_elements=[e])
    frame = StructuralFrame(structural_groups=[g])
    frame.grid = grid

    # Surface points in domain 1 (x=2.5 and x=3.5)
    frame.surface_points = pd.DataFrame({
        "X": [2.5, 3.5], "Y": [0.5, 0.5], "Z": [0.5, 0.5], "formation": ["layer", "layer"],
    })
    # Orientation in domain 0 (x=0.5) — wrong domain
    frame.orientations = pd.DataFrame({
        "X": [0.5], "Y": [0.5], "Z": [0.5],
        "G_x": [0.0], "G_y": [0.0], "G_z": [1.0],
        "formation": ["layer"],
    })

    g.update_interpolation_context(np.array([[2.5, 0.5, 0.5], [3.5, 0.5, 0.5]]))
    g.set_interpolation_method(InterpolationMethod.FINITE_DIFFERENCES)
    g.scalar_field = np.zeros(grid.resolution, dtype=float)

    # Fault active for this group (youngest_idx=0, group_idx=0 → active → domains stay separate)
    fault_elem = _FakeFaultElem("F1", {(0, 1)})
    fake_ff = _FakeFaultFrameWithFaults(dom_map, [fault_elem])
    frame.fault_activity = {"F1": 0}

    with pytest.raises(ValueError, match="orientation"):
        mod.validate_interpolation_inputs(frame=frame, fault_frame=fake_ff)


def test_validate_passes_when_orientations_present_in_correct_domain():
    """
    Same two-domain setup as above, but orientations ARE in the same domain as
    surface points — validation should pass.
    """
    grid = RegularGrid(extent=(0.0, 4.0, 0.0, 1.0, 0.0, 1.0), resolution=(4, 1, 1))
    dom_map = np.array([0, 0, 1, 1]).reshape(4, 1, 1)

    e = StructuralElement(name="layer")
    g = StructuralGroup(name="G", structural_elements=[e])
    frame = StructuralFrame(structural_groups=[g])
    frame.grid = grid

    frame.surface_points = pd.DataFrame({
        "X": [2.5, 3.5], "Y": [0.5, 0.5], "Z": [0.5, 0.5], "formation": ["layer", "layer"],
    })
    # Orientation also in domain 1 (x=3.5) — correct domain
    frame.orientations = pd.DataFrame({
        "X": [3.5], "Y": [0.5], "Z": [0.5],
        "G_x": [0.0], "G_y": [0.0], "G_z": [1.0],
        "formation": ["layer"],
    })

    g.update_interpolation_context(np.array([[2.5, 0.5, 0.5], [3.5, 0.5, 0.5]]))
    g.set_interpolation_method(InterpolationMethod.FINITE_DIFFERENCES)
    g.scalar_field = np.zeros(grid.resolution, dtype=float)

    fault_elem = _FakeFaultElem("F1", {(0, 1)})
    fake_ff = _FakeFaultFrameWithFaults(dom_map, [fault_elem])
    frame.fault_activity = {"F1": 0}

    mod.validate_interpolation_inputs(frame=frame, fault_frame=fake_ff)  # must not raise
