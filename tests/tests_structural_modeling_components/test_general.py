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
