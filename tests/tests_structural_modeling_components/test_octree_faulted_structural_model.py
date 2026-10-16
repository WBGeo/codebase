"""
Structural modeling with faults on an OctreeGrid: domain-aware refinement and
coarsening, one fit per group and merged domain component, and the full pipeline
compared with the regular grid at the finest resolution (models 2 and 9 of the
integration tests).
"""

from __future__ import annotations

import functools
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic_core import to_jsonable_python

import core.structural_modeling_components.general as general
import core.structural_modeling_components.general_faults as gf
import core.structural_modeling_components.general_octree as general_octree
from core.object_components import StructuralModelResults
from core.structural_modeling_components.structural_modeling_utility.octree_refinement import (
    coarsen_cells,
    refine_cells,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import CellBalance, OctreeGrid
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod
from core.structural_modeling_components.structural_workbench_components import pre_check_fault_model_grid

pytest.importorskip("gempy")  # faults are fitted with GemPy


@functools.lru_cache(maxsize=None)
def _integration_module(name: str):
    """The integration tests are no package; load one to reuse its input data."""
    path = Path(__file__).parent / "integration" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# name -> (integration test module, fault activity as (fault, youngest affected group))
MODELS = {
    "model2": ("test_model2_fault_with_unconformity_analytical", []),
    "model2_fault_below_unconformity": ("test_model2_fault_with_unconformity_analytical", [("F1", "LowerSeries")]),
    "model9": ("test_model9_multi_fault_analytical", [("F1", "A"), ("F2", "B")]),
}


def octree_grid() -> OctreeGrid:
    return OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(3, 2, 3), max_level=3)


def compute(model: str, grid, method: InterpolationMethod) -> StructuralModelResults:
    module_name, activity = MODELS[model]
    elements, faults = _integration_module(module_name)._build_input_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fault_result = gf.compute_fault_domains(gf.build_fault_frame(faults, grid))
        frame = general.build_structural_frame(
            input_data_elements=elements, grid=grid, fault_model_results=fault_result)
        for fault, group in activity:
            frame.set_fault_activity_by_group(fault, group)
        for group in frame.structural_groups:
            group.set_interpolation_method(method)
        return general.compute_structural_model(frame, extract_meshes=False, verbose=False)


@pytest.fixture(scope="module", params=[
    ("model9", InterpolationMethod.RADIAL_BASIS_FUNCTION),
    ("model9", InterpolationMethod.UNIVERSAL_KRIGING),
    ("model2", InterpolationMethod.RADIAL_BASIS_FUNCTION),
    ("model2", InterpolationMethod.UNIVERSAL_COKRIGING),
    ("model2_fault_below_unconformity", InterpolationMethod.RADIAL_BASIS_FUNCTION),
], ids=lambda p: f"{p[0]}-{p[1].name}")
def results(request) -> tuple[StructuralModelResults, StructuralModelResults]:
    model, method = request.param
    grid = octree_grid()
    return compute(model, grid, method), compute(model, grid.equivalent_regular_grid(), method)


# -----------------------------------------------------------------------------
# Full pipeline: octree vs. regular grid at the finest resolution
# -----------------------------------------------------------------------------

def test_faulted_octree_result_matches_the_regular_grid(results):
    octree, regular = results[0].structural_frame, results[1].structural_frame

    assert len(octree.cells) < np.prod(octree.grid.finest_resolution)
    octree.check_cell_arrays()
    np.testing.assert_array_equal(octree.to_regular(octree.lith_block), regular.lith_block)
    # The exact domains are the fault frame's, on its own cells.
    np.testing.assert_array_equal(octree.fault_frame.to_regular(octree.fault_frame.domain_map),
                                  regular.cell_domain_ids())
    for oct_group, reg_group in zip(octree.structural_groups, regular.structural_groups):
        np.testing.assert_array_equal(octree.to_regular(oct_group.mask), reg_group.mask)
        for oct_elem, reg_elem in zip(oct_group.structural_elements, reg_group.structural_elements):
            assert oct_elem.scalar_value == reg_elem.scalar_value
            assert oct_elem.id == reg_elem.id


def test_scalar_fields_at_max_level_cells_equal_the_regular_grid(results):
    octree, regular = results[0].structural_frame, results[1].structural_frame
    fine = octree.cells.level == octree.grid.max_level
    i, j, k = octree.cells.ijk[fine].T
    for oct_group, reg_group in zip(octree.structural_groups, regular.structural_groups):
        # GemPy computes in float32; the other interpolators agree to rounding.
        np.testing.assert_allclose(oct_group.scalar_field[fine], reg_group.scalar_field[i, j, k], atol=1e-4)


def test_no_cell_spans_a_fault_that_affects_the_group_present(results):
    """Cells merge across a fault only where the group present there ignores it."""
    octree, regular = results[0].structural_frame, results[1].structural_frame
    grid = octree.grid
    regular_domains = regular.cell_domain_ids()
    group_of_id = {e.id: g for g, group in enumerate(regular.structural_groups) for e in group.structural_elements}
    oldest = len(regular.structural_groups) - 1
    for cell in np.flatnonzero(octree.cells.level < grid.max_level):
        n = 2 ** (grid.max_level - int(octree.cells.level[cell]))
        i, j, k = octree.cells.ijk[cell].astype(int) * n
        lith = regular.lith_block[i:i + n, j:j + n, k:k + n]
        assert np.all(lith == lith.flat[0])  # the lithology doesn't change inside the cell
        group = group_of_id.get(int(lith.flat[0]), oldest)
        components = general_octree._component_of_domains(
            octree, group, regular_domains[i:i + n, j:j + n, k:k + n].ravel())
        assert np.all(components == components[0])


def _largest_face_jump(grid, cells):
    """The largest level difference between face-adjacent finest voxels."""
    level_map = np.empty(grid.finest_resolution, dtype=int)
    for level, ijk in zip(cells.level, cells.ijk):
        n = 2 ** (grid.max_level - int(level))
        i, j, k = np.asarray(ijk, dtype=int) * n
        level_map[i:i + n, j:j + n, k:k + n] = level
    return max(int(np.abs(np.diff(level_map, axis=a)).max(initial=0)) for a in range(3))


def test_balanced_cells_are_graded_and_keep_the_result():
    """With cell_balance = face_2to1 both stored octrees are 2:1 balanced; the model is unchanged."""
    module_name, activity = MODELS["model2_fault_below_unconformity"]
    elements, faults = _integration_module(module_name)._build_input_data()
    results = {}
    for balance in (CellBalance.NONE, CellBalance.FACE_2TO1):
        grid = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(3, 2, 3), max_level=3,
                          cell_balance=balance)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fault_result = gf.compute_fault_domains(gf.build_fault_frame(faults, grid))
            frame = general.build_structural_frame(input_data_elements=elements, grid=grid,
                                                   fault_model_results=fault_result)
            for fault, group in activity:
                frame.set_fault_activity_by_group(fault, group)
            results[balance] = general.compute_structural_model(frame, extract_meshes=False,
                                                                verbose=False).structural_frame
    plain, balanced = results[CellBalance.NONE], results[CellBalance.FACE_2TO1]
    assert _largest_face_jump(plain.grid, plain.cells) > 1  # so balancing has something to do
    assert _largest_face_jump(balanced.grid, balanced.cells) <= 1
    assert _largest_face_jump(balanced.grid, balanced.fault_frame.cells) <= 1
    assert len(balanced.cells) > len(plain.cells)
    balanced.check_cell_arrays()
    np.testing.assert_array_equal(balanced.to_regular(balanced.lith_block), plain.to_regular(plain.lith_block))
    for b_group, p_group in zip(balanced.structural_groups, plain.structural_groups):
        np.testing.assert_array_equal(balanced.to_regular(b_group.mask), plain.to_regular(p_group.mask))
    np.testing.assert_array_equal(balanced.fault_frame.to_regular(balanced.fault_frame.domain_map),
                                  plain.fault_frame.to_regular(plain.fault_frame.domain_map))


def test_cells_merge_across_a_fault_above_the_unconformity_it_does_not_reach():
    """
    Model 2 with the fault offsetting only the lower group, on a grid where the
    fault plane x = 0.5 is no cell boundary: above the unconformity (z = 0.68)
    nothing changes across the fault, so no max-level cells are kept there.
    Below it the fault is active and stays at max level.
    """
    module_name, _ = MODELS["model2"]
    elements, faults = _integration_module(module_name)._build_input_data()
    grid = OctreeGrid(extent=(0.0, 1.05, 0.0, 1.0, 0.0, 1.0), base_resolution=(3, 2, 3), max_level=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fault_result = gf.compute_fault_domains(gf.build_fault_frame(faults, grid))
        frame = general.build_structural_frame(input_data_elements=elements, grid=grid,
                                               fault_model_results=fault_result)
        frame.set_fault_activity_by_group("F1", "LowerSeries")
        octree = general.compute_structural_model(frame, extract_meshes=False, verbose=False).structural_frame
    mins, maxs = grid.cell_bounds(octree.cells.level, octree.cells.ijk)
    on_fault = (mins[:, 0] < 0.5) & (maxs[:, 0] > 0.5)
    # Max-level cells whose parent lies between the rock3 and rock4 bases (z = 0.68, 0.84):
    # nothing changes inside such a parent, so they must have merged.
    fine = octree.cells.level == grid.max_level
    pmins, pmaxs = grid.cell_bounds(grid.max_level - 1, octree.cells.ijk.astype(int) // 2)
    parent_between = (pmins[:, 2] > 0.68) & (pmaxs[:, 2] < 0.84)
    assert not np.any(fine & on_fault & parent_between)
    assert np.any(on_fault & (mins[:, 2] > 0.68) & (maxs[:, 2] < 0.84))  # the merged cells are there
    below = (maxs[:, 2] < 0.2) & on_fault
    assert np.all(octree.cells.level[below] == grid.max_level)
    # The fault model's own cells are at max level along the whole fault.
    fmins, fmaxs = grid.cell_bounds(fault_result.fault_frame.cells.level, fault_result.fault_frame.cells.ijk)
    f_on_fault = (fmins[:, 0] < 0.5) & (fmaxs[:, 0] > 0.5)
    assert np.all(fault_result.fault_frame.cells.level[f_on_fault] == grid.max_level)


def test_faulted_octree_result_survives_the_runner_round_trip(results):
    octree = results[0]
    restored = StructuralModelResults(**json.loads(json.dumps(to_jsonable_python(octree))))
    frame = restored.structural_frame
    frame.check_cell_arrays()
    np.testing.assert_array_equal(frame.lith_block, octree.structural_frame.lith_block)
    np.testing.assert_array_equal(frame.cell_domain_ids(), octree.structural_frame.cell_domain_ids())


# -----------------------------------------------------------------------------
# Grid consistency
# -----------------------------------------------------------------------------

def test_a_fault_model_on_another_grid_is_refused():
    _, faults = _integration_module(MODELS["model2"][0])._build_input_data()
    other = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(3, 2, 3), max_level=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fault_result = gf.compute_fault_domains(gf.build_fault_frame(faults, other))

    with pytest.raises(ValueError, match="different grid"):
        pre_check_fault_model_grid(octree_grid(), fault_result)
    with pytest.raises(ValueError, match="different grid"):
        pre_check_fault_model_grid(octree_grid().equivalent_regular_grid(), fault_result)
    pre_check_fault_model_grid(other, fault_result)
    pre_check_fault_model_grid(octree_grid(), None)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def test_component_extent_pads_only_where_the_component_reaches_the_extent(results):
    """Same fit extents as the regular grid's crop of its edge-padded domain map."""
    octree, regular = results[0].structural_frame, results[1].structural_frame
    domain_ids = general_octree._domain_ids(octree)
    p = general._SURFACE_PADDING_CELLS
    padded_grid = general._build_padded_grid(regular.grid, p)
    padded_map = np.pad(regular.cell_domain_ids(), p, mode="edge")
    for d in domain_ids:
        xs, ys, zs = (np.flatnonzero(np.any(padded_map == d, axis=axes)) for axes in ((1, 2), (0, 2), (0, 1)))
        sub = general.build_subgrid_from_bbox(padded_grid, (xs[0], xs[-1], ys[0], ys[-1], zs[0], zs[-1]))
        np.testing.assert_allclose(general_octree._component_extent(octree, [int(d)], domain_ids),
                                   sub.extent, atol=1e-12)


class _FixedField:
    def __init__(self, scalar_values):
        self.scalar_values = scalar_values


def test_components_that_disagree_on_a_scalar_value_are_refused():
    frame = general.build_structural_frame(
        input_data_elements=_integration_module(MODELS["model2"][0])._build_input_data()[0], grid=octree_grid())
    group = frame.structural_groups[0]
    names = [e.name for e in group.structural_elements]
    collected: dict = {}
    general_octree._collect_scalar_values(group, _FixedField({names[0]: 2.0, names[1]: 1.0}), collected)
    general_octree._collect_scalar_values(group, _FixedField({names[0]: 2.0, names[1]: 1.0}), collected)
    with pytest.raises(ValueError, match="different scalar values"):
        general_octree._collect_scalar_values(group, _FixedField({names[0]: 2.5, names[1]: 1.0}), collected)


def _two_domains(level, ijk):
    """Domain 1 for x >= 0.5 (cells right of the middle face), else 0."""
    grid = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(2, 1, 1), max_level=3)
    return (grid.cell_centers(level, ijk)[:, 0] >= 0.5).astype(np.int64)


def test_a_point_shared_by_two_domains_is_labelled_once_per_domain():
    grid = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(2, 1, 1), max_level=3)
    seen = []

    def label(points, domains):
        seen.append(np.column_stack([points, domains]))
        # Each domain has its own plane: z = 0.3 left, z = 0.7 right.
        return (points[:, 2] > np.where(domains == 0, 0.3, 0.7)).astype(int)

    cells = refine_cells(grid, label, cell_domains=_two_domains)
    first = seen[0]
    on_face = first[:, 0] == 0.5
    # The 4 corners on the shared face x = 0.5 are labelled with both domains.
    assert on_face.sum() == 8 and set(first[on_face, 3]) == {0, 1}
    # Each side is refined toward its own plane only.
    mins, maxs = grid.cell_bounds(cells.level, cells.ijk)
    left = maxs[:, 0] <= 0.5
    assert np.all(cells.level[left & (mins[:, 2] < 0.3) & (maxs[:, 2] > 0.3)] == grid.max_level)
    assert not np.any(left & (mins[:, 2] < 0.7) & (maxs[:, 2] > 0.7) & (cells.level == grid.max_level))


def test_coarsening_never_merges_cells_of_two_domains():
    # One base block split down the middle into two domains; constant labels.
    grid = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(1, 1, 1), max_level=2)
    domains = lambda level, ijk: (grid.cell_centers(level, ijk)[:, 0] >= 0.5).astype(np.int64)  # noqa: E731
    refined = refine_cells(grid, lambda p: (p[:, 0] > 0.5).astype(int))
    coarse = coarsen_cells(grid, refined, lambda p, d: np.zeros(len(p), dtype=int), cell_domains=domains)
    assert len(coarse) == 8 and np.all(coarse.level == 1)
    no_domains = coarsen_cells(grid, refined, lambda p: np.zeros(len(p), dtype=int))
    assert len(no_domains) == 1
