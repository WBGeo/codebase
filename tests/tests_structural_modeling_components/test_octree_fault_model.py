"""
Fault modeling on an OctreeGrid: fault refinement, domain ids, domain pairs,
separated domains and the cross-cut check, compared with the regular grid at the
finest resolution.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from pydantic_core import to_jsonable_python

import core.structural_modeling_components.general_faults as gf
from core.object_components import FaultModelResults, InputData_FaultElements
from core.structural_modeling_components.general_faults_octree import _weighted_median
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeGrid
from core.structural_modeling_components.structural_objects.structural_objects import FaultMeshType


# -----------------------------------------------------------------------------
# Input: two faults that don't cross inside the extent
#   F1 dips about 68 degrees: x = 0.27 + 0.4 z (off the cell lattice, so no cell
#   center lies exactly on it and float32 rounding can't decide a side)
#   F2 is vertical at x = 0.85
# -----------------------------------------------------------------------------

def _fault_rows(name, x_of_z, normal):
    rows, oris = [], []
    for y in np.linspace(0.05, 0.95, 5):
        for z in np.linspace(0.05, 0.95, 6):
            x = x_of_z(z)
            rows.append(dict(X=x, Y=y, Z=z, formation=name))
            oris.append(dict(X=x, Y=y, Z=z, G_x=normal[0], G_y=normal[1], G_z=normal[2], formation=name))
    return rows, oris


def two_faults_input(f2_x_of_z=lambda z: 0.85, f2_normal=(1.0, 0.0, 0.0)) -> InputData_FaultElements:
    n1 = np.array([1.0, 0.0, -0.4]) / np.linalg.norm([1.0, 0.0, -0.4])
    r1, o1 = _fault_rows("F1", lambda z: 0.27 + 0.4 * z, n1)
    r2, o2 = _fault_rows("F2", f2_x_of_z, f2_normal)
    return InputData_FaultElements(
        name="two_faults", fault_names=["F1", "F2"],
        fault_surface_points=pd.DataFrame(r1 + r2), fault_orientations=pd.DataFrame(o1 + o2),
    )


def octree_grid() -> OctreeGrid:
    return OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(3, 2, 2), max_level=3)


def compute(data, grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gf.compute_fault_domains(gf.build_fault_frame(data, grid))


def model9_faults_input() -> InputData_FaultElements:
    """The two vertical faults of the model 9 integration test, at x = 1/3 and x = 2/3: three domains."""
    r1, o1 = _fault_rows("F1", lambda z: 1.0 / 3.0, (1.0, 0.0, 0.0))
    r2, o2 = _fault_rows("F2", lambda z: 2.0 / 3.0, (1.0, 0.0, 0.0))
    return InputData_FaultElements(
        name="model9_faults", fault_names=["F1", "F2"],
        fault_surface_points=pd.DataFrame(r1 + r2), fault_orientations=pd.DataFrame(o1 + o2),
    )


@pytest.fixture(scope="module", params=[two_faults_input, model9_faults_input], ids=["dipping", "model9"])
def results(request) -> tuple[FaultModelResults, FaultModelResults]:
    grid = octree_grid()
    data = request.param()
    return compute(data, grid), compute(data, grid.equivalent_regular_grid())


# -----------------------------------------------------------------------------
# Octree vs. regular grid at the finest resolution
# -----------------------------------------------------------------------------

def test_fault_octree_matches_the_regular_grid(results):
    octree, regular = results[0].fault_frame, results[1].fault_frame

    assert len(octree.cells) < np.prod(octree.grid.finest_resolution)
    octree.check_cell_arrays()
    np.testing.assert_array_equal(octree.to_regular(octree.domain_map), regular.domain_map)
    assert sorted(octree.domain_masks) == sorted(regular.domain_masks)
    for oct_fault, reg_fault in zip(octree.fault_elements, regular.fault_elements):
        assert oct_fault.name == reg_fault.name
        np.testing.assert_array_equal(octree.to_regular(oct_fault.mask), reg_fault.mask)
        assert oct_fault.scalar_value == pytest.approx(reg_fault.scalar_value, abs=1e-5)
        assert oct_fault.domain_pairs == reg_fault.domain_pairs
        assert oct_fault.separated_domains == reg_fault.separated_domains


def test_fault_fields_at_max_level_cells_equal_the_regular_grid(results):
    octree, regular = results[0].fault_frame, results[1].fault_frame
    fine = octree.cells.level == octree.grid.max_level
    i, j, k = octree.cells.ijk[fine].T
    for oct_fault, reg_fault in zip(octree.fault_elements, regular.fault_elements):
        # GemPy computes in float32; the dense and the custom grid round differently.
        np.testing.assert_allclose(oct_fault.scalar_field[fine], reg_fault.scalar_field[i, j, k], atol=1e-4)


def test_coarse_fault_cells_lie_in_one_domain(results):
    """Every cell above max level covers voxels of a single regular-grid domain (D2: children inherit)."""
    octree, regular = results[0].fault_frame, results[1].fault_frame
    grid = octree.grid
    coarse = np.flatnonzero(octree.cells.level < grid.max_level)
    assert coarse.size
    for cell in coarse:
        n = 2 ** (grid.max_level - int(octree.cells.level[cell]))
        i, j, k = octree.cells.ijk[cell].astype(int) * n
        block = regular.domain_map[i:i + n, j:j + n, k:k + n]
        assert np.all(block == octree.domain_map[cell])


def test_fault_octree_has_the_unmasked_and_extended_meshes(results):
    octree = results[0].fault_frame
    for fault in octree.fault_elements:
        assert fault.extended_scalar_field is None
        assert set(fault.vertices) == {FaultMeshType.UNMASKED, FaultMeshType.EXTENDED}
        assert len(fault.vertices[FaultMeshType.UNMASKED]) > 0


def test_fault_result_survives_the_runner_round_trip(results):
    octree = results[0]
    restored = FaultModelResults(**json.loads(json.dumps(to_jsonable_python(octree))))
    frame = restored.fault_frame
    assert isinstance(frame.grid, OctreeGrid)
    frame.check_cell_arrays()
    np.testing.assert_array_equal(frame.domain_map, octree.fault_frame.domain_map)
    for restored_fault, fault in zip(frame.fault_elements, octree.fault_frame.fault_elements):
        assert restored_fault.domain_pairs == fault.domain_pairs
        assert restored_fault.separated_domains == fault.separated_domains


@pytest.mark.parametrize("grid", [octree_grid(), octree_grid().equivalent_regular_grid()], ids=["octree", "regular"])
def test_crossing_faults_are_flagged_and_kept_for_inspection(grid):
    """Both grids: no domains, but fields and fault meshes; structural modeling refuses the result."""
    import core.structural_modeling_components.general as general
    from test_octree_structural_model import unconformity_input

    # F2 dips the other way and crosses F1 near z = 0.4.
    n2 = np.array([1.0, 0.0, 0.5]) / np.linalg.norm([1.0, 0.0, 0.5])
    crossing = two_faults_input(f2_x_of_z=lambda z: 0.6 - 0.5 * z, f2_normal=n2)
    result = compute(crossing, grid)
    assert result.cross_cutting is True
    frame = result.fault_frame
    assert frame.domain_map is None and not frame.domain_masks
    for fault in frame.fault_elements:
        assert fault.domain_pairs is None and fault.separated_domains is None
        assert fault.scalar_field is not None
        assert len(fault.vertices[FaultMeshType.UNMASKED]) > 0
    with pytest.raises(ValueError, match="cross-cutting"):
        general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid,
                                       fault_model_results=result)


def test_faults_that_do_not_cross_are_not_flagged(results):
    assert results[0].cross_cutting is False and results[1].cross_cutting is False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def test_split_into_domains_gives_the_same_ids_on_any_array_shape():
    rng = np.random.default_rng(0)
    masks = [rng.random((5, 4, 3)) > 0.5 for _ in range(3)]
    volume = gf.split_into_domains(masks)
    flat = gf.split_into_domains([m.ravel() for m in masks])
    np.testing.assert_array_equal(volume.ravel(), flat)


def test_weighted_median_counts_each_value_by_its_weight():
    assert _weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])) == 2.0
    assert _weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 8.0])) == 3.0
    assert _weighted_median(np.array([5.0, np.nan, 1.0]), np.array([1.0, 100.0, 3.0])) == 1.0
    assert _weighted_median(np.array([np.nan]), np.array([1.0])) == 0.0
