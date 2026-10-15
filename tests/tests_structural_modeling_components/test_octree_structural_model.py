"""
Octree path of the structural model: RBF fit/evaluate, octree refinement, and the
full pipeline compared with the regular grid at the equivalent finest resolution.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from pydantic_core import to_jsonable_python

import core.structural_modeling_components.general as general
from core.object_components import InputData_StructuralElements, StructuralModelResults
from core.structural_modeling_components.interpolator_functions.radial_basis_function import (
    fit_group_radial_basis_function,
    interpolate_group_radial_basis_function,
)
from core.structural_modeling_components.structural_modeling_utility.octree_refinement import (
    base_cells,
    refine_cells,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    OctreeCells,
    OctreeGrid,
    RegularGrid,
)
from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultElement,
    FaultFrame,
    InterpolationMethod,
)


# -----------------------------------------------------------------------------
# Input: two groups with an unconformity, no faults (see test_unconformities_analytical)
#   UpperSeries: rock4, rock3 (horizontal at z = 0.82, 0.62)
#   LowerSeries: rock2, rock1 (45 degrees in x, eroded by UpperSeries)
# -----------------------------------------------------------------------------

def unconformity_input() -> InputData_StructuralElements:
    xs = np.linspace(0.05, 0.95, 5)
    surfaces = {
        "rock1": lambda x: 0.20 + (x - 0.5),
        "rock2": lambda x: 0.40 + (x - 0.5),
        "rock3": lambda x: 0.62,
        "rock4": lambda x: 0.82,
    }
    rows = [{"X": float(x), "Y": float(y), "Z": float(z_of(x)), "formation": name}
            for name, z_of in surfaces.items() for x in xs for y in xs]
    # build_structural_frame needs orientations; RBF ignores them.
    orientations = [dict(row, G_x=0.0, G_y=0.0, G_z=1.0) for row in rows]
    return InputData_StructuralElements(
        name="unconformity",
        mapping_object={"UpperSeries": ("rock4", "rock3"), "LowerSeries": ("rock2", "rock1")},
        surface_points=pd.DataFrame(rows),
        orientations=pd.DataFrame(orientations),
    )


def octree_grid(base=(2, 2, 2), max_level=3) -> OctreeGrid:
    return OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=base, max_level=max_level)


def compute(grid) -> StructuralModelResults:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
        return general.compute_structural_model(frame, extract_meshes=False)


@pytest.fixture(scope="module")
def results() -> tuple[StructuralModelResults, StructuralModelResults]:
    grid = octree_grid()
    return compute(grid), compute(grid.equivalent_regular_grid())


# -----------------------------------------------------------------------------
# RBF fit / evaluate
# -----------------------------------------------------------------------------

def test_rbf_fit_evaluates_to_the_regular_grid_interpolation():
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 5, 4))
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
    group = frame.structural_groups[1]
    sp = frame.surface_points[frame.surface_points["formation"].isin(["rock1", "rock2"])]

    on_grid, values_on_grid = interpolate_group_radial_basis_function(
        group=group, grid=grid, group_surface_points_df=sp)
    field = fit_group_radial_basis_function(group=group, group_surface_points_df=sp)

    assert field.scalar_values == values_on_grid
    # interpolate_group_* returns reshape(resolution).T; undo the transpose to compare.
    np.testing.assert_array_equal(field.evaluate(grid.grid_coordinates).reshape(grid.resolution), on_grid.T)


# -----------------------------------------------------------------------------
# Refinement
# -----------------------------------------------------------------------------

def test_refinement_follows_a_plane_down_to_max_level():
    grid = octree_grid(base=(2, 2, 2), max_level=4)
    z_plane = 0.37
    cells = refine_cells(grid, lambda p: (p[:, 2] > z_plane).astype(int))

    # The cells tile the extent.
    assert (8 ** (grid.max_level - cells.level.astype(int))).sum() == np.prod(grid.finest_resolution)
    # Every cell the plane passes through is at max level; coarser cells don't touch it.
    mins, maxs = grid.cell_bounds(cells.level, cells.ijk)
    crossed = (mins[:, 2] < z_plane) & (maxs[:, 2] > z_plane)
    assert np.all(cells.level[crossed] == grid.max_level)
    assert crossed.sum() == grid.finest_resolution[0] * grid.finest_resolution[1]
    assert len(cells) < np.prod(grid.finest_resolution)


def test_refinement_without_change_keeps_the_base_blocks():
    grid = octree_grid(base=(3, 2, 1), max_level=3)
    cells = refine_cells(grid, lambda p: np.zeros(len(p), dtype=int))
    expected = base_cells(grid)
    np.testing.assert_array_equal(cells.level, expected.level)
    np.testing.assert_array_equal(cells.ijk, expected.ijk)


def test_refinement_continues_from_given_cells():
    grid = octree_grid(base=(1, 1, 1), max_level=3)
    start = OctreeCells(level=np.array([3], dtype=np.uint8), ijk=np.array([[0, 0, 0]], dtype=np.int32))
    # A max-level start cell is kept as it is, whatever its corners say.
    cells = refine_cells(grid, lambda p: (p[:, 0] > 0.01).astype(int), start=start)
    assert len(cells) == 1 and cells.level[0] == 3


# -----------------------------------------------------------------------------
# Full pipeline: octree vs. regular grid at the finest resolution
# -----------------------------------------------------------------------------

def test_octree_result_matches_the_regular_grid(results):
    octree_result, regular_result = results
    oct_frame, reg_frame = octree_result.structural_frame, regular_result.structural_frame

    assert len(oct_frame.cells) < np.prod(oct_frame.grid.finest_resolution)
    oct_frame.check_cell_arrays()
    np.testing.assert_array_equal(oct_frame.to_regular(oct_frame.lith_block), reg_frame.lith_block)
    for oct_group, reg_group in zip(oct_frame.structural_groups, reg_frame.structural_groups):
        np.testing.assert_array_equal(oct_frame.to_regular(oct_group.mask), reg_group.mask)
        assert [e.scalar_value for e in oct_group.structural_elements] == \
               [e.scalar_value for e in reg_group.structural_elements]
        assert [e.id for e in oct_group.structural_elements] == [e.id for e in reg_group.structural_elements]


def test_scalar_fields_at_max_level_cells_equal_the_regular_grid(results):
    octree_result, regular_result = results
    oct_frame, reg_frame = octree_result.structural_frame, regular_result.structural_frame
    fine = oct_frame.cells.level == oct_frame.grid.max_level
    i, j, k = oct_frame.cells.ijk[fine].T
    for oct_group, reg_group in zip(oct_frame.structural_groups, reg_frame.structural_groups):
        np.testing.assert_allclose(oct_group.scalar_field[fine], reg_group.scalar_field[i, j, k], atol=1e-12)


def test_eroded_surfaces_stay_refined(results):
    """The lower group's own surfaces are refined even above the unconformity, where they are eroded."""
    oct_frame = results[0].structural_frame
    lower = oct_frame.structural_groups[1]
    mins, maxs = oct_frame.grid.cell_bounds(oct_frame.cells.level, oct_frame.cells.ijk)
    centers = oct_frame.cells.centers(oct_frame.grid)
    rock2_top = 0.40 + (centers[:, 0] - 0.5)   # rock2 surface at the cell's x
    crosses_rock2 = (mins[:, 2] < rock2_top) & (maxs[:, 2] > rock2_top)
    above_unconformity = mins[:, 2] > 0.62
    assert lower.mask.all()
    eroded = crosses_rock2 & above_unconformity
    assert eroded.any()
    assert np.all(oct_frame.cells.level[eroded] == oct_frame.grid.max_level)


def test_octree_result_survives_the_runner_round_trip(results):
    octree_result = results[0]
    restored = StructuralModelResults(**json.loads(json.dumps(to_jsonable_python(octree_result))))
    frame = restored.structural_frame
    assert isinstance(frame.grid, OctreeGrid)
    frame.check_cell_arrays()
    np.testing.assert_array_equal(frame.lith_block, octree_result.structural_frame.lith_block)


def test_octree_frame_is_not_meshed_yet(results):
    oct_frame = results[0].structural_frame
    assert all(not e.vertices for g in oct_frame.structural_groups for e in g.structural_elements)
    assert all(g.extended_scalar_field is None for g in oct_frame.structural_groups)


# -----------------------------------------------------------------------------
# Not supported yet
# -----------------------------------------------------------------------------

def test_other_interpolators_are_refused_on_an_octree():
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=octree_grid())
    frame.structural_groups[0].set_interpolation_method(InterpolationMethod.ORDINARY_KRIGING)
    with pytest.raises(NotImplementedError, match="UpperSeries"):
        general.compute_structural_model(frame)


def test_faults_are_refused_on_an_octree():
    grid = octree_grid()
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
    fault_frame = FaultFrame(fault_elements=[FaultElement(name="F1")])
    fault_frame.set_grid(grid)
    fault_frame.cells = base_cells(grid)
    fault_frame.set_domain_map(np.zeros(len(fault_frame.cells), dtype=int))
    frame.set_fault_frame(fault_frame)
    with pytest.raises(NotImplementedError, match="faults"):
        general.compute_structural_model(frame)
