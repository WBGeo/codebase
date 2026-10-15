"""
Per-cell result layout for both grid types: OctreeCells, cell_centers, to_regular,
the frame helpers, and the JSON round trip the Workbench runner applies to results.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic_core import to_jsonable_python

import core.structural_modeling_components.general as general
from core.object_components import FaultModelResults, InputData_StructuralElements, StructuralModelResults
from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    CORNER_OFFSETS,
    OctreeCells,
    OctreeGrid,
    RegularGrid,
    cell_array_shape,
    cell_centers,
    to_regular,
)
from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultElement,
    FaultFrame,
    StructuralElement,
    StructuralFrame,
    StructuralGroup,
)


# -----------------------------------------------------------------------------
# A small adaptive octree: 2 x 2 x 1 base blocks of 1 m, max level 2.
# Base block (1, 0, 0) is split once, and its first child is split again.
# -----------------------------------------------------------------------------

def make_grid() -> OctreeGrid:
    return OctreeGrid(extent=(0.0, 2.0, 0.0, 2.0, 0.0, 1.0), base_resolution=(2, 2, 1), max_level=2)


def split(level: np.ndarray, ijk: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    children_ijk = 2 * ijk[index] + CORNER_OFFSETS
    level = np.concatenate([np.delete(level, index), np.full(8, level[index] + 1)])
    ijk = np.concatenate([np.delete(ijk, index, axis=0), children_ijk])
    return level, ijk


def make_cells() -> OctreeCells:
    level = np.zeros(4, dtype=np.int64)
    ijk = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    level, ijk = split(level, ijk, 1)   # base block (1, 0, 0) -> 8 level-1 cells (now indices 3..10)
    level, ijk = split(level, ijk, 3)   # its child (2, 0, 0)  -> 8 level-2 cells
    return OctreeCells(level=level.astype(np.uint8), ijk=ijk.astype(np.int32))


def finest_cells_per_cell(grid: OctreeGrid, cells: OctreeCells) -> np.ndarray:
    return 8 ** (grid.max_level - cells.level.astype(int))


# -----------------------------------------------------------------------------
# OctreeCells and the module-level helpers
# -----------------------------------------------------------------------------

def test_cells_tile_the_extent():
    grid, cells = make_grid(), make_cells()
    assert len(cells) == 3 + 7 + 8
    assert finest_cells_per_cell(grid, cells).sum() == np.prod(grid.finest_resolution)


def test_cells_shape_validation():
    with pytest.raises(ValueError):
        OctreeCells(level=np.zeros(3, dtype=np.uint8), ijk=np.zeros((2, 3), dtype=np.int32))


def test_locate_finds_each_cell_from_its_center():
    grid, cells = make_grid(), make_cells()
    np.testing.assert_array_equal(cells.locate(grid, cells.centers(grid)), np.arange(len(cells)))


def test_locate_random_points_land_in_their_cell():
    grid, cells = make_grid(), make_cells()
    points = np.random.default_rng(1).random((300, 3)) * [2.0, 2.0, 1.0]
    index = cells.locate(grid, points)
    assert np.all(index >= 0)
    mins, maxs = grid.cell_bounds(cells.level[index], cells.ijk[index])
    assert np.all((points >= mins) & (points <= maxs))


def test_cell_array_shape_and_centers_for_both_grids():
    grid, cells = make_grid(), make_cells()
    assert cell_array_shape(grid, cells) == (len(cells),)
    np.testing.assert_allclose(cell_centers(grid, cells), cells.centers(grid))
    with pytest.raises(ValueError):
        cell_array_shape(grid, None)

    regular = RegularGrid(extent=(0.0, 2.0, 0.0, 2.0, 0.0, 1.0), resolution=(4, 3, 2))
    assert cell_array_shape(regular, None) == (4, 3, 2)
    np.testing.assert_allclose(cell_centers(regular, None), regular.grid_coordinates)
    with pytest.raises(ValueError):
        cell_array_shape(regular, cells)


def test_regular_cell_centers_follow_ravel_order():
    regular = RegularGrid(extent=(0.0, 4.0, 0.0, 3.0, 0.0, 2.0), resolution=(4, 3, 2))
    values = np.arange(24).reshape(4, 3, 2)
    centers = cell_centers(regular, None)
    i, j, k = 3, 1, 0
    np.testing.assert_allclose(centers[values.ravel() == values[i, j, k]][0],
                               [regular.gridx[i], regular.gridy[j], regular.gridz[k]])


def test_to_regular_octree_default_is_finest_resolution():
    grid, cells = make_grid(), make_cells()
    values = np.arange(len(cells))
    block = to_regular(grid, cells, values)
    assert block.shape == grid.finest_resolution
    # Every octree cell covers exactly its share of finest cells.
    np.testing.assert_array_equal(np.bincount(block.ravel(), minlength=len(cells)),
                                  finest_cells_per_cell(grid, cells))
    # And each finest cell sits inside the octree cell whose value it got.
    fine = grid.equivalent_regular_grid()
    mins, maxs = grid.cell_bounds(cells.level[block.ravel()], cells.ijk[block.ravel()])
    assert np.all((fine.grid_coordinates > mins) & (fine.grid_coordinates < maxs))


def test_to_regular_octree_custom_resolution():
    grid, cells = make_grid(), make_cells()
    block = to_regular(grid, cells, np.arange(len(cells)), resolution=(2, 2, 1))
    assert block.shape == (2, 2, 1)
    # The center of base block (1, 0, 0) lies on a corner shared by four of its
    # children; the containing cell is the upper one in every axis.
    expected = cells.locate(grid, np.array([[1.5, 0.5, 0.5]]))[0]
    assert block[1, 0, 0] == expected


def test_to_regular_regular_grid():
    regular = RegularGrid(extent=(0.0, 4.0, 0.0, 4.0, 0.0, 4.0), resolution=(4, 4, 4))
    values = np.arange(64).reshape(4, 4, 4)
    assert to_regular(regular, None, values) is values
    coarse = to_regular(regular, None, values, resolution=(2, 2, 2))
    assert coarse.shape == (2, 2, 2)
    assert coarse[1, 0, 1] == values[3, 1, 3]   # coarse center (3, 1, 3) lies on the corner of fine cell (3, 1, 3)
    with pytest.raises(ValueError):
        to_regular(regular, None, np.zeros(5))


# -----------------------------------------------------------------------------
# Frame helpers
# -----------------------------------------------------------------------------

def make_octree_frames() -> tuple[StructuralFrame, FaultFrame]:
    """A fault frame on the base blocks and a structural frame on the refined cells."""
    grid = make_grid()
    fault_cells = OctreeCells(level=np.zeros(4, dtype=np.uint8),
                              ijk=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.int32))
    fault = FaultElement(name="F1", scalar_value=0.0)
    fault.scalar_field = np.array([-1.0, 1.0, -1.0, 1.0])
    fault.mask = fault.scalar_field > 0
    fault_frame = FaultFrame(fault_elements=[fault])
    fault_frame.set_grid(grid)
    fault_frame.cells = fault_cells
    fault_frame.set_domain_map(np.array([0, 1, 0, 1]))
    fault_frame.domain_masks = {0: fault_frame.domain_map == 0, 1: fault_frame.domain_map == 1}

    cells = make_cells()
    group = StructuralGroup(name="G", structural_elements=[StructuralElement(name="a", scalar_value=0.5, id=1)])
    group.scalar_field = cells.centers(grid)[:, 2].copy()
    group.mask = np.ones(len(cells), dtype=bool)
    frame = StructuralFrame(structural_groups=[group])
    frame.grid = grid
    frame.cells = cells
    frame.surface_points = pd.DataFrame({"X": [0.5], "Y": [0.5], "Z": [0.5], "formation": ["a"]})
    frame.lith_block = np.where(group.scalar_field >= 0.5, 1, 0)
    frame.set_fault_frame(fault_frame)
    return frame, fault_frame


def test_octree_frame_arrays_pass_the_check():
    frame, _ = make_octree_frames()
    frame.check_cell_arrays()
    assert frame.cell_array_shape() == (len(frame.cells),)
    assert frame.fault_frame.cell_array_shape() == (4,)


def test_check_cell_arrays_rejects_wrong_shapes():
    frame, _ = make_octree_frames()
    frame.lith_block = np.zeros((2, 2, 1))
    with pytest.raises(ValueError, match="lith_block"):
        frame.check_cell_arrays()

    frame, _ = make_octree_frames()
    frame.fault_frame.fault_elements[0].mask = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError, match="F1.mask"):
        frame.check_cell_arrays()


def test_cell_domain_ids_inherit_from_fault_cells():
    frame, fault_frame = make_octree_frames()
    domain_ids = frame.cell_domain_ids()
    assert domain_ids.shape == (len(frame.cells),)
    # Base blocks with ix == 1 are domain 1; that includes every refined cell.
    base_ix = frame.cells.ijk[:, 0] // 2 ** frame.cells.level.astype(int)
    np.testing.assert_array_equal(domain_ids, np.where(base_ix == 1, 1, 0))


def test_cell_domain_ids_regular_and_without_faults():
    regular = RegularGrid(extent=(0.0, 4.0, 0.0, 1.0, 0.0, 1.0), resolution=(4, 1, 1))
    frame = StructuralFrame(structural_groups=[])
    frame.grid = regular
    np.testing.assert_array_equal(frame.cell_domain_ids(), np.zeros((4, 1, 1)))

    fault_frame = FaultFrame(fault_elements=[FaultElement(name="F1")])
    fault_frame.set_grid(regular)
    fault_frame.set_domain_map(np.array([0, 0, 1, 1]).reshape(4, 1, 1))
    frame.set_fault_frame(fault_frame)
    np.testing.assert_array_equal(frame.cell_domain_ids(), fault_frame.domain_map)


def test_frame_to_regular_uses_the_frames_own_cells():
    frame, fault_frame = make_octree_frames()
    lith = frame.to_regular(frame.lith_block)
    domains = fault_frame.to_regular(fault_frame.domain_map)
    assert lith.shape == domains.shape == frame.grid.finest_resolution
    # 8 finest cells along x, the first 4 in base blocks with ix == 0.
    np.testing.assert_array_equal(domains[:4], 0)
    np.testing.assert_array_equal(domains[4:], 1)


def test_set_fault_frame_rejects_a_different_grid_type():
    frame, fault_frame = make_octree_frames()
    fault_frame.set_grid(RegularGrid(extent=(0.0, 2.0, 0.0, 2.0, 0.0, 1.0), resolution=(2, 2, 1)))
    with pytest.raises(ValueError, match="same grid"):
        frame.set_fault_frame(fault_frame)


# -----------------------------------------------------------------------------
# Round trip through the Workbench runner's (de)serialization, both grid types
# -----------------------------------------------------------------------------

def runner_round_trip(result):
    """What py_runner does: json.dumps(to_jsonable_python(r)) out, type(**json.loads()) in."""
    return type(result)(**json.loads(json.dumps(to_jsonable_python(result))))


def make_regular_frames() -> tuple[StructuralFrame, FaultFrame]:
    grid = RegularGrid(extent=(0.0, 4.0, 0.0, 1.0, 0.0, 1.0), resolution=(4, 1, 1))
    fault = FaultElement(name="F1", scalar_value=0.0)
    fault.scalar_field = np.array([-2.0, -1.0, 1.0, 2.0]).reshape(4, 1, 1)
    fault.mask = fault.scalar_field > 0
    fault_frame = FaultFrame(fault_elements=[fault])
    fault_frame.set_grid(grid)
    fault_frame.set_domain_map(np.array([0, 0, 1, 1]).reshape(4, 1, 1))

    group = StructuralGroup(name="G", structural_elements=[StructuralElement(name="a", scalar_value=0.5, id=1)])
    group.scalar_field = np.linspace(0, 1, 4).reshape(4, 1, 1)
    group.mask = np.ones((4, 1, 1), dtype=bool)
    frame = StructuralFrame(structural_groups=[group])
    frame.grid = grid
    frame.surface_points = pd.DataFrame({"X": [0.5], "Y": [0.5], "Z": [0.5], "formation": ["a"]})
    frame.lith_block = np.array([0, 0, 1, 1]).reshape(4, 1, 1)
    frame.set_fault_frame(fault_frame)
    return frame, fault_frame


@pytest.mark.parametrize("make_frames", [make_regular_frames, make_octree_frames], ids=["regular", "octree"])
def test_results_survive_the_runner_round_trip(make_frames):
    frame, fault_frame = make_frames()
    structural = runner_round_trip(StructuralModelResults(structural_frame=frame))
    faults = runner_round_trip(FaultModelResults(fault_frame=fault_frame))

    restored = structural.structural_frame
    assert type(restored.grid) is type(frame.grid)
    assert restored.grid == frame.grid
    assert type(faults.fault_frame.grid) is type(fault_frame.grid)
    if frame.cells is None:
        assert restored.cells is None and restored.fault_frame.cells is None
    else:
        np.testing.assert_array_equal(restored.cells.level, frame.cells.level)
        np.testing.assert_array_equal(restored.cells.ijk, frame.cells.ijk)
        np.testing.assert_array_equal(restored.fault_frame.cells.ijk, fault_frame.cells.ijk)
    np.testing.assert_array_equal(restored.lith_block, frame.lith_block)
    np.testing.assert_allclose(restored.structural_groups[0].scalar_field, frame.structural_groups[0].scalar_field)
    np.testing.assert_array_equal(restored.fault_frame.domain_map, fault_frame.domain_map)
    restored.check_cell_arrays()
    faults.fault_frame.check_cell_arrays()


# -----------------------------------------------------------------------------
# Modeling entry points with an OctreeGrid
# -----------------------------------------------------------------------------

def test_build_structural_frame_leaves_scalar_fields_empty_on_an_octree():
    data = InputData_StructuralElements(
        name="t",
        mapping_object={"G": ("a", "b")},
        surface_points=pd.DataFrame({
            "X": [0.5, 1.5, 0.5, 1.5], "Y": [0.5] * 4, "Z": [0.3, 0.3, 0.7, 0.7],
            "formation": ["a", "a", "b", "b"],
        }),
        orientations=pd.DataFrame({
            "X": [0.5], "Y": [0.5], "Z": [0.3], "G_x": [0.0], "G_y": [0.0], "G_z": [1.0], "formation": ["a"],
        }),
    )
    frame = general.build_structural_frame(input_data_elements=data, grid=make_grid())
    assert all(group.scalar_field is None for group in frame.structural_groups)
    assert frame.cells is None
