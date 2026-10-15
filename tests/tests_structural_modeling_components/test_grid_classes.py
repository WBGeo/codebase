from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np
import pytest

from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    CORNER_OFFSETS,
    GapMitigation,
    OctreeGrid,
    RegularGrid,
)


# Non-zero origin and a non-cubic extent split into cubic 50 m base blocks.
EXTENT = (100.0, 300.0, -50.0, 150.0, 10.0, 60.0)
BASE = (4, 4, 1)


def make_octree(max_level: int = 2) -> OctreeGrid:
    return OctreeGrid(extent=EXTENT, base_resolution=BASE, max_level=max_level)


def all_cells(grid: OctreeGrid, level: int) -> np.ndarray:
    """Every cell index at a level, in the same (ij) order as RegularGrid.grid_coordinates."""
    n = grid.cells_per_axis(level)
    ii, jj, kk = np.meshgrid(*(np.arange(k) for k in n), indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])


# -----------------------------------------------------------------------------
# Construction and validation
# -----------------------------------------------------------------------------

def test_derived_geometry():
    grid = make_octree(max_level=2)
    np.testing.assert_allclose(grid.origin, [100.0, -50.0, 10.0])
    np.testing.assert_allclose(grid.base_spacing, [50.0, 50.0, 50.0])
    assert grid.finest_resolution == (16, 16, 4)
    np.testing.assert_allclose(grid.finest_spacing, [12.5, 12.5, 12.5])


def test_defaults():
    grid = OctreeGrid(extent=(0, 1000, 0, 1000, 0, 1000), base_resolution=(4, 4, 4))
    assert grid.max_level == 4
    assert grid.gap_mitigation is GapMitigation.BALANCE_MESH_DEPTH_ADJACENCY
    assert grid.finest_resolution == (64, 64, 64)


def test_gap_mitigation_from_string():
    grid = OctreeGrid(extent=EXTENT, base_resolution=BASE, gap_mitigation="none")
    assert grid.gap_mitigation is GapMitigation.NONE


def test_gap_mitigation_dropdown_values_deserialize_like_the_runner():
    # The Workbench runner sends the dropdown value. For a StrEnum it accepts
    # values as well as names; any other Enum is looked up by name only, which
    # failed live with KeyError: 'balance_mesh_depth_adjacency'.
    assert issubclass(GapMitigation, StrEnum)
    for value in ("balance_mesh_depth_adjacency", "none"):
        assert value in GapMitigation._value2member_map_
        grid = OctreeGrid(extent=EXTENT, base_resolution=BASE, gap_mitigation=value)
        assert grid.gap_mitigation == value


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(extent=(0.0, 0.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(1, 1, 1)),
        dict(extent=(0.0, 1.0, 2.0, 1.0, 0.0, 1.0), base_resolution=(1, 1, 1)),
        dict(extent=EXTENT, base_resolution=(4, 0, 1)),
        dict(extent=EXTENT, base_resolution=(4, 4, -1)),
        dict(extent=EXTENT, base_resolution=BASE, max_level=-1),
        dict(extent=EXTENT, base_resolution=BASE, max_level=11),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        OctreeGrid(**kwargs)


def test_max_level_zero_is_grid_of_base_blocks():
    grid = OctreeGrid(extent=EXTENT, base_resolution=BASE, max_level=0)
    assert grid.finest_resolution == BASE
    np.testing.assert_allclose(grid.finest_spacing, grid.base_spacing)


def test_stretched_base_blocks_warn_with_suggestion(caplog):
    with caplog.at_level(logging.WARNING):
        OctreeGrid(extent=(0, 20000, 0, 20000, 0, 4000), base_resolution=(1, 1, 1))
    assert "stretched" in caplog.text
    assert "(5, 5, 1)" in caplog.text


def test_cubic_base_blocks_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        OctreeGrid(extent=(0, 20000, 0, 20000, 0, 4000), base_resolution=(5, 5, 1))
    assert caplog.text == ""


# -----------------------------------------------------------------------------
# Cell geometry
# -----------------------------------------------------------------------------

def test_cells_per_axis_and_cell_size():
    grid = make_octree(max_level=3)
    np.testing.assert_array_equal(grid.cells_per_axis(0), [4, 4, 1])
    np.testing.assert_array_equal(grid.cells_per_axis(3), [32, 32, 8])
    np.testing.assert_allclose(grid.cell_size(1), [25.0, 25.0, 25.0])
    np.testing.assert_allclose(
        grid.cell_size(np.array([0, 2, 3])),
        [[50.0, 50.0, 50.0], [12.5, 12.5, 12.5], [6.25, 6.25, 6.25]],
    )


def test_levels_outside_range_raise():
    grid = make_octree(max_level=2)
    with pytest.raises(ValueError):
        grid.cell_size(3)
    with pytest.raises(ValueError):
        grid.cell_centers(np.array([0, -1]), np.zeros((2, 3), dtype=int))
    with pytest.raises(ValueError):
        grid.cells_per_axis(3)


def test_max_level_centers_match_equivalent_regular_grid():
    grid = make_octree(max_level=2)
    regular = grid.equivalent_regular_grid()
    assert isinstance(regular, RegularGrid)
    assert regular.resolution == grid.finest_resolution
    centers = grid.cell_centers(grid.max_level, all_cells(grid, grid.max_level))
    np.testing.assert_allclose(centers, regular.grid_coordinates)


def test_base_cells_tile_the_extent():
    grid = make_octree()
    mins, maxs = grid.cell_bounds(0, all_cells(grid, 0))
    np.testing.assert_allclose(mins.min(axis=0), EXTENT[0::2])
    np.testing.assert_allclose(maxs.max(axis=0), EXTENT[1::2])
    volume = np.prod(maxs - mins, axis=1).sum()
    assert volume == pytest.approx(np.prod(np.subtract(EXTENT[1::2], EXTENT[0::2])))


def test_corner_order():
    grid = make_octree()
    ijk = np.array([[1, 2, 0]])
    corners = grid.cell_corners(1, ijk)[0]
    mins, maxs = grid.cell_bounds(1, ijk)
    np.testing.assert_allclose(corners[0], mins[0])
    np.testing.assert_allclose(corners[7], maxs[0])
    for c in range(8):
        expected = np.where(CORNER_OFFSETS[c] == 1, maxs[0], mins[0])
        np.testing.assert_allclose(corners[c], expected)


def test_corners_with_mixed_levels():
    grid = make_octree(max_level=2)
    levels = np.array([0, 2])
    ijk = np.array([[3, 3, 0], [0, 0, 0]])
    corners = grid.cell_corners(levels, ijk)
    assert corners.shape == (2, 8, 3)
    np.testing.assert_allclose(corners[0, 7], [300.0, 150.0, 60.0])
    np.testing.assert_allclose(corners[1, 7], [112.5, -37.5, 22.5])


def test_children_tile_their_parent():
    grid = make_octree(max_level=2)
    parent = np.array([[2, 1, 0]])
    children = 2 * parent + CORNER_OFFSETS
    p_min, p_max = grid.cell_bounds(1, parent)
    c_min, c_max = grid.cell_bounds(2, children)
    np.testing.assert_allclose(c_min.min(axis=0), p_min[0])
    np.testing.assert_allclose(c_max.max(axis=0), p_max[0])
    centers = grid.cell_centers(2, children)
    assert np.all((centers > p_min) & (centers < p_max))


# -----------------------------------------------------------------------------
# Point location
# -----------------------------------------------------------------------------

def test_locate_round_trips_cell_centers():
    grid = make_octree(max_level=2)
    for level in range(grid.max_level + 1):
        ijk = all_cells(grid, level)
        np.testing.assert_array_equal(grid.locate(grid.cell_centers(level, ijk), level), ijk)


def test_locate_is_consistent_across_levels():
    grid = make_octree(max_level=3)
    rng = np.random.default_rng(0)
    points = grid.origin + rng.random((200, 3)) * (np.array(EXTENT[1::2]) - grid.origin)
    for level in range(grid.max_level):
        np.testing.assert_array_equal(grid.locate(points, level), grid.locate(points, level + 1) // 2)


def test_locate_faces_and_outside_points():
    grid = make_octree(max_level=1)
    points = np.array([
        [125.0, -50.0, 10.0],   # on the face between x-cells 0 and 1 -> upper cell
        [300.0, 150.0, 60.0],   # upper corner of the extent -> last cell
        [0.0, 1000.0, -5.0],    # outside -> clamped
    ])
    np.testing.assert_array_equal(
        grid.locate(points, 1),
        [[1, 0, 0], [7, 7, 1], [0, 7, 0]],
    )
