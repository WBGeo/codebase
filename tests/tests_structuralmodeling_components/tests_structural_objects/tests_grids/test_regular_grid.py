
from __future__ import annotations

import numpy as np
import pytest

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid


def test_post_init_computes_spacing_and_axes_cell_centers():
    grid = RegularGrid(extent=(0.0, 10.0, 0.0, 20.0, -5.0, 5.0), resolution=(5, 4, 2))

    # spacing = (dx, dy, dz)
    assert grid.spacing == (2.0, 5.0, 5.0)

    # cell centers: start at min + dx/2, end at max - dx/2
    assert np.allclose(grid.gridx, np.array([1, 3, 5, 7, 9], dtype=float))
    assert np.allclose(grid.gridy, np.array([2.5, 7.5, 12.5, 17.5], dtype=float))
    assert np.allclose(grid.gridz, np.array([-2.5, 2.5], dtype=float))


def test_meshgrid_shapes_and_grid_coordinates_shape_and_dtype():
    grid = RegularGrid(extent=(0.0, 2.0, 0.0, 3.0, 0.0, 4.0), resolution=(2, 3, 4))

    gx, gy, gz = grid.g
    assert gx.shape == (2, 3, 4)
    assert gy.shape == (2, 3, 4)
    assert gz.shape == (2, 3, 4)

    assert grid.grid_coordinates.shape == (2 * 3 * 4, 3)
    assert grid.grid_coordinates.dtype == np.float64


def test_grid_coordinates_order_is_meshgrid_ij_ravel_stacked():
    # Use tiny grid so we can assert exact ordering.
    # extent chosen so centers are exactly integers.
    grid = RegularGrid(extent=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0), resolution=(2, 2, 2))
    # Centers are [0.5, 1.5] on all axes.

    # meshgrid indexing="ij" means i (x) changes slowest, k (z) fastest when raveling C-order arrays.
    expected = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 1.5],
            [0.5, 1.5, 0.5],
            [0.5, 1.5, 1.5],
            [1.5, 0.5, 0.5],
            [1.5, 0.5, 1.5],
            [1.5, 1.5, 0.5],
            [1.5, 1.5, 1.5],
        ],
        dtype=float,
    )

    assert np.allclose(grid.grid_coordinates, expected)


def test_xyz_to_indices_basic_mapping():
    grid = RegularGrid(extent=(0.0, 10.0, 0.0, 20.0, -5.0, 5.0), resolution=(5, 4, 2))
    # spacing (2,5,5); origin (xmin, ymin, zmin) = (0,0,-5)

    coords = np.array(
        [
            [0.0, 0.0, -5.0],   # exactly origin -> (0,0,0)
            [1.9, 4.9, -0.1],   # still inside first x,y cell, z cell 0 -> (0,0,0)
            [2.0, 5.0, 0.0],    # boundary -> next cell in x,y,z -> (1,1,1) because floor_divide
            [9.999, 19.999, 4.999],  # near max -> last indices (4,3,1)
        ],
        dtype=float,
    )

    idx = grid.xyz_to_indices(coords)
    assert np.array_equal(idx[0], np.array([0, 0, 0]))
    assert np.array_equal(idx[1], np.array([0, 0, 0]))
    assert np.array_equal(idx[2], np.array([1, 1, 1]))
    assert np.array_equal(idx[3], np.array([4, 3, 1]))


def test_xyz_to_indices_accepts_multiple_points_and_returns_int_array():
    grid = RegularGrid(extent=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0), resolution=(2, 2, 2))
    coords = np.array([[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]], dtype=float)
    idx = grid.xyz_to_indices(coords)

    assert idx.shape == (2, 3)
    assert np.issubdtype(idx.dtype, np.integer)


def test_xyz_to_indices_raises_if_spacing_missing():
    # create a "partially initialized" instance without running __post_init__
    grid = object.__new__(RegularGrid)
    grid.extent = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    with pytest.raises(AttributeError, match="Grid must have 'origin' and 'spacing'"):
        grid.xyz_to_indices(np.array([[0.0, 0.0, 0.0]], dtype=float))
