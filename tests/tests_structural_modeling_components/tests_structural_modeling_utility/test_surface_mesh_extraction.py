# tests/test_marching_cubes_utils.py

from __future__ import annotations

import numpy as np
import pytest

from core.structural_modeling_components.general import _build_padded_grid
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction import (
    marching_cubes, marching_cubes_per_element)


def test_marching_cubes_calls_skimage_for_each_level_and_offsets_vertices(monkeypatch):
    # Arrange
    block = np.zeros((2, 2, 2), dtype=float)
    elements = [10, 20]  # important: non-0/1 IDs
    spacing = (1.0, 2.0, 3.0)
    extent = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)

    calls: list[dict] = []

    def fake_marching_cubes(block_arg, level, spacing=None, **kwargs):
        calls.append({"block": block_arg, "level": level, "spacing": spacing, "kwargs": kwargs})
        # verts are simple and recognizable
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float)
        faces = np.array([[0, 1, 0]], dtype=int)
        normals = np.zeros_like(verts)
        values = np.zeros((verts.shape[0],), dtype=float)
        return verts, faces, normals, values

    # Patch the function where it is looked up: in your module under test
    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction as mod

    monkeypatch.setattr(mod.measure, "marching_cubes", fake_marching_cubes)

    # Act
    mc_vertices, mc_edges = marching_cubes(block, elements, spacing, extent)

    # Assert: called once per element with float(level) and spacing forwarded
    assert [c["level"] for c in calls] == [float(e) for e in elements]
    assert all(c["spacing"] == spacing for c in calls)

    # Assert: vertices are offset to the first cell centre (extent min + spacing/2)
    offset = np.array([extent[0], extent[2], extent[4]], dtype=float) + np.array(spacing) / 2
    expected_verts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float) + offset

    assert len(mc_vertices) == 2
    assert np.allclose(mc_vertices[0], expected_verts)
    assert np.allclose(mc_vertices[1], expected_verts)

    # Assert: faces passed through
    assert len(mc_edges) == 2
    assert np.array_equal(mc_edges[0], np.array([[0, 1, 0]], dtype=int))


def test_marching_cubes_accepts_float_levels(monkeypatch):
    block = np.zeros((2, 2, 2), dtype=float)
    elements = [0.5, 1.25]
    spacing = (1.0, 1.0, 1.0)
    extent = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    seen_levels = []

    def fake_marching_cubes(block_arg, level, spacing=None, **kwargs):
        seen_levels.append(level)
        verts = np.zeros((1, 3), dtype=float)
        faces = np.zeros((1, 3), dtype=int)
        return verts, faces, None, None

    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction as mod
    monkeypatch.setattr(mod.measure, "marching_cubes", fake_marching_cubes)

    marching_cubes(block, elements, spacing, extent)

    # ensure cast to float(level) is preserved
    assert seen_levels == [float(x) for x in elements]


def test_marching_cubes_per_element_passes_mask_and_offsets_vertices(monkeypatch):
    block = np.zeros((3, 3, 3), dtype=float)
    element = 7.0
    spacing = (2.0, 2.0, 2.0)
    extent = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    mask = np.zeros(block.shape, dtype=bool)
    mask[1, 1, 1] = True

    captured = {}

    def fake_marching_cubes(block_arg, level, spacing=None, mask=None, **kwargs):
        captured["block"] = block_arg
        captured["level"] = level
        captured["spacing"] = spacing
        captured["mask"] = mask
        verts = np.array([[5.0, 6.0, 7.0]], dtype=float)
        faces = np.array([[0, 0, 0]], dtype=int)
        return verts, faces, None, None

    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction as mod
    monkeypatch.setattr(mod.measure, "marching_cubes", fake_marching_cubes)

    vertices, edges = marching_cubes_per_element(block, element, spacing, extent, mask)

    assert captured["level"] == element
    assert captured["spacing"] == spacing
    assert captured["mask"] is mask

    offset = np.array([extent[0], extent[2], extent[4]], dtype=float) + np.array(spacing) / 2
    assert np.allclose(vertices, np.array([[5.0, 6.0, 7.0]]) + offset)
    assert np.array_equal(edges, np.array([[0, 0, 0]], dtype=int))


# -----------------------------------------------------------------------------
# Real skimage on RegularGrid: surfaces must sit where the field crosses the level
# -----------------------------------------------------------------------------

_GRIDS = [
    ((0.0, 10.0, 0.0, 10.0, 0.0, 10.0), (10, 10, 10)),
    ((100.0, 1100.0, 0.0, 500.0, -800.0, 0.0), (20, 10, 16)),
]


def _field_on_grid(grid: RegularGrid, axis: int) -> np.ndarray:
    """Coordinate field f = x, y or z evaluated like an interpolator: at grid points."""
    return grid.grid_coordinates[:, axis].reshape(grid.resolution)


@pytest.mark.parametrize("extent, resolution", _GRIDS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_marching_cubes_per_element_places_plane_at_its_level(extent, resolution, axis):
    grid = RegularGrid(extent=extent, resolution=resolution)
    level = 0.5 * (extent[2 * axis] + extent[2 * axis + 1]) + 0.3 * grid.spacing[axis]

    vertices, _ = marching_cubes_per_element(
        _field_on_grid(grid, axis), level, grid.spacing, grid.extent, mask=None
    )

    assert len(vertices) > 0
    assert np.allclose(vertices[:, axis], level)


@pytest.mark.parametrize("extent, resolution", _GRIDS)
def test_marching_cubes_places_tilted_plane_at_its_level(extent, resolution):
    grid = RegularGrid(extent=extent, resolution=resolution)
    field = grid.grid_coordinates.sum(axis=1).reshape(grid.resolution)
    level = 0.5 * sum(extent)

    (vertices,), _ = marching_cubes(field, [level], grid.spacing, grid.extent)

    assert np.allclose(vertices.sum(axis=1), level)


def test_vertices_stay_inside_the_cell_centre_hull():
    grid = RegularGrid(extent=(0.0, 10.0, 0.0, 10.0, 0.0, 10.0), resolution=(10, 10, 10))
    vertices, _ = marching_cubes_per_element(
        _field_on_grid(grid, 2), 5.0, grid.spacing, grid.extent, mask=None
    )

    hull_min = np.array([grid.gridx[0], grid.gridy[0], grid.gridz[0]])
    hull_max = np.array([grid.gridx[-1], grid.gridy[-1], grid.gridz[-1]])
    assert np.all(vertices >= hull_min - 1e-9)
    assert np.all(vertices <= hull_max + 1e-9)


def test_padded_grid_surface_matches_the_unpadded_one():
    """EXTENDED meshes pass a padded block with a padded extent through the same helper."""
    grid = RegularGrid(extent=(100.0, 1100.0, 0.0, 500.0, -800.0, 0.0), resolution=(20, 10, 16))
    padded = _build_padded_grid(grid, 2)
    level = -390.0

    vertices, _ = marching_cubes_per_element(
        _field_on_grid(padded, 2), level, padded.spacing, padded.extent, mask=None
    )

    assert np.allclose(vertices[:, 2], level)
    # The padded surface reaches beyond the model box, as the GMSH path requires.
    assert vertices[:, 0].min() < grid.extent[0] and vertices[:, 0].max() > grid.extent[1]
