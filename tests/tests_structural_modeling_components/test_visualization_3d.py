"""
3D plots on both grid types: the lithology block as cell faces on the walls of
the extent (show_lith_block), and the fault model's domain blocks on an octree.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import pyvista as pv
from matplotlib.colors import to_rgb

import core.structural_modeling_components.general as general
import core.structural_modeling_components.general_faults as gf
from core.structural_modeling_components.structural_modeling_visualization import (
    structural_modeling_visualization as viz,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeGrid

pytest.importorskip("gempy")  # faults are fitted with GemPy


@pytest.fixture(autouse=True)
def off_screen(monkeypatch):
    monkeypatch.setattr(pv, "OFF_SCREEN", True)


@pytest.fixture(scope="module")
def results():
    """Model 2 of the integration tests (fault + unconformity) on an octree and its regular grid."""
    from test_octree_faulted_structural_model import MODELS, _integration_module, octree_grid
    elements, faults = _integration_module(MODELS["model2"][0])._build_input_data()
    out = {}
    for name, grid in [("octree", octree_grid()), ("regular", octree_grid().equivalent_regular_grid())]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fault_result = gf.compute_fault_domains(gf.build_fault_frame(faults, grid))
            frame = general.build_structural_frame(
                input_data_elements=elements, grid=grid, fault_model_results=fault_result)
            out[name] = general.compute_structural_model(frame, verbose=False).structural_frame
    return out


def _box(frame):
    lo, hi = np.array(frame.grid.extent[0::2]), np.array(frame.grid.extent[1::2])
    return lo, hi, hi - lo


@pytest.mark.parametrize("kind", ["octree", "regular"])
def test_wall_faces_tile_the_outside_of_the_model(results, kind):
    frame = results[kind]
    faces = viz._wall_faces(frame)
    lo, hi, size = _box(frame)
    np.testing.assert_allclose(faces.compute_cell_sizes()["Area"].sum(),
                               2 * (size[0] * size[1] + size[1] * size[2] + size[0] * size[2]), rtol=1e-12)
    # Every vertex lies on a wall.
    on_wall = np.isclose(faces.points, lo) | np.isclose(faces.points, hi)
    assert np.all(on_wall.any(axis=1))
    if kind == "regular":
        nx, ny, nz = frame.grid.resolution
        assert faces.n_cells == 2 * (nx * ny + ny * nz + nx * nz)
    else:
        assert faces.n_cells < 2 * sum(np.prod(np.delete(frame.grid.finest_resolution, a)) for a in range(3))


@pytest.mark.parametrize("kind", ["octree", "regular"])
def test_wall_faces_take_the_lithology_color_of_their_cell(results, kind):
    frame = results[kind]
    faces = viz._wall_faces(frame)
    lo, hi, _ = _box(frame)
    # A point just inside each face, in the cell the face belongs to.
    centers = faces.cell_centers().points
    inward = np.where(np.isclose(centers, lo), 1.0, np.where(np.isclose(centers, hi), -1.0, 0.0))
    inside = centers + inward * 1e-6 * (hi - lo)
    if kind == "octree":
        lith = np.asarray(frame.lith_block)[frame.cells.locate(frame.grid, inside)]
    else:
        ijk = np.clip(((inside - lo) / np.array(frame.grid.spacing)).astype(int), 0,
                      np.array(frame.grid.resolution) - 1)
        lith = frame.lith_block[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    id_to_color, _ = viz._lithology_colors(frame)
    expected = np.array([np.round(np.array(to_rgb(id_to_color[int(i)])) * 255) for i in lith])
    np.testing.assert_array_equal(faces.cell_data["rgb"], expected)
    assert len(np.unique(lith)) > 1


@pytest.mark.parametrize("kind", ["octree", "regular"])
@pytest.mark.parametrize("show_lith_block", [False, True])
def test_structural_model_3d_plots_on_both_grids(results, kind, show_lith_block):
    plotter = viz.plot_structural_model_3D(results[kind], show_lith_block=show_lith_block, off_screen=True,
                                           show=False)
    try:
        assert plotter.renderer.actors
    finally:
        plotter.close()


def test_octree_hexahedra_fill_the_extent(results):
    fault_frame = results["octree"].fault_frame
    hexes = viz._octree_hexahedra(fault_frame.grid, fault_frame.cells)
    _, _, size = _box(fault_frame)
    assert hexes.n_cells == len(fault_frame.cells)
    np.testing.assert_allclose(hexes.volume, np.prod(size), rtol=1e-12)


@pytest.mark.parametrize("kind", ["octree", "regular"])
def test_fault_model_3d_plots_its_domains_on_both_grids(results, kind):
    fault_frame = results[kind].fault_frame
    plotter = viz.plot_fault_model_3D(fault_frame, show=False)
    try:
        # The domain blocks are the volume meshes among the actors: one per domain, together the extent.
        blocks = [actor.mapper.dataset for actor in plotter.renderer.actors.values()
                  if isinstance(actor, pv.Actor) and isinstance(actor.mapper.dataset, pv.UnstructuredGrid)]
        _, _, size = _box(fault_frame)
        assert len(blocks) == len(np.unique(np.asarray(fault_frame.domain_map))) == 2
        np.testing.assert_allclose(sum(block.volume for block in blocks), np.prod(size), rtol=1e-9)
    finally:
        plotter.close()
