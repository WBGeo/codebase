"""
2D sections on both grid types: frame.section gives the regular grid's slices on
an octree, and the 2D plots (with the cell outlines of show_grid) run on both.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

import core.structural_modeling_components.general as general
import core.structural_modeling_components.general_faults as gf
from core.structural_modeling_components.structural_modeling_visualization import (
    structural_modeling_visualization as viz,
)

pytest.importorskip("gempy")  # faults are fitted with GemPy


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    backend = plt.get_backend()
    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")
    plt.switch_backend(backend)


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
            out[name] = general.compute_structural_model(frame, extract_meshes=False, verbose=False).structural_frame
    return out


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_octree_sections_are_the_regular_grids_slices(results, axis):
    octree, regular = results["octree"], results["regular"]
    for index in (0, regular.grid.resolution[axis] // 2, regular.grid.resolution[axis] - 1):
        np.testing.assert_array_equal(octree.section(octree.lith_block, axis, index),
                                      regular.section(regular.lith_block, axis, index))
        for o_group, r_group in zip(octree.structural_groups, regular.structural_groups):
            np.testing.assert_array_equal(octree.section(o_group.mask, axis, index),
                                          regular.section(r_group.mask, axis, index))
        np.testing.assert_array_equal(octree.fault_frame.section(octree.fault_frame.domain_map, axis, index),
                                      regular.fault_frame.section(regular.fault_frame.domain_map, axis, index))


def test_a_section_outside_the_grid_is_refused(results):
    octree = results["octree"]
    with pytest.raises(IndexError):
        octree.section(octree.lith_block, 0, octree.grid.finest_resolution[0])


@pytest.mark.parametrize("kind", ["octree", "regular"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_section_cells_tile_the_section(results, kind, axis):
    frame = results[kind]
    index = 3
    rects = viz._section_cells(frame, axis, index)
    a, b = [k for k in range(3) if k != axis]
    extent = np.array(frame.grid.extent).reshape(3, 2)
    areas = (rects[:, 2, 0] - rects[:, 0, 0]) * (rects[:, 2, 1] - rects[:, 0, 1])
    np.testing.assert_allclose(areas.sum(), np.ptp(extent[a]) * np.ptp(extent[b]), rtol=1e-12)
    if kind == "regular":
        assert len(rects) == frame.grid.resolution[a] * frame.grid.resolution[b]
    else:
        # Exactly the cells whose span along the axis contains the plane, fewer than the regular slice.
        plane = frame.grid.origin[axis] + (index + 0.5) * frame.grid.finest_spacing[axis]
        mins, maxs = frame.grid.cell_bounds(frame.cells.level, frame.cells.ijk)
        assert len(rects) == np.sum((mins[:, axis] < plane) & (maxs[:, axis] > plane))
        assert len(rects) < np.prod(np.array(frame.grid.finest_resolution)[[a, b]])


def _outlines(ax):
    return [c for c in ax.collections if type(c) is PolyCollection]  # Quiver arrows subclass it


@pytest.mark.parametrize("kind", ["octree", "regular"])
@pytest.mark.parametrize("show_grid", [False, True])
def test_structural_model_2d_plots_on_both_grids(results, kind, show_grid):
    frame = results[kind]
    for axis in ("x", "y", "z"):
        viz.plot_structural_model_2D(frame, axis=axis, show_grid=show_grid)
        ax = plt.gcf().axes[0]
        assert ax.images, axis  # the lithology
        assert len(_outlines(ax)) == (1 if show_grid else 0)
        plt.close("all")


@pytest.mark.parametrize("kind", ["octree", "regular"])
@pytest.mark.parametrize("show_grid", [False, True])
def test_fault_model_2d_plots_on_both_grids(results, kind, show_grid):
    fault_frame = results[kind].fault_frame
    viz.plot_fault_model_2D(fault_frame, axis="y", show_grid=show_grid)
    ax = plt.gcf().axes[0]
    assert ax.images
    assert len(_outlines(ax)) == (1 if show_grid else 0)


@pytest.mark.parametrize("kind", ["octree", "regular"])
def test_frame_section_plots_run_on_both_grids(results, kind):
    frame = results[kind]
    frame.plot_scalar_field_section(group_nr=1, axis="y", index=3)
    frame.plot_age_mask_section(group_nr=0, axis="x", index=3)
