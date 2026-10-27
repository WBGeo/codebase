from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # headless -- no display needed/available in CI

from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_fault_model_2D,
    plot_fault_model_3D,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultElement, FaultFrame, FaultMeshType,
)


def _fault_frame_with_scalar_fields(domain_map) -> FaultFrame:
    """A minimal FaultFrame with a real scalar field/value but no dependency
    on compute_fault_domains -- isolates plot_fault_model_2D's own logic
    from the interpolation pipeline."""
    grid = RegularGrid(extent=(0, 1, 0, 1, 0, 1), resolution=(4, 4, 4))
    ff = FaultFrame(fault_elements=[FaultElement("F1", color="#ff0000")])
    ff.set_grid(grid)

    xs, _, _ = np.meshgrid(
        np.linspace(0, 1, 4), np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing="ij"
    )
    ff.fault_elements[0].set_scalar_field(xs)  # scalar field == X coordinate
    ff.fault_elements[0].set_scalar_value(0.5)  # isovalue at X=0.5

    ff.fault_surface_points_df = pd.DataFrame(
        {"X": [0.5], "Y": [0.5], "Z": [0.5], "formation": ["F1"]}
    )
    ff.domain_map = domain_map
    return ff


def test_fault_contours_drawn_without_domain_map_warns_but_does_not_raise():
    # domain_map=None mirrors what compute_fault_domains leaves when it
    # detects a cross-cut (domaining skipped entirely) -- the fix under test
    # is that fault contours still draw in this case, not just input points.
    ff = _fault_frame_with_scalar_fields(domain_map=None)

    with pytest.warns(UserWarning, match="cross-cutting"):
        plot_fault_model_2D(ff, show_results=True, show_fault_contours=True)

    # The contour itself was actually drawn (not just "didn't crash") --
    # ax.contour() on a non-degenerate crossing adds collections to the axes.
    fig = _last_figure()
    ax = fig.axes[0]
    assert len(ax.collections) > 0, "Expected a fault contour to be drawn despite domain_map=None"


def test_fault_contours_drawn_with_domain_map_present_no_warning(recwarn):
    domain_map = np.zeros((4, 4, 4), dtype=int)
    ff = _fault_frame_with_scalar_fields(domain_map=domain_map)

    plot_fault_model_2D(ff, show_results=True, show_fault_contours=True)

    assert not any("cross-cutting" in str(w.message) for w in recwarn.list)


def _last_figure():
    import matplotlib.pyplot as plt
    return plt.gcf()


# -----------------------------------------------------------------------------
# plot_fault_model_3D -- crosscut-line highlighting
# -----------------------------------------------------------------------------

def _rect_plane(corner_a, corner_b, corner_c, corner_d) -> tuple[np.ndarray, np.ndarray]:
    """Two-triangle rectangular plane (a,b,c,d given in order around the rect)."""
    vertices = np.array([corner_a, corner_b, corner_c, corner_d], dtype=float)
    edges = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, edges


def _meshed_fault_frame(*fault_specs: tuple[str, np.ndarray, np.ndarray]) -> FaultFrame:
    """A FaultFrame with hand-built meshes directly on each fault -- bypasses
    compute_fault_domains/marching_cubes entirely, isolating the crosscut-line
    highlighting logic from both interpolation and surface extraction."""
    grid = RegularGrid(extent=(0, 1, 0, 1, 0, 1), resolution=(4, 4, 4))
    elements = []
    for name, vertices, edges in fault_specs:
        elem = FaultElement(name, color="#336699")
        elem.set_mesh(FaultMeshType.UNMASKED, vertices, edges)
        elements.append(elem)
    ff = FaultFrame(fault_elements=elements)
    ff.set_grid(grid)
    return ff


def test_plot_fault_model_3D_highlights_crosscut_line_for_crossing_faults():
    # Vertical plane at X=0.5 (spans Y,Z) and horizontal plane at Z=0.5
    # (spans X,Y) -- these necessarily cross at X=0.5,Z=0.5 for all Y.
    v1, e1 = _rect_plane((0.5, 0, 0), (0.5, 1, 0), (0.5, 1, 1), (0.5, 0, 1))
    v2, e2 = _rect_plane((0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5))
    ff = _meshed_fault_frame(("F_vert", v1, e1), ("F_horiz", v2, e2))

    plotter = plot_fault_model_3D(ff, show=False, notebook=True)

    crosscut_actors = [name for name in plotter.renderer.actors if "crosscut" in name]
    assert len(crosscut_actors) == 1, f"Expected exactly one crosscut line, got {crosscut_actors}"


def test_plot_fault_model_3D_no_crosscut_line_for_parallel_faults():
    # Two parallel vertical planes at X=0.3 and X=0.7 -- never cross.
    v1, e1 = _rect_plane((0.3, 0, 0), (0.3, 1, 0), (0.3, 1, 1), (0.3, 0, 1))
    v2, e2 = _rect_plane((0.7, 0, 0), (0.7, 1, 0), (0.7, 1, 1), (0.7, 0, 1))
    ff = _meshed_fault_frame(("F_a", v1, e1), ("F_b", v2, e2))

    plotter = plot_fault_model_3D(ff, show=False, notebook=True)

    crosscut_actors = [name for name in plotter.renderer.actors if "crosscut" in name]
    assert crosscut_actors == []


def test_plot_fault_model_3D_highlight_crosscuts_can_be_disabled():
    v1, e1 = _rect_plane((0.5, 0, 0), (0.5, 1, 0), (0.5, 1, 1), (0.5, 0, 1))
    v2, e2 = _rect_plane((0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5))
    ff = _meshed_fault_frame(("F_vert", v1, e1), ("F_horiz", v2, e2))

    plotter = plot_fault_model_3D(ff, show=False, notebook=True, highlight_crosscuts=False)

    crosscut_actors = [name for name in plotter.renderer.actors if "crosscut" in name]
    assert crosscut_actors == []
