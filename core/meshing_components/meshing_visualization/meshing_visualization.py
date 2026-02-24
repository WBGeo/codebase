from __future__ import annotations

import warnings
from typing import List

import pyvista as pv

from core.object_components import MeshResults, StructuralModelResults


_FALLBACK_COLORS = [
    '#673ab7', '#34a853', '#ea4335', '#fbbc05', '#4285f4',
    '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
    '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5',
]


def plot_mesh_3d(
    mesh_results: MeshResults,
    geomodel_result: StructuralModelResults,
    style: str = "surface",
    show_plotter: bool = True,
) -> pv.Plotter:
    """
    Plot the mesh for process simulation in 3D.

    Args:
        mesh_results: The mesh to plot.
        geomodel_result: Structural model result providing element labels and colors.
        style: PyVista mesh style ('surface', 'wireframe', or 'points').
        show_plotter: If True, display the interactive plot.

    Returns:
        The configured PyVista Plotter instance.
    """
    warnings.filterwarnings(
        "ignore",
        message="Points is not a float type. This can cause issues when transforming or applying "
                "filters. Casting to ``np.float32``. Disable this by passing "
                "``force_float=False``.",
    )

    frame = geomodel_result.structural_frame

    # Collect (name, color) youngest→oldest across all structural groups only.
    # Faults are meshed as surfaces (triangles), not volumes, so they have no
    # corresponding MultiBlock entry and must not be included here.
    # structural_elements is ordered youngest→oldest; mesh blocks are ordered by
    # ascending surface_id (oldest→youngest), so we reverse before assigning.
    labels_fwd: List[str] = []
    colors_fwd: List[str] = []

    idx = 0
    for group in frame.structural_groups:
        for elem in group.structural_elements:
            labels_fwd.append(elem.name)
            colors_fwd.append(
                elem.color if elem.color is not None
                else _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
            )
            idx += 1

    # Blocks are ordered bottom→top: basement first (block 0), then geological
    # layers from oldest to youngest. structural_elements is youngest→oldest,
    # so reverse it and prepend the basement entry.
    labels: List[str] = ["basement"] + list(reversed(labels_fwd))
    colors: List[str] = ["#808080"] + list(reversed(colors_fwd))

    plotter = pv.Plotter(off_screen=not show_plotter)

    n_blocks = mesh_results.mesh.n_blocks
    plotted: List[int] = []
    for i in range(n_blocks):
        block = mesh_results.mesh[i]
        if block is None:
            continue
        color = colors[i] if i < len(colors) else _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        plotter.add_mesh(block, show_edges=True, style=style, color=color)
        plotted.append(i)

    # Legend ordered youngest on top → oldest → basement at the bottom,
    # which is the reverse of the block plot order (bottom→top).
    legend_entries = [
        [labels[i] if i < len(labels) else f"Block {i}",
         colors[i] if i < len(colors) else _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]]
        for i in reversed(plotted)
    ]
    plotter.add_legend(legend_entries, size=(0.13, 0.13), loc='lower right', face='circle')
    plotter.show_bounds(grid=True)
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show_plotter:
        plotter.show()

    return plotter