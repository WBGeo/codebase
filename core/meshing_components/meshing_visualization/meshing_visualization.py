
import warnings
from typing import List

import numpy as np
import pyvista as pv
from core.object_components import MeshResults, StructuralModelResults
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_inspector, InspectorHelper


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

    # -------------------------------------------------
    # BUILD LABELS + COLORS
    # -------------------------------------------------
    labels_fwd: List[str] = []
    colors_fwd: List[str] = []

    idx = 0

    for group in frame.structural_groups:

        for elem in group.structural_elements:

            labels_fwd.append(elem.name)

            colors_fwd.append(
                elem.color
                if elem.color is not None
                else _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
            )

            idx += 1

    labels = ["basement"] + list(reversed(labels_fwd))
    colors = ["#808080"] + list(reversed(colors_fwd))

    # -------------------------------------------------
    # SAFE BLOCK SELECTION
    # Ignore ALL triangle blocks
    # -------------------------------------------------
    n_blocks = mesh_results.mesh.n_blocks

    block_indices = []

    for i in range(n_blocks):

        block = mesh_results.mesh[i]

        if block is None:
            continue

        skip_block = False

        # -------------------------------------------------
        # Skip PolyData triangle surfaces
        # -------------------------------------------------
        if isinstance(block, pv.PolyData):
            print("Skipping PolyData block")
            skip_block = True

        # -------------------------------------------------
        # Skip blocks containing only triangle cells
        # -------------------------------------------------
        else:

            try:
                cell_types = np.unique(block.celltypes)


                # VTK_TRIANGLE = 5
                if len(cell_types) == 1 and cell_types[0] == 5:
                    skip_block = True

            except Exception:
                pass

        if skip_block:
            continue

        block_indices.append(i)

    # -------------------------------------------------
    # PLOTTING
    # -------------------------------------------------
    plotter = pv.Plotter(off_screen=not show_plotter)

    plotted: List[int] = []

    for i in block_indices:

        block = mesh_results.mesh[i]

        if block is None:
            continue

        color = (
            colors[i]
            if i < len(colors)
            else _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        )

        plotter.add_mesh(
            block,
            show_edges=True,
            style=style,
            color=color,
        )

        plotted.append(i)

    # -------------------------------------------------
    # LEGEND
    # -------------------------------------------------
    legend_entries = [
        [
            labels[i] if i < len(labels) else f"Block {i}",
            colors[i]
            if i < len(colors)
            else _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)],
        ]
        for i in reversed(plotted)
    ]

    if legend_entries:

        plotter.add_legend(
            legend_entries,
            size=(0.13, 0.13),
            loc="lower right",
            face="circle",
        )

    # -------------------------------------------------
    # VIEW SETTINGS
    # -------------------------------------------------
    plotter.show_bounds(grid=True)

    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show_plotter:
        plotter.show()

    return plotter

@wbgeo_component(identifier='wbgeo::inspect_mesh_3d',
                 title='Plot Mesh in 3D',
                 description='...')
@wbgeo_inspector()
async def inspect_implicit_mesh_3d(
    mesh_results: MeshResults, _inspector: InspectorHelper):
  # load structural_model_result from the execution trace
  structural_model_result = await (await _inspector.trace(StructuralModelResults)).get_value()
  # and call the render function with both the mesh_results and the object from our trace
  plot_mesh_3d(mesh_results, structural_model_result, "surface", True)
