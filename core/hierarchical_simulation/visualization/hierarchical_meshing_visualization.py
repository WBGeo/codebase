"""
Interactive visualization for hierarchical WBGeo meshes.

Each hierarchical mesh is displayed in its own PyVista subplot.
Every mesh block is added as an independent actor so that it can
be switched ON/OFF interactively, similar to ParaView's block list.
"""

from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv

from core.object_components import MeshResults, StructuralModelResults
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_inspector


# =============================================================================
# Helpers
# =============================================================================

def _block_name(mesh: MeshResults, index: int) -> str:
    """Return a readable name for a block stored in ``MeshResults.mesh``."""

    multiblock = mesh.mesh

    # PyVista MultiBlock can carry explicit block names.
    try:
        name = multiblock.get_block_name(index)
        if name:
            return str(name)
    except (AttributeError, IndexError, TypeError):
        pass

    # WBGeo meshes do not always store names on the MultiBlock itself.
    # In that case use the cell type, which is much more informative than
    # an anonymous number for engineering-object blocks.
    try:
        block = multiblock[index]
        if block is not None and block.n_cells:
            cell_types = np.unique(block.celltypes)
            vtk_names = {
                int(pv.CellType.TETRA): "tetra",
                int(pv.CellType.HEXAHEDRON): "hexahedron",
                int(pv.CellType.VOXEL): "voxel",
                int(pv.CellType.WEDGE): "wedge",
                int(pv.CellType.PYRAMID): "pyramid",
                int(pv.CellType.TRIANGLE): "triangle",
                int(pv.CellType.QUAD): "quad",
                int(pv.CellType.POLYGON): "polygon",
                int(pv.CellType.LINE): "line",
                int(pv.CellType.POLY_LINE): "polyline",
                int(pv.CellType.VERTEX): "vertex",
                int(pv.CellType.POLY_VERTEX): "polyvertex",
            }
            suffix = "_".join(vtk_names.get(int(t), f"vtk{int(t)}") for t in cell_types)
            return f"Block_{index}_{suffix}"
    except (AttributeError, IndexError, TypeError):
        pass

    return f"Block_{index}"


def _block_dimension(block) -> int:
    """Classify a WBGeo/PyVista block by its actual VTK cell types.

    3 = volume, 2 = surface, 1 = line, 0 = vertex.
    WBGeo unstructured meshes use tetra/triangle/line/vertex blocks, but
    the extra VTK types below keep the viewer usable for structured meshes.
    """
    if block is None or block.n_cells == 0:
        return -1

    cell_types = {int(t) for t in np.unique(block.celltypes)}

    vertex_types = {1, 2}                    # VTK_VERTEX, VTK_POLY_VERTEX
    line_types = {3, 4}                      # VTK_LINE, VTK_POLY_LINE
    surface_types = {5, 6, 7, 8, 9}         # TRIANGLE..QUAD
    volume_types = {10, 11, 12, 13, 14}     # TETRA..PYRAMID

    # Pure blocks first.  This is the normal WBGeo case.
    if cell_types.issubset(vertex_types):
        return 0
    if cell_types.issubset(line_types):
        return 1
    if cell_types.issubset(surface_types):
        return 2
    if cell_types.issubset(volume_types):
        return 3

    # Fallback for a mixed VTK block: classify by highest dimension present.
    if cell_types & volume_types:
        return 3
    if cell_types & surface_types:
        return 2
    if cell_types & line_types:
        return 1
    if cell_types & vertex_types:
        return 0
    return -1

def _dimension_label(dimension: int) -> str:

    return {
        3: "volume",
        2: "surface",
        1: "line",
        0: "vertex",
    }.get(dimension, "other")


# =============================================================================
# Block colors
# =============================================================================

# A distinct, deterministic color is assigned to each displayed block.
# The palette repeats only if a mesh contains more blocks than colors.
_BLOCK_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",
]


def _block_color(index: int) -> str:
    """Return a stable display color for a mesh block index."""
    return _BLOCK_COLORS[index % len(_BLOCK_COLORS)]


# =============================================================================
# Block actor creation
# =============================================================================

def _add_blocks(
    plotter: pv.Plotter,
    mesh: MeshResults,
    subplot: Tuple[int, int],
    volume_opacity: float = 1.0,
    omit_last_blocks: int = 6,
):
    """
    Add each visible mesh block as a separate PyVista actor.

    Keeping one actor per block is what allows individual visibility
    switching later.
    """

    row, column = subplot
    plotter.subplot(row, column)

    actors = {}

    # MeshResults is WBGeo's wrapper.  The actual PyVista MultiBlock used by
    # the normal WBGeo visualizer is stored in ``MeshResults.mesh``.
    multiblock = mesh.mesh

    # WBGeo appends six auxiliary/boundary blocks that are intentionally
    # excluded from the hierarchical visualization.
    stop = max(0, multiblock.n_blocks - max(0, int(omit_last_blocks)))

    for i in range(stop):

        block = multiblock[i]

        if block is None or block.n_cells == 0:
            continue

        name = _block_name(mesh, i)
        dimension = _block_dimension(block)
        color = _block_color(i)

        # -------------------------------------------------------------
        # Rendering properties by geometric dimension
        # -------------------------------------------------------------

        if dimension == 3:

            actor = plotter.add_mesh(
                block,
                color=color,
                style="surface",
                show_edges=True,
                opacity=volume_opacity,
                label=name,
            )

        elif dimension == 2:

            actor = plotter.add_mesh(
                block,
                color=color,
                style="surface",
                show_edges=True,
                opacity=1.0,
                label=name,
            )

        elif dimension == 1:

            actor = plotter.add_mesh(
                block,
                color=color,
                line_width=5,
                render_lines_as_tubes=True,
                label=name,
            )

        elif dimension == 0:

            actor = plotter.add_mesh(
                block,
                color=color,
                point_size=12,
                render_points_as_spheres=True,
                label=name,
            )

        else:

            actor = plotter.add_mesh(
                block,
                color=color,
                show_edges=True,
                label=name,
            )

        actors[name] = {
            "actor": actor,
            "dimension": dimension,
            "block_index": i,
            "color": color,
        }

    return actors


# =============================================================================
# Checkbox callbacks
# =============================================================================

def _visibility_callback(plotter, actor):

    def callback(state):

        actor.SetVisibility(bool(state))
        plotter.render()

    return callback


def _group_visibility_callback(
    plotter,
    actors,
    dimension,
):

    def callback(state):

        for data in actors.values():

            if data["dimension"] == dimension:
                data["actor"].SetVisibility(bool(state))

        plotter.render()

    return callback


def _all_visibility_callback(
    plotter,
    actors,
    state_to_set,
):

    def callback(_state):

        for data in actors.values():
            data["actor"].SetVisibility(state_to_set)

        plotter.render()

    return callback


# =============================================================================
# Checkbox panel
# =============================================================================

def _add_block_panel(
    plotter: pv.Plotter,
    actors: Dict,
    subplot: Tuple[int, int],
    start_x: int = 10,
    start_y: int = 650,
):

    row, column = subplot
    plotter.subplot(row, column)

    # -----------------------------------------------------------------
    # Group controls
    # -----------------------------------------------------------------

    y = start_y

    groups = [
        ("Volumes", 3),
        ("Surfaces", 2),
        ("Lines", 1),
        ("Vertices", 0),
    ]

    for label, dimension in groups:

        plotter.add_checkbox_button_widget(
            _group_visibility_callback(
                plotter,
                actors,
                dimension,
            ),
            value=True,
            position=(start_x, y),
            size=18,
        )

        plotter.add_text(
            label,
            position=(start_x + 25, y),
            font_size=9,
            name=f"group_{row}_{column}_{dimension}",
        )

        y -= 28

    # -----------------------------------------------------------------
    # Individual block controls
    # -----------------------------------------------------------------

    y -= 15

    for name, data in actors.items():

        dimension = data["dimension"]

        plotter.add_checkbox_button_widget(
            _visibility_callback(
                plotter,
                data["actor"],
            ),
            value=True,
            position=(start_x, y),
            size=16,
        )

        label = f"{name} [{_dimension_label(dimension)}]"

        plotter.add_text(
            label,
            position=(start_x + 22, y),
            font_size=7,
            name=f"block_{row}_{column}_{data['block_index']}",
        )

        y -= 23

        # Prevent widgets running below the window.
        if y < 20:
            break


# =============================================================================
# Main visualization
# =============================================================================

def plot_hierarchical_meshes_3d(
    mesh1: MeshResults,
    mesh2: MeshResults,
    geomodel_result: StructuralModelResults,
    style: str = "surface",
    show_plotter: bool = True,
    link_views: bool = True,
    volume_opacity: float = 1.0,
    show_block_controls: bool = True,
    omit_last_blocks: int = 6,
) -> pv.Plotter:
    """
    Plot two hierarchical meshes with interactive block visibility.

    Parameters
    ----------
    mesh1
        Bigger hierarchical mesh.

    mesh2
        Smaller hierarchical mesh.

    geomodel_result
        WBGeo structural model result.

    style
        Kept for compatibility with the existing WBGeo visualization API.

    show_plotter
        Immediately open the PyVista window.

    link_views
        Link the cameras of the two hierarchical views.

    volume_opacity
        Opacity of 3-D blocks. Set to 1.0 for opaque volumes.
        Values such as 0.2 make internal surfaces and lines easier to see.

    show_block_controls
        Add individual block ON/OFF controls.

    omit_last_blocks
        Number of trailing WBGeo auxiliary blocks to ignore completely.
        The hierarchical viewer defaults to 6.
    """

    plotter = pv.Plotter(
        shape=(1, 2),
        window_size=(1500, 850),
    )

    # =================================================================
    # Mesh 1
    # =================================================================

    plotter.subplot(0, 0)

    actors1 = _add_blocks(
        plotter,
        mesh1,
        subplot=(0, 0),
        volume_opacity=volume_opacity,
        omit_last_blocks=omit_last_blocks,
    )

    plotter.add_text(
        "Hierarchical mesh 1 (bigger)",
        position="upper_edge",
        font_size=14,
    )

    plotter.show_axes()

    if show_block_controls:

        _add_block_panel(
            plotter,
            actors1,
            subplot=(0, 0),
            start_x=10,
            start_y=700,
        )

    # =================================================================
    # Mesh 2
    # =================================================================

    plotter.subplot(0, 1)

    actors2 = _add_blocks(
        plotter,
        mesh2,
        subplot=(0, 1),
        volume_opacity=volume_opacity,
        omit_last_blocks=omit_last_blocks,
    )

    plotter.add_text(
        "Hierarchical mesh 2 (smaller)",
        position="upper_edge",
        font_size=14,
    )

    plotter.show_axes()

    if show_block_controls:

        _add_block_panel(
            plotter,
            actors2,
            subplot=(0, 1),
            start_x=10,
            start_y=700,
        )

    # =================================================================
    # Camera
    # =================================================================

    if link_views:
        plotter.link_views()

    plotter.subplot(0, 0)
    plotter.view_isometric()

    plotter.subplot(0, 1)
    plotter.view_isometric()

    # Keep references available for debugging / external manipulation.
    plotter._hierarchical_actors_mesh1 = actors1
    plotter._hierarchical_actors_mesh2 = actors2

    if show_plotter:
        plotter.show()

    return plotter


# =============================================================================
# WBGeo inspector
# =============================================================================

@wbgeo_component(
    identifier="wbgeo::inspect_hierarchical_meshes_3d",
    title="Inspect Hierarchical Meshes 3D",
    description=(
        "Interactive visualization of the bigger and smaller hierarchical "
        "meshes with individual block visibility controls."
    ),
    color="#A52A2A",
    border_color="#000000",
    group="Hierarchical Simulation",
)
@wbgeo_inspector()
def inspect_hierarchical_meshes_3d(
    mesh1: MeshResults,
    mesh2: MeshResults,
    geomodel_result: StructuralModelResults,
):

    return plot_hierarchical_meshes_3d(
        mesh1=mesh1,
        mesh2=mesh2,
        geomodel_result=geomodel_result,
        show_plotter=True,
        link_views=True,
        volume_opacity=1.0,
        show_block_controls=True,
    )
