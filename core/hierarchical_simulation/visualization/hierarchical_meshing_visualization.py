"""
Interactive visualization for hierarchical WBGeo meshes.

Each hierarchical mesh is displayed in its own PyVista subplot.
Every mesh block is added as an independent actor so that it can
be switched ON/OFF interactively, similar to ParaView's block list.
"""

from typing import Dict, List, Tuple, Optional

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
            _group_visibility_callback( plotter, actors, dimension, ),
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
            _visibility_callback( plotter, data["actor"], ),
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
# Interactive inspection helpers
# =============================================================================

def _mesh_global_node_id(mesh: MeshResults, point: np.ndarray) -> Optional[int]:
    """Return the nearest WBGeo node index for a picked coordinate."""
    try:
        nodes = np.asarray(mesh.nodes, dtype=float)
        if nodes.ndim == 2 and nodes.shape[1] >= 3 and len(nodes):
            return int(np.argmin(np.linalg.norm(nodes[:, :3] - point[:3], axis=1)))
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def _nearest_block_point(mesh: MeshResults, point: np.ndarray, omit_last_blocks: int = 6):
    """Find the displayed block and local point closest to ``point``."""
    multiblock = mesh.mesh
    stop = max(0, multiblock.n_blocks - max(0, int(omit_last_blocks)))
    best = None
    for i in range(stop):
        block = multiblock[i]
        if block is None or block.n_points == 0:
            continue
        pts = np.asarray(block.points)
        local_id = int(np.argmin(np.linalg.norm(pts - point[:3], axis=1)))
        distance = float(np.linalg.norm(pts[local_id] - point[:3]))
        if best is None or distance < best[0]:
            best = (distance, i, local_id, block)
    return best


def _array_values_at(dataset, association: str, index: int) -> List[str]:
    """Format point/cell arrays for the information panel."""
    values = []
    data = dataset.point_data if association == "point" else dataset.cell_data
    for name in data.keys():
        try:
            value = np.asarray(data[name])[index]
            if np.ndim(value) == 0:
                rendered = f"{value}"
            else:
                rendered = np.array2string(np.asarray(value), precision=5, separator=", ")
            values.append(f"{name}: {rendered}")
        except (IndexError, TypeError, ValueError):
            continue
    return values


def _set_info_text(plotter: pv.Plotter, subplot: Tuple[int, int], text: str):
    """Update the persistent selection-information panel in one subplot."""
    row, column = subplot
    plotter.subplot(row, column)
    plotter.add_text(
        text,
        position="lower_left",
        font_size=9,
        name=f"hierarchical_selection_info_{row}_{column}",
    )


def _make_point_callback(plotter, mesh, subplot, mesh_label, omit_last_blocks):
    """Create a callback reporting node id, coordinates, block and point arrays."""
    def callback(point):
        if point is None:
            return
        point = np.asarray(point, dtype=float)
        nearest = _nearest_block_point(mesh, point, omit_last_blocks)
        if nearest is None:
            return
        _, block_index, local_id, block = nearest
        xyz = np.asarray(block.points[local_id], dtype=float)
        global_id = _mesh_global_node_id(mesh, xyz)
        lines = [
            f"{mesh_label} - POINT",
            f"Block: {_block_name(mesh, block_index)} (index {block_index})",
            f"Block point ID: {local_id}",
            f"Node ID: {global_id if global_id is not None else 'n/a'}",
            f"X: {xyz[0]:.6g}",
            f"Y: {xyz[1]:.6g}",
            f"Z: {xyz[2]:.6g}",
        ]
        lines.extend(_array_values_at(block, "point", local_id)[:6])
        _set_info_text(plotter, subplot, "\n".join(lines))
        plotter.render()
    return callback


def _make_cell_callback(plotter, mesh, subplot, mesh_label, omit_last_blocks):
    """Create a callback reporting the closest original cell and its data."""
    def callback(picked):
        if picked is None or getattr(picked, "n_cells", 0) == 0:
            return
        try:
            center = np.asarray(picked.cell_centers().points[0], dtype=float)
        except (AttributeError, IndexError, ValueError):
            return

        multiblock = mesh.mesh
        stop = max(0, multiblock.n_blocks - max(0, int(omit_last_blocks)))
        best = None
        for i in range(stop):
            block = multiblock[i]
            if block is None or block.n_cells == 0:
                continue
            centers = np.asarray(block.cell_centers().points)
            cell_id = int(np.argmin(np.linalg.norm(centers - center, axis=1)))
            distance = float(np.linalg.norm(centers[cell_id] - center))
            if best is None or distance < best[0]:
                best = (distance, i, cell_id, block, centers[cell_id])
        if best is None:
            return

        _, block_index, cell_id, block, cell_center = best
        cell = block.get_cell(cell_id)
        point_ids = list(getattr(cell, "point_ids", []))
        lines = [
            f"{mesh_label} - CELL",
            f"Block: {_block_name(mesh, block_index)} (index {block_index})",
            f"Cell ID: {cell_id}",
            f"Cell type: {getattr(cell, 'type', 'n/a')}",
            f"Center: ({cell_center[0]:.6g}, {cell_center[1]:.6g}, {cell_center[2]:.6g})",
            f"Point IDs: {point_ids[:12]}{' ...' if len(point_ids) > 12 else ''}",
        ]
        lines.extend(_array_values_at(block, "cell", cell_id)[:6])
        _set_info_text(plotter, subplot, "\n".join(lines))
        plotter.render()
    return callback


def _enable_hierarchical_point_mode(
    plotter,
    mesh1,
    mesh2,
    omit_last_blocks,
):
    """Enable one point picker shared by both hierarchical subplots.

    PyVista maintains a single picking mode for a Plotter. Enabling picking
    once for Mesh 1 and then again for Mesh 2 disables/replaces the first
    picker. This shared callback determines which viewport was clicked and
    dispatches the picked point to the correct mesh.
    """

    try:
        plotter.disable_picking()
    except (AttributeError, RuntimeError):
        pass

    cb1 = _make_point_callback(
        plotter, mesh1, (0, 0), "Mesh 1", omit_last_blocks
    )
    cb2 = _make_point_callback(
        plotter, mesh2, (0, 1), "Mesh 2", omit_last_blocks
    )

    def callback(point):
        if point is None:
            return

        # For the 1x2 layout, use the mouse x-position to determine which
        # renderer received the click. This avoids the second PyVista picker
        # replacing the first one.
        try:
            x, _ = plotter.iren.get_event_position()
            width, _ = plotter.window_size
            if x < width / 2:
                cb1(point)
            else:
                cb2(point)
        except Exception:
            # Safe fallback: choose the mesh whose displayed point is nearest.
            p = np.asarray(point, dtype=float)
            n1 = _nearest_block_point(mesh1, p, omit_last_blocks)
            n2 = _nearest_block_point(mesh2, p, omit_last_blocks)
            d1 = np.inf if n1 is None else n1[0]
            d2 = np.inf if n2 is None else n2[0]
            (cb1 if d1 <= d2 else cb2)(point)

    # Enable picking only once. The clicked renderer is selected by VTK/PyVista.
    plotter.subplot(0, 0)
    plotter.enable_point_picking(
        callback=callback,
        show_message=False,
        show_point=True,
        point_size=14,
        color="yellow",
        left_clicking=True,
        pickable_window=False,
    )

    _set_info_text( plotter, (0, 0), "Mesh 1\nPOINT mode: left-click a mesh point\nX/Y/Z appear here" )
    _set_info_text( plotter, (0, 1), "Mesh 2\nPOINT mode: left-click a mesh point\nX/Y/Z appear here" )


def _enable_cell_mode(plotter, mesh, subplot, mesh_label, omit_last_blocks):
    """Enable cell inspection for the active subplot."""
    plotter.subplot(*subplot)
    try:
        plotter.disable_picking()
    except (AttributeError, RuntimeError):
        pass
    plotter.enable_cell_picking(
        callback=_make_cell_callback(plotter, mesh, subplot, mesh_label, omit_last_blocks),
        through=False,
        show_message=False,
        show=True,
        style="wireframe",
        color="yellow",
    )
    _set_info_text(plotter, subplot, f"{mesh_label}\nCELL mode: pick a cell\nPress P for point mode")

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
    enable_inspection: bool = True,
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

    enable_inspection
        Add ParaView-like interactive inspection. Point mode is enabled by
        default. Left-click a point to display node ID, XYZ coordinates, block
        name and point arrays. Press ``P`` for point mode and ``C`` for cell
        mode in the currently active subplot. Cell mode reports cell ID, cell
        type, center, point IDs and cell arrays.
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

    plotter.add_text( "Hierarchical mesh 1 (bigger)", position="upper_edge", font_size=14, )

    plotter.show_axes()

    if show_block_controls:

        _add_block_panel( plotter, actors1, subplot=(0, 0), start_x=10, start_y=700, )

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

    plotter.add_text( "Hierarchical mesh 2 (smaller)", position="upper_edge", font_size=14, )

    plotter.show_axes()

    if show_block_controls:

        _add_block_panel( plotter, actors2, subplot=(0, 1), start_x=10, start_y=700, )

    # =================================================================
    # ParaView-like inspection
    # =================================================================

    if enable_inspection:
        # Point picking is the default inspection mode.  P/C switch the
        # currently active renderer between point and cell inspection.
        _enable_hierarchical_point_mode( plotter, mesh1, mesh2, omit_last_blocks, )

        def point_mode_active_subplot():
            index = getattr(plotter, "_active_renderer_index", 0)
            if index == 1:
                _enable_point_mode(plotter, mesh2, (0, 1), "Mesh 2", omit_last_blocks)
            else:
                _enable_point_mode(plotter, mesh1, (0, 0), "Mesh 1", omit_last_blocks)

        def cell_mode_active_subplot():
            index = getattr(plotter, "_active_renderer_index", 0)
            if index == 1:
                _enable_cell_mode(plotter, mesh2, (0, 1), "Mesh 2", omit_last_blocks)
            else:
                _enable_cell_mode(plotter, mesh1, (0, 0), "Mesh 1", omit_last_blocks)

        plotter.add_key_event("p", point_mode_active_subplot)
        plotter.add_key_event("c", cell_mode_active_subplot)

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
        enable_inspection=True,
    )
