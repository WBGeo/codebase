"""
Visualization for the experimental SfePy hydrothermal simulation sandbox:
plotting SimulationResults (3D snapshots, cross-sections, before/after
diffs) and a pre-flight material-assignment check for
HydrothermalProblemBuilder before running a solve that can take a while.

Consolidated into one file (previously split across results_visualisation.py,
comparison_plots.py, builder_visualisation.py) -- these are all facets of
the same "look at this simulation pipeline" concern, not independent
modules, and didn't need three separate files.

PyVista styling (camera angle, show_bounds, legend face style) follows the
conventions established in
core.structural_modeling_components.structural_modeling_visualization
(used consistently across all of its 3D plots, and by
core.meshing_components.meshing_visualization.plot_mesh_3d) for a
consistent look across the codebase's 3D plots, rather than this package's
own ad-hoc defaults.
"""
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

from core.meshing_components.mesh_format.vtk.VTK_format import VTKInputs
from core.object_components import SimulationResults
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    HydrothermalProblemBuilder,
    RockUnitProperties,
)

logger = logging.getLogger(__name__)

# Deliberately not used for any lithology -- the fault zone should always
# stand out distinctly rather than risk blending in with a lithology's own
# color (see plot_builder_materials).
_FAULT_ZONE_COLOR = "#111111"

# Human-readable label (with units, where known) for a solved variable's
# raw name -- falls back to the raw name itself for anything not listed
# here (e.g. a future variable, or cell_data like block_id), so this never
# needs to be kept in lockstep with every possible var_name.
_VARIABLE_LABELS = {
    "T": "Temperature (°C)",
    "p": "Pressure (Pa)",
}


def _variable_label(var_name: str) -> str:
    """Human-readable axis/colorbar label (with units) for a solved variable's raw name."""
    return _VARIABLE_LABELS.get(var_name, var_name)


def _format_rock_properties(props: RockUnitProperties) -> str:
    """
    Compact one-line summary of a RockUnitProperties for legend labels --
    porosity, permeability, solid thermal conductivity, solid volumetric
    heat capacity, in that order (matching the constructor's own field
    order). Scientific notation for permeability/heat capacity, which
    routinely span many orders of magnitude (e.g. 1e-19 to 1e-13 m^2),
    plain notation for porosity/conductivity, which don't.

    Plain-ASCII labels (poro/perm/k_s/rhoc), not Greek symbols (phi/k/
    lambda/rho*c) -- PyVista's legend text actor doesn't reliably render
    non-ASCII glyphs (verified: phi/lambda/rho showed up blank or as the
    wrong character in an actual rendered screenshot), so this isn't a
    style choice, it's required for the text to display correctly.
    """
    return (
        f"poro={props.porosity:g} "
        f"perm={props.permeability:.1e}m2 "
        f"k_s={props.k_solid:g}W/mK "
        f"rhoc={props.rho_c_solid:.1e}J/m3K"
    )


def _apply_camera_convention(plotter: pv.Plotter) -> None:
    """
    Same fixed camera angle used throughout
    structural_modeling_visualization.py's 3D plots and
    meshing_visualization.plot_mesh_3d -- applied here too so this
    package's 3D plots default to the same view instead of PyVista's own
    default, for a consistent look across the codebase.
    """
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0


# ----------------------------------------------------------------------
# Grid construction
# ----------------------------------------------------------------------

def build_grid_from_class(sim: SimulationResults, time: float) -> pv.UnstructuredGrid:
    """
    Build a PyVista grid for one saved time step of a SimulationResults --
    an UnstructuredGrid if `sim` has real cell connectivity for `time`, else a
    bare PolyData point cloud (nodes only). Attaches every variable in
    `sim.node_data_by_time[time]` as point data. Shared by every plotting
    function in this module rather than each reimplementing the conversion.

    Raises:
        ValueError: `time` isn't a key in `sim.nodes_by_time`.
    """
    if time not in sim.nodes_by_time:
        raise ValueError(f"Time {time} not found in simulation results.")

    nodes = sim.nodes_by_time[time]
    cells = sim.cells_by_time.get(time, None)
    celltypes = sim.celltypes_by_time.get(time, None)

    if cells is None or len(cells) == 0:
        mesh = pv.PolyData(nodes)
    else:
        mesh = pv.UnstructuredGrid(cells, celltypes, nodes)

    for name, arr in sim.node_data_by_time.get(time, {}).items():
        mesh.point_data[name] = arr
    for name, arr in sim.cell_data_by_time.get(time, {}).items():
        mesh.cell_data[name] = arr

    return mesh


# ----------------------------------------------------------------------
# Single-snapshot plots
# ----------------------------------------------------------------------

def plot_variable_at_a_time(
    sim: SimulationResults,
    var_name: str,
    time: float,
    cmap: str = "viridis",
    show_edges: bool = False,
    edge_color: str = "grey",
    edge_line_width: float = 0.5,
    show_contours: bool = False,
    n_contours: int = 10,
    contour_color: str = "black",
    scale: Tuple[float, float, float] = (1, 1, 1),
) -> None:
    """
    show_edges draws the mesh's own cell edges on top of the colored
    surface -- edge_color/edge_line_width default to a thin grey rather
    than PyVista's own default (solid black at the default width), which
    otherwise visually overpowers the simulation result underneath it,
    especially on a fine mesh.

    show_contours additionally draws isolines of var_name directly on the
    3D surface (via PyVista's own surf.contour(), which works on point
    data defined on a 2D surface just as well as the more common
    isosurface-of-a-3D-volume use case -- verified empirically) -- a 3D
    analogue of the contour lines plot_cross_section_2D already draws in
    its 2D matplotlib slices.
    """
    grid = build_grid_from_class(sim, time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}.")

    # algorithm=None (not the default "dataset_surface") -- the default
    # produced an empty mesh (0 points) for this pipeline's SfePy-exported
    # grids, verified empirically.
    surf = grid.extract_surface(algorithm=None)
    surf.points *= scale

    plotter = pv.Plotter()
    plotter.add_mesh(
        surf, scalars=var_name, cmap=cmap,
        show_edges=show_edges, edge_color=edge_color, line_width=edge_line_width,
        # add_mesh(scalars=...) already adds its own scalar bar -- setting
        # its title here (rather than a separate add_scalar_bar() call)
        # avoids a second, redundant bar with a mismatched colormap
        # (verified empirically: add_scalar_bar() called separately doesn't
        # pick up this mesh's own cmap).
        scalar_bar_args={"title": _variable_label(var_name)},
    )
    if show_contours:
        if var_name not in surf.point_data:
            raise ValueError(f"show_contours=True needs '{var_name}' as point data, only found as cell data.")
        contours = surf.contour(isosurfaces=n_contours, scalars=var_name)
        # show_scalar_bar=False: contour() leaves var_name as the returned
        # mesh's active scalars, which would otherwise add a second,
        # redundant scalar bar even though color= is a fixed solid color
        # here, not scalars= -- verified empirically.
        plotter.add_mesh(contours, color=contour_color, line_width=2, show_scalar_bar=False)
    plotter.add_text(f"{_variable_label(var_name)} at time {time}", font_size=20, position="upper_edge")
    plotter.show_bounds(location="furthest", grid=True)
    _apply_camera_convention(plotter)
    plotter.show()


# ----------------------------------------------------------------------
# Comparison / diff plots -- isolate what actually changed between two
# time steps, rather than relying on eyeballing two separate absolute
# snapshots (see plot_variable_difference's docstring for why that's
# often not enough to catch a real-but-modest change).
# ----------------------------------------------------------------------

def plot_variable_difference(
    sim: SimulationResults,
    var_name: str,
    time_a: Optional[float] = None,
    time_b: Optional[float] = None,
    cmap: str = "RdBu_r",
    show_edges: bool = False,
    scale: Tuple[float, float, float] = (1, 1, 1),
) -> None:
    """
    Plot the change in var_name between two time steps (time_b - time_a),
    defaulting to the first and last saved steps. Uses a diverging colormap
    with color limits symmetric around zero, so increases/decreases are
    visually distinguishable and "no change" reads as the cmap's neutral
    midpoint rather than an arbitrary color -- unlike plotting the two
    absolute snapshots separately, where both are auto-scaled to the same
    [T_top, T_bottom]-type range and a real but modest change can be easy to
    miss against the dominant background gradient.
    """
    times = sorted(sim.nodes_by_time.keys())
    if not times:
        raise ValueError("sim has no time steps.")
    if time_a is None:
        time_a = times[0]
    if time_b is None:
        time_b = times[-1]

    grid_a = build_grid_from_class(sim, time_a)
    grid_b = build_grid_from_class(sim, time_b)

    if var_name not in grid_a.point_data or var_name not in grid_b.point_data:
        raise ValueError(f"Variable '{var_name}' not found (as point data) at time {time_a} and/or {time_b}.")

    diff = grid_b.point_data[var_name] - grid_a.point_data[var_name]
    diff_name = f"delta_{var_name}"

    grid = grid_b.copy()
    grid.point_data[diff_name] = diff

    surf = grid.extract_surface(algorithm=None)
    surf.points = surf.points * np.asarray(scale)

    max_abs = float(np.max(np.abs(diff)))
    logger.info("%s: max |change| between t=%s and t=%s is %.6g", var_name, time_a, time_b, max_abs)
    if max_abs == 0.0:
        max_abs = 1.0  # degenerate all-zero diff -- avoid a zero-width color range

    plotter = pv.Plotter()
    plotter.add_mesh(
        surf, scalars=diff_name, cmap=cmap, show_edges=show_edges, clim=(-max_abs, max_abs),
        scalar_bar_args={"title": f"Δ {_variable_label(var_name)}"},
    )
    plotter.add_text(f"Change in {_variable_label(var_name)}: t={time_a} -> t={time_b}", font_size=18, position="upper_edge")
    plotter.show_bounds(location="furthest", grid=True)
    _apply_camera_convention(plotter)
    plotter.show()


def _in_plane_axes(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str]]:
    """
    Two orthonormal in-plane basis vectors for the slice plane with the
    given normal, plus axis labels for the plot. For an axis-aligned normal
    (the common case -- e.g. (1, 0, 0)), the basis is the other two domain
    axes directly, so plot axes read as real x/y/z coordinates rather than
    an arbitrary rotated frame.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)

    if np.allclose(np.abs(n), [1, 0, 0]):
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), ("y", "z")
    if np.allclose(np.abs(n), [0, 1, 0]):
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), ("x", "z")
    if np.allclose(np.abs(n), [0, 0, 1]):
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), ("x", "y")

    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, ("u", "v")


def _slice_on_plane(
    sim: SimulationResults,
    var_name: str,
    time: float,
    origin: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice sim at `time` on the given plane and project onto (u, v) in-plane coordinates."""
    grid = build_grid_from_class(sim, time)
    if var_name not in grid.point_data:
        raise ValueError(f"Variable '{var_name}' not found (as point data) at time {time}.")
    sliced = grid.slice(origin=origin, normal=normal)
    if sliced.n_points == 0:
        raise RuntimeError("Slice produced no points -- check origin/normal against the model's extent.")
    pts = sliced.points
    u = pts @ u_axis
    v = pts @ v_axis
    values = np.asarray(sliced.point_data[var_name])
    return u, v, values


def _interpolate_to_grid(u, v, values, u_min, u_max, v_min, v_max, resolution):
    """Linearly interpolate scattered (u, v, values) points onto a regular resolution×resolution grid over [u_min, u_max] x [v_min, v_max]."""
    ug = np.linspace(u_min, u_max, resolution)
    vg = np.linspace(v_min, v_max, resolution)
    UG, VG = np.meshgrid(ug, vg)
    Z = griddata((u, v), values, (UG, VG), method="linear")
    return UG, VG, Z


def plot_cross_section_2D(
    sim: SimulationResults,
    var_name: str,
    time_a: Optional[float] = None,
    time_b: Optional[float] = None,
    origin: Tuple[float, float, float] = (0, 0, 0),
    normal: Tuple[float, float, float] = (1, 0, 0),
    cmap: str = "coolwarm",
    n_contours: int = 15,
    resolution: int = 200,
) -> None:
    """
    Slice var_name at two time steps on the same plane and plot them side by
    side with matplotlib -- filled contours plus contour lines, sharing one
    color scale so the two panels are directly, visually comparable (unlike
    two separate 3D PyVista plots/camera views). Defaults to the first and
    last saved time steps.

    Only meaningful for point data (T, p) -- the slice is interpolated onto
    a shared regular 2D grid via scipy.griddata, working for any mesh_type
    (structured, unstructured, implicit) since it doesn't assume a regular
    node ordering, just point coordinates + values.

    Note: both panels share one absolute color scale spanning [T_top,
    T_bottom]-type background range, so a real but modest change can still
    be subtle here -- see plot_cross_section_difference_2D for a version
    that isolates just the change itself.
    """
    times = sorted(sim.nodes_by_time.keys())
    if not times:
        raise ValueError("sim has no time steps.")
    if time_a is None:
        time_a = times[0]
    if time_b is None:
        time_b = times[-1]

    u_axis, v_axis, (u_label, v_label) = _in_plane_axes(normal)
    u_a, v_a, val_a = _slice_on_plane(sim, var_name, time_a, origin, normal, u_axis, v_axis)
    u_b, v_b, val_b = _slice_on_plane(sim, var_name, time_b, origin, normal, u_axis, v_axis)

    u_min, u_max = min(u_a.min(), u_b.min()), max(u_a.max(), u_b.max())
    v_min, v_max = min(v_a.min(), v_b.min()), max(v_a.max(), v_b.max())
    UG, VG, Za = _interpolate_to_grid(u_a, v_a, val_a, u_min, u_max, v_min, v_max, resolution)
    _, _, Zb = _interpolate_to_grid(u_b, v_b, val_b, u_min, u_max, v_min, v_max, resolution)

    vmin = float(np.nanmin([np.nanmin(Za), np.nanmin(Zb)]))
    vmax = float(np.nanmax([np.nanmax(Za), np.nanmax(Zb)]))
    levels = np.linspace(vmin, vmax, n_contours)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    last_cf = None
    for ax, Z, t in zip(axes, (Za, Zb), (time_a, time_b)):
        last_cf = ax.contourf(UG, VG, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        cs = ax.contour(UG, VG, Z, levels=levels, colors="k", linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
        ax.set_title(f"t={t}")
        ax.set_xlabel(u_label)
        ax.set_ylabel(v_label)
        ax.set_aspect("equal")

    fig.colorbar(last_cf, ax=axes, orientation="vertical", label=_variable_label(var_name), fraction=0.046, pad=0.04)
    fig.suptitle(f"{_variable_label(var_name)} cross-section (origin={origin}, normal={normal}): initial vs final")
    plt.show()


def plot_cross_section_difference_2D(
    sim: SimulationResults,
    var_name: str,
    time_a: Optional[float] = None,
    time_b: Optional[float] = None,
    origin: Tuple[float, float, float] = (0, 0, 0),
    normal: Tuple[float, float, float] = (1, 0, 0),
    cmap: str = "RdBu_r",
    n_contours: int = 15,
    resolution: int = 200,
) -> None:
    """
    Single-panel 2D cross-section of value(time_b) - value(time_a) (default:
    first/last saved step), diverging colormap symmetric around zero. Unlike
    plot_cross_section_2D's two side-by-side absolute snapshots -- which
    both still show the full [T_top, T_bottom]-type background gradient,
    making a real but geometrically-modest change subtle -- this strips the
    background out entirely and shows only what actually moved, so the
    change's own spatial pattern (e.g. tracking a fold/unconformity) is not
    competing visually with the dominant vertical gradient.
    """
    times = sorted(sim.nodes_by_time.keys())
    if not times:
        raise ValueError("sim has no time steps.")
    if time_a is None:
        time_a = times[0]
    if time_b is None:
        time_b = times[-1]

    u_axis, v_axis, (u_label, v_label) = _in_plane_axes(normal)
    u_a, v_a, val_a = _slice_on_plane(sim, var_name, time_a, origin, normal, u_axis, v_axis)
    u_b, v_b, val_b = _slice_on_plane(sim, var_name, time_b, origin, normal, u_axis, v_axis)

    u_min, u_max = min(u_a.min(), u_b.min()), max(u_a.max(), u_b.max())
    v_min, v_max = min(v_a.min(), v_b.min()), max(v_a.max(), v_b.max())
    UG, VG, Za = _interpolate_to_grid(u_a, v_a, val_a, u_min, u_max, v_min, v_max, resolution)
    _, _, Zb = _interpolate_to_grid(u_b, v_b, val_b, u_min, u_max, v_min, v_max, resolution)
    Z = Zb - Za

    max_abs = float(np.nanmax(np.abs(Z)))
    if max_abs == 0.0:
        max_abs = 1.0  # degenerate all-zero diff -- avoid a zero-width color range
    levels = np.linspace(-max_abs, max_abs, n_contours)

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(UG, VG, Z, levels=levels, cmap=cmap, vmin=-max_abs, vmax=max_abs)
    cs = ax.contour(UG, VG, Z, levels=levels, colors="k", linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    ax.set_xlabel(u_label)
    ax.set_ylabel(v_label)
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, orientation="vertical", label=f"Δ {_variable_label(var_name)}", fraction=0.046, pad=0.04)
    fig.suptitle(f"Δ {_variable_label(var_name)} cross-section (origin={origin}, normal={normal}): t={time_a} -> t={time_b}")
    plt.show()


# ----------------------------------------------------------------------
# Pre-flight material-assignment check (before running a solve)
# ----------------------------------------------------------------------

def plot_builder_materials(
    builder: HydrothermalProblemBuilder,
    style: str = "surface",
    show_edges: bool = True,
    show_plotter: bool = True,
) -> pv.Plotter:
    """
    Plot builder.mesh_results in 3D, one solid color per distinct material
    name from builder.compute_cell_materials() (lithologies, plus the fault
    zone if builder.fault_zone_properties is set), with a legend -- lets a
    mistaken material assignment (e.g. a fault zone landing in the wrong
    place, or fault_activity not restricting it as expected) be caught by
    eye before running a solve that can take a while, rather than only after.

    Reuses core.meshing_components.mesh_format.vtk.VTK_format.VTKInputs to
    build the combined PyVista grid, rather than re-implementing the meshio
    -> PyVista conversion -- that class already iterates mesh_results.elements
    in the same block/cell order compute_cell_materials() does, so the two
    arrays line up without needing a separate ordering assumption.

    Legend matches structural_modeling_visualization.plot_structural_model_3D's
    legend for the *same* structural model (builder.geomodel_result is that
    same model, so this is directly reconstructable), rather than an
    independent styling:
    - Colors come from each StructuralElement's own .color (never None
      after construction) instead of an independent fallback palette, so
      the same lithology reads as the same color across the structural
      model plot, the mesh plot, and this one. Basement isn't part of any
      structural_groups entry; given the same grey plot_mesh_3d already
      uses for it ("#808080").
    - Legend entries are grouped under a bold-ish header row per
      structural_groups entry (its real group name, e.g. "Strat_Series1"),
      each followed by "• element_name" bullets in that element's color,
      each in turn followed by a one-line summary of that material's
      RockUnitProperties (porosity/permeability/thermal conductivity/heat
      capacity, the actual values this solve uses) -- not a flat list of
      bare material names, and not something you'd otherwise see without
      cross-referencing the builder's rock_properties dict by hand.
    - Top-to-bottom order matches plot_structural_model_3D's: a "Fault
      Zone" section first (if active), then structural_groups in their
      stored order (youngest group at the top, oldest at the bottom -- the
      first entry in PyVista's `labels` list renders at the top), then
      "Basement" last (it sits below/outside every defined group, the
      oldest thing in the model).

    show_edges defaults to True (useful for spotting individual cell
    assignment during interactive debugging), but turns visually noisy on
    a fine mesh -- every cell's edge starts blending into the next,
    especially in a static screenshot rather than an interactive,
    zoomable view. Set False for anything above roughly 50k cells, or for
    a clean, publication/gallery-style render.
    """
    materials = builder.compute_cell_materials()
    present = set(np.unique(materials).tolist())
    grid = VTKInputs(builder.mesh_results.nodes, builder.mesh_results.elements).create_mesh()

    frame = builder.geomodel_result.structural_frame
    fault_name = builder.fault_zone_properties.name if builder.fault_zone_properties is not None else None

    color_by_name: Dict[str, str] = {"basement": "#808080"}
    for group in frame.structural_groups:
        for elem in group.structural_elements:
            color_by_name[elem.name] = elem.color
    if fault_name is not None:
        color_by_name[fault_name] = _FAULT_ZONE_COLOR

    properties_by_name: Dict[str, RockUnitProperties] = dict(builder.rock_properties)
    if fault_name is not None:
        properties_by_name[fault_name] = builder.fault_zone_properties

    def legend_entry(name: str) -> Tuple[str, str, str]:
        """(material_name, legend_label, color) for one material -- label includes its rock properties."""
        label = f"• {name}  ({_format_rock_properties(properties_by_name[name])})"
        return name, label, color_by_name[name]

    # (header, [(material_name, legend_label, color), ...])
    sections: List[Tuple[str, List[Tuple[str, str, str]]]] = []
    if fault_name is not None and fault_name in present:
        sections.append(("Fault Zone", [legend_entry(fault_name)]))
    for group in frame.structural_groups:  # stored youngest -> oldest, same as structural_groups itself
        entries = [legend_entry(elem.name) for elem in group.structural_elements if elem.name in present]
        if entries:
            sections.append((group.name, entries))
    if "basement" in present:
        sections.append(("Basement", [legend_entry("basement")]))

    plotter = pv.Plotter(off_screen=not show_plotter)
    for _, entries in sections:
        for name, _label, color in entries:
            cell_ids = np.where(materials == name)[0]
            sub = grid.extract_cells(cell_ids)
            plotter.add_mesh(sub, show_edges=show_edges, style=style, color=color)

    flat_legend: List[Tuple[str, str]] = []
    for header, entries in sections:
        flat_legend.append((header, "black"))
        flat_legend.extend((label, color) for _name, label, color in entries)

    if flat_legend:
        plotter.add_legend(
            labels=flat_legend,
            size=(0.32, 0.03 * len(flat_legend)),
            loc="lower right",
            face="rectangle",
        )

    plotter.show_bounds(location="furthest", grid=True)
    _apply_camera_convention(plotter)
    if show_plotter:
        plotter.show()
    return plotter
