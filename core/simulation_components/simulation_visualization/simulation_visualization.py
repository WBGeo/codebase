"""
Plotting of SimulationResults (3D snapshots, cross-sections, before/after
diffs, line and time-series samples) for both simulators.

The pre-solve material preview lives in material_preview.py and is
re-exported here as plot_builder_materials.

PyVista styling (camera angle, show_bounds, legend face style) follows the
conventions established in
core.structural_modeling_components.structural_modeling_visualization
(used consistently across all of its 3D plots, and by
core.meshing_components.meshing_visualization.plot_mesh_3d) for a
consistent look across the codebase's 3D plots, rather than this package's
own ad-hoc defaults.
"""
import inspect
import logging
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata

from core.object_components import SimulationResults
from core.simulation_components.simulation_visualization.material_preview import (  # noqa: F401  (re-export)
    plot_builder_materials,
)

logger = logging.getLogger(__name__)

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


def _time_step_index(sim: SimulationResults, time: float) -> int:
    """Zero-based saved-step index for a simulation time key."""
    return sorted(sim.nodes_by_time).index(time)


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


#: Whether the installed pyvista's DataSetFilters.extract_surface() accepts
#: an `algorithm` kwarg -- checked via introspection, not a hardcoded
#: version cutoff, since the actual break is "does this pyvista have the
#: parameter". extract_surface()'s own default algorithm
#: ("dataset_surface", a vtkDataSetSurfaceFilter) produces an empty mesh (0
#: points) for this pipeline's SfePy-exported grids -- confirmed live on
#: *both* the local dev venv's pyvista (0.48.4) *and* this pipeline's own
#: pinned requirements.txt version, pyvista==0.44.1 (what's actually
#: installed in the Docker Workbench, where this was originally caught).
#: algorithm=None requests the correct vtkGeometryFilter-based behavior on
#: pyvista new enough to have the kwarg; pyvista==0.44.1 predates the
#: kwarg entirely, so _extract_surface_compat falls back to
#: grid.extract_geometry() there instead -- a plain vtkGeometryFilter
#: wrapper with the same output, not yet deprecated in 0.44.1 (it is in
#: 0.48.4, in favor of algorithm=None, which is why this fallback isn't
#: just used unconditionally).
_EXTRACT_SURFACE_SUPPORTS_ALGORITHM = "algorithm" in inspect.signature(pv.DataSet.extract_surface).parameters


def _extract_surface_compat(grid: pv.DataSet) -> pv.PolyData:
    """grid.extract_surface() via vtkGeometryFilter, not extract_surface()'s own
    empty-mesh-prone default -- see _EXTRACT_SURFACE_SUPPORTS_ALGORITHM."""
    if _EXTRACT_SURFACE_SUPPORTS_ALGORITHM:
        return grid.extract_surface(algorithm=None)
    return grid.extract_geometry()


# ----------------------------------------------------------------------
# Grid construction
# ----------------------------------------------------------------------

def _build_grid_geometry(sim: SimulationResults, time: float) -> pv.UnstructuredGrid:
    """
    Build a PyVista grid for one saved time step's connectivity only (nodes/
    cells/celltypes, no data attached) -- split out from build_grid_from_class
    so a fixed mesh's geometry (identical across every saved timestep -- this
    pipeline never remeshes/deforms mid-solve) can be built once and reused
    across timesteps via _attach_time_data, instead of every plot rebuilding
    the full VTK grid (by far the expensive part of build_grid_from_class,
    not attaching data arrays) once per timestep it needs.

    Raises:
        ValueError: `time` isn't a key in `sim.nodes_by_time`.
    """
    if time not in sim.nodes_by_time:
        _validate_saved_time(sim, time)

    nodes = sim.nodes_by_time[time]
    cells = sim.cells_by_time.get(time, None)
    celltypes = sim.celltypes_by_time.get(time, None)

    if cells is None or len(cells) == 0:
        return pv.PolyData(nodes)
    return pv.UnstructuredGrid(cells, celltypes, nodes)


def _attach_time_data(
    mesh: pv.DataSet, sim: SimulationResults, time: float, var_names: Optional[Sequence[str]] = None,
) -> None:
    """Attach `time`'s node/cell data onto an existing mesh in place, replacing whatever was there before.

    var_names: if given, only these variables are attached instead of every
    variable at this time step -- lets a caller that only plots one or two
    variables (e.g. plot_variable_at_a_time) skip allocating/copying the
    rest onto the mesh. None (default) attaches everything, as before --
    needed by callers like _export_simulation_results_to_vtk that genuinely
    need the full result.
    """
    mesh.point_data.clear()
    mesh.cell_data.clear()
    node_data = sim.node_data_by_time.get(time, {})
    cell_data = sim.cell_data_by_time.get(time, {})
    names = var_names if var_names is not None else set(node_data) | set(cell_data)
    for name in names:
        if name in node_data:
            mesh.point_data[name] = node_data[name]
        if name in cell_data:
            mesh.cell_data[name] = cell_data[name]


def _same_geometry(sim: SimulationResults, time_a: float, time_b: float) -> bool:
    """
    Whether time_a/time_b share identical mesh geometry (nodes/cells/
    celltypes) -- true for every solve in this pipeline (a fixed FEM mesh, no
    remeshing/deformation between saved steps), but checked explicitly rather
    than assumed, so a two-timestep plot never silently reuses mismatched
    geometry if that ever stops being true.
    """
    nodes_a, nodes_b = sim.nodes_by_time.get(time_a), sim.nodes_by_time.get(time_b)
    if nodes_a is None or nodes_b is None or nodes_a.shape != nodes_b.shape or not np.array_equal(nodes_a, nodes_b):
        return False

    cells_a, cells_b = sim.cells_by_time.get(time_a), sim.cells_by_time.get(time_b)
    if (cells_a is None) != (cells_b is None):
        return False
    if cells_a is not None and (len(cells_a) != len(cells_b) or not np.array_equal(cells_a, cells_b)):
        return False

    celltypes_a, celltypes_b = sim.celltypes_by_time.get(time_a), sim.celltypes_by_time.get(time_b)
    if (celltypes_a is None) != (celltypes_b is None):
        return False
    if celltypes_a is not None and not np.array_equal(celltypes_a, celltypes_b):
        return False

    return True


def build_grid_from_class(
    sim: SimulationResults, time: float, var_names: Optional[Sequence[str]] = None,
) -> pv.UnstructuredGrid:
    """
    Build a PyVista grid for one saved time step of a SimulationResults --
    an UnstructuredGrid if `sim` has real cell connectivity for `time`, else a
    bare PolyData point cloud (nodes only). Attaches every variable in
    `sim.node_data_by_time[time]` as point data, unless `var_names` narrows
    that down (see _attach_time_data) -- most plotting functions only ever
    plot one variable and don't need the rest attached. Shared by every
    plotting function in this module rather than each reimplementing the
    conversion.

    Raises:
        ValueError: `time` isn't a key in `sim.nodes_by_time`.
    """
    mesh = _build_grid_geometry(sim, time)
    _attach_time_data(mesh, sim, time, var_names)
    return mesh


# VTK bookkeeping arrays are not physical simulation results.  Compare names
# case-insensitively so files produced by different exporters behave alike.
_METADATA_FIELDS = {
    "globalnodeid", "pedigreenodeid", "globalelementid",
    "pedigreeelementid", "globalcellid", "pedigreecellid",
    "vtkoriginalpointids", "vtkoriginalcellids", "vtkghosttype",
}


def _available_variables(sim: SimulationResults, time: float) -> List[str]:
    """Return numeric scalar result fields, preserving their original order.

    Works with arbitrary field names (e.g. T, temperature, temp, te) and
    ignores VTK identifier arrays. Vector/tensor fields can still be plotted
    explicitly with ``var_name`` where supported by the plotting function.
    """
    node_data = sim.node_data_by_time.get(time, {})
    cell_data = sim.cell_data_by_time.get(time, {})
    variables = []
    for fields in (node_data, cell_data):
        for name, values in fields.items():
            if name.lower() in _METADATA_FIELDS or name in variables:
                continue
            array = np.asarray(values)
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
                continue
            variables.append(name)
    return variables


def _default_var_name(sim: SimulationResults, time: float) -> str:
    """Choose a plottable scalar without requiring model-specific field names."""
    variables = _available_variables(sim, time)
    if not variables:
        raise ValueError(f"No numeric scalar simulation fields found at time {time}.")
    # Preserve the existing SfePy default, but accept any other model's names.
    return "T" if "T" in variables else variables[0]


def _validate_saved_time(sim: SimulationResults, time: float, parameter: str = "time") -> None:
    """Raise an informative error with all saved step indices and actual time values."""
    times = sorted(sim.nodes_by_time)
    if not times:
        raise ValueError("sim has no time steps.")
    if time not in sim.nodes_by_time:
        available = ", ".join(f"step {i}: {t:g}" for i, t in enumerate(times))
        raise ValueError(
            f"Invalid {parameter}={time!r}. Available saved time steps "
            f"(index: time value): {available}. "
            "Pass the actual time value, not the zero-based step index."
        )


def _validate_cross_section_inputs(sim: SimulationResults, var_name: str,
                                   time_a: float, time_b: float) -> None:
    """Validate both times and point-data field before building/slicing a mesh."""
    _validate_saved_time(sim, time_a, "time_a")
    _validate_saved_time(sim, time_b, "time_b")
    for time in dict.fromkeys((time_a, time_b)):
        if var_name not in sim.node_data_by_time.get(time, {}):
            raise _variable_not_found_error(sim, time, var_name, point_only=True)


def _variable_not_found_error(sim: SimulationResults, time: float, var_name: str, *, point_only: bool = False) -> ValueError:
    """Explain an invalid variable name and list fields actually present at this timestep."""
    node_names = list(sim.node_data_by_time.get(time, {}))
    cell_names = list(sim.cell_data_by_time.get(time, {}))
    available = node_names if point_only else list(dict.fromkeys(node_names + cell_names))
    kind = "point-data variables" if point_only else "variables"
    listed = ", ".join(repr(name) for name in available) if available else "(none)"
    return ValueError(
        f"Variable {var_name!r} not found at time {time}. "
        f"Available {kind}: {listed}. "
        "Variable names are case-sensitive; use an exact name from the list."
    )


# ----------------------------------------------------------------------
# Single-snapshot plots
# ----------------------------------------------------------------------

def plot_variable_at_a_time(
    sim: SimulationResults,
    var_name: Optional[str] = None,
    time: Optional[float] = None,
    cmap: str = "viridis",
    show_edges: bool = False,
    edge_color: str = "grey",
    edge_line_width: float = 0.5,
    show_contours: bool = False,
    n_contours: int = 10,
    contour_color: str = "black",
    scale: Tuple[float, float, float] = (1, 1, 1),
    off_screen: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    anti_aliasing: bool = True,
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
    isosurface-of-a-3D-volume use case) -- a 3D analogue of the contour
    lines plot_cross_section_2D already draws in its 2D matplotlib slices.

    time defaults to the final saved time step; var_name defaults to "T"
    if present, else whatever field the sim actually has (see
    _default_var_name) -- a CustomSfepyBuilder result need not solve
    hydrothermal physics at all.

    off_screen/window_size/anti_aliasing tune the render's own resource
    footprint rather than what's shown -- this is the only 3D plot in the
    codebase whose surface is colored by a continuous scalar field with a
    colorbar (every other 3D plot here uses flat, solid per-block/material
    colors), and that combination is disproportionately expensive under
    software/CPU rendering (no GPU, no real display -- e.g. a headless
    Workbench container) rather than hardware-accelerated OpenGL. Defaults
    match this function's previous, unparameterized behavior (on-screen,
    PyVista's own default window size, antialiasing on) for existing
    interactive/script callers; the Workbench inspector passes tighter
    values instead (see inspect_simulation_result_plot_variable_at_a_time).
    """
    if not sim.nodes_by_time:
        raise ValueError("sim has no time steps.")
    if time is None:
        time = max(sim.nodes_by_time.keys())
    _validate_saved_time(sim, time)
    if var_name is None:
        var_name = _default_var_name(sim, time)

    grid = build_grid_from_class(sim, time, var_names=[var_name])

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise _variable_not_found_error(sim, time, var_name)

    surf = _extract_surface_compat(grid)
    surf.points *= scale

    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    if not anti_aliasing:
        plotter.disable_anti_aliasing()
    plotter.add_mesh(
        surf, scalars=var_name, cmap=cmap,
        show_edges=show_edges, edge_color=edge_color, line_width=edge_line_width,
        # add_mesh(scalars=...) already adds its own scalar bar -- setting
        # its title here (rather than a separate add_scalar_bar() call)
        # avoids a second, redundant bar with a mismatched colormap
        # (add_scalar_bar() called separately doesn't pick up this mesh's
        # own cmap).
        scalar_bar_args={"title": _variable_label(var_name)},
    )
    if show_contours:
        if var_name not in surf.point_data:
            raise ValueError(f"show_contours=True needs '{var_name}' as point data, only found as cell data.")
        contours = surf.contour(isosurfaces=n_contours, scalars=var_name)
        # show_scalar_bar=False: contour() leaves var_name as the returned
        # mesh's active scalars, which would otherwise add a second,
        # redundant scalar bar even though color= is a fixed solid color
        # here, not scalars=.
        plotter.add_mesh(contours, color=contour_color, line_width=2, show_scalar_bar=False)
    plotter.add_text(f"{_variable_label(var_name)} at time step {_time_step_index(sim, time)}", font_size=20, position="upper_edge")
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

    # Built once and reused for both timesteps when they share geometry
    # (the common case -- see _same_geometry) instead of building two full
    # VTK grids just to read off one variable's values at each.
    grid = _build_grid_geometry(sim, time_a)
    _attach_time_data(grid, sim, time_a, var_names=[var_name])
    if var_name not in grid.point_data:
        raise _variable_not_found_error(sim, time_a, var_name, point_only=True)
    val_a = np.asarray(grid.point_data[var_name]).copy()

    if not _same_geometry(sim, time_a, time_b):
        grid = _build_grid_geometry(sim, time_b)
    _attach_time_data(grid, sim, time_b, var_names=[var_name])
    if var_name not in grid.point_data:
        raise _variable_not_found_error(sim, time_b, var_name, point_only=True)
    val_b = grid.point_data[var_name]

    diff = val_b - val_a
    diff_name = f"delta_{var_name}"
    grid.point_data[diff_name] = diff

    surf = _extract_surface_compat(grid)
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
    plotter.add_text(f"Change in {_variable_label(var_name)}: step {_time_step_index(sim, time_a)} -> step {_time_step_index(sim, time_b)}", font_size=18, position="upper_edge")
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


def _slice_on_plane_from_grid(
    mesh: pv.DataSet,
    var_name: str,
    origin: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice an already-built (geometry + data attached) grid on the given plane
    and project onto (u, v) in-plane coordinates -- split from build_grid_from_class
    so callers needing the same mesh at two timesteps (plot_cross_section_2D,
    plot_cross_section_difference_2D) can build the (expensive) VTK grid once
    and just swap data between slices, rather than rebuilding it per timestep.
    """
    if var_name not in mesh.point_data:
        raise ValueError(f"Variable '{var_name}' not found (as point data).")
    sliced = mesh.slice(origin=origin, normal=normal)
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
    var_name: Optional[str] = None,
    time_a: Optional[float] = None,
    time_b: Optional[float] = None,
    origin: Tuple[float, float, float] = (0, 0, 0),
    normal: Tuple[float, float, float] = (0, 1, 0),
    cmap: str = "coolwarm",
    n_contours: int = 15,
    resolution: int = 200,
) -> None:
    """
    Slice var_name at two time steps on the same plane and plot them side by
    side with matplotlib -- filled contours plus contour lines, sharing one
    color scale so the two panels are directly, visually comparable (unlike
    two separate 3D PyVista plots/camera views). time_a/time_b default to
    the first and last saved time steps; var_name defaults to "T" if
    present, else whatever field the sim actually has (see
    _default_var_name) -- a CustomSfepyBuilder result need not solve
    hydrothermal physics at all.

    normal defaults to (0, 1, 0) -- a slice perpendicular to Y, spanning
    X and Z -- matching how this pipeline's own example models are laid
    out (model2/model5's own plot_cross_section_2D calls use this same
    normal) so the default view shows a cross section along X rather than
    along Y.

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
    if var_name is None:
        var_name = _default_var_name(sim, time_a)

    _validate_cross_section_inputs(sim, var_name, time_a, time_b)

    u_axis, v_axis, (u_label, v_label) = _in_plane_axes(normal)

    # Built once and reused for both timesteps when they share geometry
    # (the common case -- see _same_geometry) instead of building two full
    # VTK grids just to slice each at the same plane.
    mesh = _build_grid_geometry(sim, time_a)
    _attach_time_data(mesh, sim, time_a, var_names=[var_name])
    u_a, v_a, val_a = _slice_on_plane_from_grid(mesh, var_name, origin, normal, u_axis, v_axis)

    if not _same_geometry(sim, time_a, time_b):
        mesh = _build_grid_geometry(sim, time_b)
    _attach_time_data(mesh, sim, time_b, var_names=[var_name])
    u_b, v_b, val_b = _slice_on_plane_from_grid(mesh, var_name, origin, normal, u_axis, v_axis)

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
        ax.set_title(f"Time step {_time_step_index(sim, t)}")
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

    _validate_cross_section_inputs(sim, var_name, time_a, time_b)

    u_axis, v_axis, (u_label, v_label) = _in_plane_axes(normal)

    # Built once and reused for both timesteps when they share geometry
    # (the common case -- see _same_geometry) instead of building two full
    # VTK grids just to slice each at the same plane.
    mesh = _build_grid_geometry(sim, time_a)
    _attach_time_data(mesh, sim, time_a, var_names=[var_name])
    u_a, v_a, val_a = _slice_on_plane_from_grid(mesh, var_name, origin, normal, u_axis, v_axis)

    if not _same_geometry(sim, time_a, time_b):
        mesh = _build_grid_geometry(sim, time_b)
    _attach_time_data(mesh, sim, time_b, var_names=[var_name])
    u_b, v_b, val_b = _slice_on_plane_from_grid(mesh, var_name, origin, normal, u_axis, v_axis)

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
    fig.suptitle(f"Δ {_variable_label(var_name)} cross-section (origin={origin}, normal={normal}): step {_time_step_index(sim, time_a)} -> step {_time_step_index(sim, time_b)}")
    plt.show()


# ----------------------------------------------------------------------
# Point / line sampling -- ported from the previous implementation's
# results_visualisation.py (plot_variable_along_line, print_variable_at_point,
# plot_variable_time_series), which were dropped in the rewrite rather than
# reimplemented; restored here on request since they're useful post-processing
# tools not covered by the snapshot/cross-section plots above.
# ----------------------------------------------------------------------

def plot_variable_along_line(
    sim: SimulationResults,
    var_name: str,
    time: float,
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    n_samples: int = 200,
) -> None:
    """
    Sample var_name along the straight line from p0 to p1 at one time step
    and plot it against distance along the line (via PyVista's own
    pv.Line(...).sample(grid), which linearly interpolates from the mesh).
    """
    _validate_saved_time(sim, time, "time")
    if var_name is None:
        var_name = _default_var_name(sim, time)
    if (var_name not in sim.node_data_by_time.get(time, {})
            and var_name not in sim.cell_data_by_time.get(time, {})):
        raise _variable_not_found_error(sim, time, var_name)
    grid = build_grid_from_class(sim, time, var_names=[var_name])

    line = pv.Line(p0, p1, resolution=n_samples)
    sampled = line.sample(grid)
    if sampled.n_points == 0:
        raise RuntimeError("Line sampling produced no points -- check p0/p1 against the model's extent.")

    values = sampled[var_name]
    distances = sampled["Distance"]

    plt.figure(figsize=(7, 4))
    plt.plot(distances, values, "-k")
    plt.xlabel("Distance along line")
    plt.ylabel(_variable_label(var_name))
    plt.title(f"{_variable_label(var_name)} along line at time step {_time_step_index(sim, time)}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _sample_point_on_mesh(
    mesh: pv.DataSet,
    var_name: str,
    point: Tuple[float, float, float],
) -> Optional[float]:
    """
    Sample var_name at a single point on an already-built (geometry + data
    attached) mesh. Returns None (with a warning logged) if `point` falls
    outside the mesh's bounds. Split from print_variable_at_point so
    plot_variable_time_series's loop over every saved timestep can reuse one
    mesh's geometry (see _build_grid_geometry/_same_geometry) instead of
    rebuilding the full VTK grid once per timestep.
    """
    if var_name not in mesh.point_data and var_name not in mesh.cell_data:
        raise ValueError(f"Variable '{var_name}' not found.")

    pt = pv.PolyData(np.asarray([point], dtype=float))
    sampled = pt.sample(mesh)
    if sampled.n_points == 0 or not np.any(sampled["vtkValidPointMask"]):
        logger.warning("Point %s is outside mesh bounds %s", point, mesh.bounds)
        return None

    return float(sampled[var_name][0])


def print_variable_at_point(
    sim: SimulationResults,
    var_name: str,
    time: float,
    point: Tuple[float, float, float],
) -> Optional[float]:
    """
    Sample var_name at a single point at one time step (via PyVista point
    sampling) and log it. Returns the sampled value, or None (with a
    warning logged) if `point` falls outside the mesh's bounds.
    """
    grid = build_grid_from_class(sim, time, var_names=[var_name])
    value = _sample_point_on_mesh(grid, var_name, point)
    if value is not None:
        logger.info("%s at point %s at time %s: %s", var_name, point, time, value)
    return value


def plot_variable_time_series(
    sim: SimulationResults,
    var_name: str,
    point: Tuple[float, float, float],
    decimals: int = 3,
    time_step_start: Optional[int] = None,
    time_step_end: Optional[int] = None,
) -> None:
    """
    Sample var_name at a single point across every saved time step and plot
    the time series. A point outside the mesh's bounds at a given step is
    plotted as a gap (NaN) rather than raising, so one bad step doesn't
    prevent seeing the rest of the series.
    """
    times = sorted(sim.nodes_by_time.keys())
    if not times:
        raise ValueError("sim has no time steps.")
    start = 0 if time_step_start is None else time_step_start
    end = len(times) - 1 if time_step_end is None else time_step_end
    available_steps = ", ".join(f"step {i}: {t:g}" for i, t in enumerate(times))
    for label, index in (("time_step_start", start), ("time_step_end", end)):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(times):
            raise ValueError(
                f"Invalid {label}={index!r}. Available saved time steps "
                f"(index: time value): {available_steps}. "
                "For time series, pass a zero-based step index, not the physical time value."
            )
    if start > end:
        raise ValueError(
            f"time_step_start={start} must be <= time_step_end={end}. "
            f"Available saved time steps (index: time value): {available_steps}."
        )
    selected_times = times[start:end + 1]
    if var_name is None:
        var_name = _default_var_name(sim, selected_times[0])
    for t in selected_times:
        if (var_name not in sim.node_data_by_time.get(t, {})
                and var_name not in sim.cell_data_by_time.get(t, {})):
            raise _variable_not_found_error(sim, t, var_name)

    # Geometry is built once (for times[0]) and reused across every
    # subsequent timestep when it's unchanged (the common case -- see
    # _same_geometry), instead of rebuilding the full VTK grid once per
    # saved step just to sample one point.
    mesh = _build_grid_geometry(sim, times[0])
    values = []
    for t in selected_times:
        if not _same_geometry(sim, times[0], t):
            mesh = _build_grid_geometry(sim, t)
        _attach_time_data(mesh, sim, t, var_names=[var_name])
        value = _sample_point_on_mesh(mesh, var_name, point)
        if value is not None:
            logger.info("%s at point %s at time %s: %s", var_name, point, t, value)
        values.append(value if value is not None else np.nan)
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(start, end + 1), values, "-o", color="tab:red")
    ax.set_xlabel("Time step")
    ax.set_ylabel(_variable_label(var_name))
    ax.set_title(f"{_variable_label(var_name)} at point {point} over time steps")
    ax.grid(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    plt.tight_layout()
    plt.show()
