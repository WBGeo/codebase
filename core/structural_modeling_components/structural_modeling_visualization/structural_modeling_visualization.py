from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex
import warnings

from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultFrame,
    StructuralFrame,
    GeoMeshType,
    FaultMeshType,
)

if TYPE_CHECKING:
    from core.object_components import InputData_StructuralElements, InputData_FaultElements


def _faces_to_vtk(faces_arr: np.ndarray) -> np.ndarray:
    """Convert (N,3) int triangle array to VTK flat face array."""
    faces_arr = np.asarray(faces_arr)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("faces must be (N, 3) triangle indices")
    return np.hstack([np.full((faces_arr.shape[0], 1), 3, dtype=np.int64), faces_arr.astype(np.int64)]).ravel()


def _make_formation_color_map(mapping_object: dict) -> dict[str, str]:
    """Return {formation_name: hex_color} using the same group/element shading as build_structural_frame."""
    group_cmap = plt.get_cmap("Accent")
    color_map: dict[str, str] = {}
    groups = list(mapping_object.items())
    n_groups = len(groups)
    for g_idx, (_, elements) in enumerate(groups):
        base_rgb = group_cmap(g_idx / max(n_groups, 1))[:3]
        base_hls = colorsys.rgb_to_hls(*base_rgb)
        n_el = len(elements)
        for i, el_name in enumerate(elements):
            lightness = 0.35 + 0.5 * (i / max(n_el - 1, 1))
            color_map[el_name] = to_hex(colorsys.hls_to_rgb(base_hls[0], lightness, base_hls[2]))
    return color_map


def plot_input_data_3D(
        input_data: InputData_StructuralElements,
        *,
        show_orientations: bool = True,
        arrow_scale: Optional[float] = None,
        point_size: int = 8,
        notebook: bool = False,
        show: bool = True,
) -> pv.Plotter:
    """
    3D PyVista plot of raw structural input data (surface points + orientations),
    colored by formation.  No StructuralFrame required.
    """
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(notebook=notebook)

    sp = input_data.surface_points
    ori = input_data.orientations
    mapping = input_data.mapping_object  # {group_name: (elem1, elem2, ...)}

    color_map = _make_formation_color_map(mapping)

    if arrow_scale is None and show_orientations and ori is not None and not ori.empty:
        pts = sp[["X", "Y", "Z"]].values
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        arrow_scale = max(diag * 0.05, 1.0)

    legend_entries: list[tuple[str, str]] = []
    for group_name, elements in mapping.items():
        legend_entries.append((group_name, "black"))
        for el_name in elements:
            color = color_map.get(el_name, "grey")
            legend_entries.append((f"• {el_name}", color))

            df_pts = sp[sp["formation"] == el_name]
            if not df_pts.empty:
                cloud = pv.PolyData(df_pts[["X", "Y", "Z"]].values.astype(np.float64))
                plotter.add_points(cloud, color=color, point_size=point_size, render_points_as_spheres=True)

            if show_orientations and ori is not None and not ori.empty:
                df_ori = ori[ori["formation"] == el_name]
                for _, row in df_ori.iterrows():
                    plotter.add_mesh(
                        pv.Arrow(
                            start=row[["X", "Y", "Z"]].values.astype(float),
                            direction=row[["G_x", "G_y", "G_z"]].values.astype(float),
                            scale=arrow_scale,
                        ),
                        color=color,
                    )

    pts = sp[["X", "Y", "Z"]].values
    ranges = pts.max(axis=0) - pts.min(axis=0)
    margin = ranges * 0.05
    bounds = (
        pts[:, 0].min() - margin[0], pts[:, 0].max() + margin[0],
        pts[:, 1].min() - margin[1], pts[:, 1].max() + margin[1],
        pts[:, 2].min() - margin[2], pts[:, 2].max() + margin[2],
    )
    plotter.show_bounds(bounds=bounds, location="furthest", grid=True)

    plotter.add_legend(labels=legend_entries, size=(0.22, 0.25), loc="lower right", face="rectangle")
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show:
        plotter.show()
    return plotter


def plot_fault_input_data_3D(
        input_data: InputData_FaultElements,
        *,
        show_orientations: bool = True,
        arrow_scale: Optional[float] = None,
        point_size: int = 8,
        notebook: bool = False,
        show: bool = True,
) -> pv.Plotter:
    """
    3D PyVista plot of raw fault input data (surface points + orientations),
    colored by fault name.  No FaultFrame required.
    """
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(notebook=notebook)

    sp = input_data.fault_surface_points
    ori = input_data.fault_orientations
    fault_names = input_data.fault_names

    fault_cmap = plt.get_cmap("tab10")
    color_map = {
        name: to_hex(fault_cmap(i / max(len(fault_names), 1))[:3])
        for i, name in enumerate(fault_names)
    }

    if arrow_scale is None and show_orientations and ori is not None and not ori.empty:
        pts = sp[["X", "Y", "Z"]].values
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        arrow_scale = max(diag * 0.05, 1.0)

    legend_entries: list[tuple[str, str]] = []
    for fault_name in fault_names:
        color = color_map.get(fault_name, "grey")
        legend_entries.append((fault_name, color))

        df_pts = sp[sp["formation"] == fault_name]
        if not df_pts.empty:
            cloud = pv.PolyData(df_pts[["X", "Y", "Z"]].values.astype(np.float64))
            plotter.add_points(cloud, color=color, point_size=point_size, render_points_as_spheres=True)

        if show_orientations and ori is not None and not ori.empty:
            df_ori = ori[ori["formation"] == fault_name]
            for _, row in df_ori.iterrows():
                plotter.add_mesh(
                    pv.Arrow(
                        start=row[["X", "Y", "Z"]].values.astype(float),
                        direction=row[["G_x", "G_y", "G_z"]].values.astype(float),
                        scale=arrow_scale,
                    ),
                    color=color,
                )

    pts = sp[["X", "Y", "Z"]].values
    ranges = pts.max(axis=0) - pts.min(axis=0)
    margin = ranges * 0.05
    bounds = (
        pts[:, 0].min() - margin[0], pts[:, 0].max() + margin[0],
        pts[:, 1].min() - margin[1], pts[:, 1].max() + margin[1],
        pts[:, 2].min() - margin[2], pts[:, 2].max() + margin[2],
    )
    plotter.show_bounds(bounds=bounds, location="furthest", grid=True)

    plotter.add_legend(labels=legend_entries, size=(0.20, 0.18), loc="lower right", face="rectangle")
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show:
        plotter.show()
    return plotter


def plot_structural_model_3D(
        frame: StructuralFrame,
        *,
        mesh_type: GeoMeshType = GeoMeshType.MASKED,
        fault_mesh_type: Optional[FaultMeshType] = None,
        show_surface_meshes: bool = True,
        show_points: bool = True,
        show_orientations: bool = True,
        fault_opacity: float = 0.35,
        notebook: bool = False,
        off_screen: bool = False,
        show: bool = True,
) -> pv.Plotter:
    """
    3D PyVista plot of a structural model with per-element meshes, input points, and faults.

    Parameters
    ----------
    frame : StructuralFrame
        Structural frame to visualize.
    mesh_type : GeoMeshType, default GeoMeshType.MASKED
        Mesh type for geological elements.
    fault_mesh_type : FaultMeshType, optional
        Mesh type for fault surfaces. Defaults to ``mesh_type`` if not set.
    show_surface_meshes : bool, default True
        Whether to render surface meshes for each element.
    show_points : bool, default True
        Whether to render input surface points.
    show_orientations : bool, default True
        Whether to render orientation arrows.
    fault_opacity : float, default 0.35
        Opacity of fault surface meshes.
    notebook : bool, default False
        Use off-screen/notebook mode for PyVista.
    off_screen : bool, default False
        Render off-screen (no window). Required for screenshot() without calling show() first.
    show : bool, default True
        Call plotter.show() at the end. Set False to add further actors before displaying.

    Returns
    -------
    pv.Plotter
        The PyVista plotter instance (useful when show=False).
    """
    fault_mesh_type = fault_mesh_type if fault_mesh_type is not None else mesh_type

    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(notebook=notebook, off_screen=off_screen)

    # ---------- plot structural meshes, points, orientations ----------
    legend_entries = []  # [(group_name, [(• elem, color), ...]), ...] in youngest->oldest order

    for group in frame.structural_groups:  # order preserved: youngest->oldest
        group_entries = []
        for elem in group.structural_elements:
            # legend entry
            group_entries.append((f"• {elem.name}", elem.color))

            # meshes (per domain)
            if show_surface_meshes:
                if mesh_type in elem.vertices and mesh_type in elem.edges:
                    plotter.add_mesh(
                        pv.PolyData(elem.vertices[mesh_type],
                                    np.insert(elem.edges[mesh_type], 0, 3, axis=1).ravel()),
                        color=elem.color, opacity=1.0, label=f"{group.name} | {elem.name}")

            # surface points
            if show_points and hasattr(frame, "get_surface_points_for_element"):
                df_points = frame.get_surface_points_for_element(elem.name)
                if df_points is not None and not df_points.empty:
                    cloud = pv.PolyData(df_points[["X", "Y", "Z"]].values.astype(np.float32))
                    plotter.add_points(cloud, color=elem.color, point_size=8, render_points_as_spheres=True)

            # orientations
            if show_orientations and frame.orientations is not None and hasattr(frame,
                                                                                                  "get_orientations_for_element"):
                df_ori = frame.get_orientations_for_element(elem.name)
                if df_ori is not None and not df_ori.empty:
                    start = df_ori[["X", "Y", "Z"]].values
                    direction = df_ori[["G_x", "G_y", "G_z"]].values
                    scale = 50.0
                    for i in range(len(start)):
                        arrow = pv.Arrow(start=start[i], direction=direction[i], scale=scale)
                        plotter.add_mesh(arrow, color=elem.color)

        legend_entries.append((group.name, group_entries))

    # ---------- plot fault meshes ----------
    if frame.fault_frame is not None:
        # add a "Faults" header in legend (bold/black achievable only approximately in pyvista legend)
        faults_header = ("Faults", "black")
        # we'll append "Faults" header at the top with the flatten stage below
        fault_entries = []

        for fault in frame.fault_frame.fault_elements:
            if fault_mesh_type not in fault.vertices or fault_mesh_type not in fault.edges:
                fault_mesh_type = FaultMeshType.UNMASKED  # fallback if requested type not present
            fv = fault.vertices[fault_mesh_type]
            ff = fault.edges[fault_mesh_type]
            if fv is None or ff is None or len(fv) == 0 or len(ff) == 0:
                continue
            faces_flat = _faces_to_vtk(np.asarray(ff))
            try:
                fmesh = pv.PolyData(np.asarray(fv), faces_flat)
            except Exception:
                fmesh = pv.PolyData(np.asarray(fv), faces_flat.astype(np.int64, copy=False))
            fcolor = fault.color or "black"
            plotter.add_mesh(
                fmesh,
                color=fcolor,
                opacity=fault_opacity,
                label=f"Fault: {fault.name}"
            )
            fault_entries.append((f"• {fault.name}", fcolor))

        # prepend a header row
        legend_entries = [("Faults", fault_entries)] + legend_entries

    # ---------- legend like before (youngest at top; group bold-ish) ----------
    # flatten & reverse (PyVista renders top-down)
    flat_legend = []
    for group_name, entries in legend_entries:
        flat_legend.append((f"{group_name}", "black"))
        for label, col in entries:
            flat_legend.append((label, col))

    plotter.add_legend(labels=flat_legend, size=(0.22, 0.22), loc="lower right", face="rectangle")

    # bounds/grid + camera
    if frame.grid is not None:
        plotter.show_bounds(bounds=frame.grid.extent, location="furthest", grid=True)

    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show:
        plotter.show()
    return plotter


def plot_structural_model_2D(
        frame: StructuralFrame,
        *,
        axis: Literal['x', 'y', 'z'] = 'y',
        index: Optional[int] = None,
        show_result: bool = True,
        show_fault_contours: bool = True,
        show_input_data: bool = True,
        title_suffix: str = "",
) -> None:
    """
    2D slice plot of a structural model showing lithology block, scalar contours, and input data.

    Parameters
    ----------
    frame : StructuralFrame
        Structural frame to visualize.
    axis : {'x', 'y', 'z'}, default 'y'
        Axis along which to slice.
    index : int, optional
        Slice index along the chosen axis. Defaults to the mid-point.
    show_result : bool, default True
        Overlay the lithology block as a colored background if available.
    show_fault_contours : bool, default True
        Overlay fault isolines from their scalar fields.
    show_input_data : bool, default True
        Overlay input surface points and orientation arrows.
    title_suffix : str, default ""
        Optional string appended to the plot title.
    """
    warnings.simplefilter("always", UserWarning)

    assert axis in ('x', 'y', 'z'), "Axis must be 'x', 'y', or 'z'."

    dim = {'x': 0, 'y': 1, 'z': 2}[axis]

    x = frame.grid.gridx
    y = frame.grid.gridy
    z = frame.grid.gridz

    if index is None:
        index = {'x': len(x), 'y': len(y), 'z': len(z)}[axis] // 2

    #  Compute spacing for imshow extent
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)
    dz = (z[-1] - z[0]) / (len(z) - 1)

    if axis == 'x':
        extent = (
            y[0] - dy / 2, y[-1] + dy / 2,
            z[0] - dz / 2, z[-1] + dz / 2
        )
        x_coords, y_coords = np.meshgrid(y, z, indexing='ij')
    elif axis == 'y':
        extent = (
            x[0] - dx / 2, x[-1] + dx / 2,
            z[0] - dz / 2, z[-1] + dz / 2
        )
        x_coords, y_coords = np.meshgrid(x, z, indexing='ij')
    else:  # 'z'
        extent = (
            x[0] - dx / 2, x[-1] + dx / 2,
            y[0] - dy / 2, y[-1] + dy / 2
        )
        x_coords, y_coords = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- lithology (optional) ----
    if show_result:
        has_result = (
            frame.lith_block is not None
            and np.size(frame.lith_block) > 0
        )
        if has_result is False:
            warnings.warn(
                "show_result=True but no lithology block found. "
                "Plotting input data only.",
                UserWarning
            )
        else:
            id_to_color = {0: "#cccccc"}  # grey base
            id_to_label = {0: "Basement"}
            if frame.lith_block is not None:
                slice_lith = np.take(frame.lith_block, index, axis=dim)
                slice_lith = slice_lith.T

                for group in frame.structural_groups:
                    for elem in group.structural_elements:
                        if elem.id:
                            id_to_color[elem.id] = elem.color
                            id_to_label[elem.id] = f"{group.name} | {elem.name}"

                sorted_ids = sorted(id_to_color)
                colors = [id_to_color[i] for i in sorted_ids]
                cmap = ListedColormap(colors)
                norm = BoundaryNorm(sorted_ids + [sorted_ids[-1] + 1], len(colors))
                ax.imshow(slice_lith, origin='lower', cmap=cmap, norm=norm, extent=extent, alpha=1, zorder=-100)

    # ---- faults as contours from their scalar fields ----
    if show_fault_contours and frame.fault_frame is not None:
        for fault_index, fault in enumerate(frame.fault_frame.fault_elements):
            f_sf = fault.scalar_field
            f_sv = fault.scalar_value
            if f_sf is None or f_sv is None:
                continue
            f_slice = np.take(f_sf, index, axis=dim).T

            # if a result was already computed
            if show_result and frame.lith_block is not None:
                # this mask is the age mask defined by youngest group affected by this fault
                group_index = frame.fault_activity_verbose[fault.name]["youngest_group_index"] - 1
                if group_index >= 0:
                    fault_mask = np.take(frame.structural_groups[group_index].get_mask(), index, axis=dim).T
                else:
                    fault_mask = np.zeros_like(f_slice, dtype=bool)

                f_slice_masked = np.ma.array(f_slice, mask=fault_mask)
            else:
                # Fallback to unmasked contour if no result or mask available
                f_slice_masked = f_slice
            fcol = fault.color or "black"
            CSf = ax.contour(x_coords, y_coords, f_slice_masked.T, levels=[f_sv], colors=[fcol], linewidths=1.5,
                             linestyles="-", zorder=10000)
            ax.clabel(CSf, fmt={f_sv: f"{fault.name}"}, fontsize=15)

    # ---- input data (optional) ----
    if show_input_data:
        for group in frame.structural_groups:
            for elem in group.structural_elements:
                df_pts = frame.get_surface_points_for_element(elem.name)
                df_ori = frame.get_orientations_for_element(elem.name)

                if df_pts is not None and not df_pts.empty:
                    if axis == 'x':
                        ax.scatter(df_pts["Y"], df_pts["Z"], color=elem.color, s=30, edgecolors='black', linewidths=0.5)
                    elif axis == 'y':
                        ax.scatter(df_pts["X"], df_pts["Z"], color=elem.color, s=30, edgecolors='black', linewidths=0.5)
                    else:
                        ax.scatter(df_pts["X"], df_pts["Y"], color=elem.color, s=30, edgecolors='black', linewidths=0.5)

                if df_ori is not None and not df_ori.empty:
                    if axis == 'x':
                        xs, ys, u, v = df_ori["Y"], df_ori["Z"], df_ori["G_y"], df_ori["G_z"]
                    elif axis == 'y':
                        xs, ys, u, v = df_ori["X"], df_ori["Z"], df_ori["G_x"], df_ori["G_z"]
                    else:
                        xs, ys, u, v = df_ori["X"], df_ori["Y"], df_ori["G_x"], df_ori["G_y"]

                    ax.quiver(xs, ys, u, v,
                              angles='xy', scale_units='xy', scale=0.03,
                              width=0.01, headwidth=3, headlength=4, headaxislength=3,
                              color=elem.color, edgecolors='black', linewidths=0.5)

    # ---- legend (group bold + elements) ----
    axis_labels = {'x': ("Y", "Z"), 'y': ("X", "Z"), 'z': ("X", "Y")}
    legend_handles = []
    for group in frame.structural_groups:
        legend_handles.append(plt.Line2D([0], [0], color='black', lw=0, label=rf"$\bf{{{group.name}}}$"))
        for elem in group.structural_elements:
            legend_handles.append(plt.Line2D([0], [0], color=elem.color, lw=3, label=f"{elem.name}"))

    # add faults header + lines
    if frame.fault_frame is not None and frame.fault_frame.fault_elements:
        legend_handles.append(plt.Line2D([0], [0], color='black', lw=0, label=rf"$\bf{{Faults}}$"))
        for fault in frame.fault_frame.fault_elements:
            fcol = fault.color or "black"
            legend_handles.append(plt.Line2D([0], [0], color=fcol, lw=3, label=fault.name))

    by_label = {h.get_label(): h for h in legend_handles}
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8,
              frameon=True)

    ax.set_xlabel(axis_labels[axis][0])
    ax.set_ylabel(axis_labels[axis][1])
    base_title = f"{axis.upper()} Slice @ index {index}"
    ax.set_title(f"{base_title} — {title_suffix}" if title_suffix else base_title)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.tight_layout()
    plt.show()


def plot_fault_model_3D(
    fault_frame: FaultFrame,
    mesh_type: FaultMeshType = FaultMeshType.UNMASKED,
    show_surface_meshes: bool = True,
    show_input_data: bool = True,
    show_domain_map: bool = True,
    domain_opacity: float = 0.15,
    notebook: bool = False,
    show: bool = True,
) -> pv.Plotter:
    """
    3D PyVista plot of a fault model with domain map, fault meshes, and input data.

    Parameters
    ----------
    fault_frame : FaultFrame
        Fault frame to visualize.
    mesh_type : FaultMeshType, default FaultMeshType.UNMASKED
        Mesh type for fault surfaces.
    show_surface_meshes : bool, default True
        Whether to render fault surface meshes.
    show_input_data : bool, default True
        Whether to render input surface points and orientation arrows.
    show_domain_map : bool, default True
        Whether to render translucent domain blocks.
    domain_opacity : float, default 0.15
        Opacity of domain block visualizations.
    notebook : bool, default False
        Use off-screen/notebook mode for PyVista.
    show : bool, default True
        Call plotter.show() at the end. Set False to add further actors before displaying.

    Returns
    -------
    pv.Plotter
        The PyVista plotter instance (useful when show=False).
    """
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(notebook=notebook)

    legend_entries_faults = []
    legend_entries_domains = []

    faults = fault_frame.fault_elements or []

    # -------------------------
    # (A) Domain map as translucent voxel blocks (optional)
    # -------------------------
    domain_map = fault_frame.domain_map
    grid = fault_frame.grid

    if show_domain_map and domain_map is not None and grid is not None:
        # Get coordinate vectors (same pattern as your 2D)
        def _get_coord(obj, name):
            val = getattr(obj, name, None)
            if val is not None:
                return val
            sub = obj.grid
            if sub is not None:
                return getattr(sub, name, None)
            return None

        x = np.asarray(_get_coord(grid, "gridx"))
        y = np.asarray(_get_coord(grid, "gridy"))
        z = np.asarray(_get_coord(grid, "gridz"))

        # spacing + origin aligned to your 2D half-cell padding convention
        dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.0
        dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 1.0
        dz = (z[-1] - z[0]) / (len(z) - 1) if len(z) > 1 else 1.0

        origin = (x[0] - dx / 2, y[0] - dy / 2, z[0] - dz / 2)
        spacing = (dx, dy, dz)

        dm = np.asarray(domain_map)
        # dm is assumed shaped (nx, ny, nz) as cell values
        nx, ny, nz = dm.shape

        # VTK ImageData uses POINT dimensions; cell dims are (nx,ny,nz) => point dims (nx+1,ny+1,nz+1)
        img = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=spacing,
            origin=origin,
        )

        # Attach as CELL data. Ordering can be tricky; this is the most common working default.
        # If colors look permuted/swapped, change order='F' to order='C'.
        img.cell_data["domain"] = dm.ravel(order="F").astype(np.float32)

        # Discrete domain colors like the 2D logic (Set3)
        domain_values = np.unique(dm)
        domain_values = domain_values[~np.isnan(domain_values)]
        domain_values = np.array(sorted(domain_values))

        if len(domain_values) > 0:
            base_cmap = plt.get_cmap("Set3")
            colors = base_cmap(np.linspace(0, 1, len(domain_values)))

            # Render each domain as a translucent block
            for i, dv in enumerate(domain_values):
                color = tuple(colors[i, :3])  # RGB

                # threshold around the discrete id
                th = img.threshold(
                    (float(dv) - 0.5, float(dv) + 0.5),
                    scalars="domain"
                )

                if th.n_cells == 0:
                    continue

                # show as filled voxels (no edges) with low opacity tint
                plotter.add_mesh(
                    th,
                    color=color,
                    opacity=domain_opacity,
                    show_edges=False,
                    lighting=False,
                )

                legend_entries_domains.append((f"■ Domain {int(dv) if float(dv).is_integer() else dv}", color))

    # -------------------------
    # (B) Fault meshes + input data
    # -------------------------
    for fault in faults:
        fcol = fault.color or "black"
        fname = getattr(fault, "name", "fault")

        # mesh (optional, skip cleanly if not computed)
        if show_surface_meshes:
            verts = fault.vertices
            edges = fault.edges
            fv = verts.get(mesh_type) if isinstance(verts, dict) else None
            ff = edges.get(mesh_type) if isinstance(edges, dict) else None

            if fv is not None and ff is not None and len(fv) > 0 and len(ff) > 0:
                faces_flat = _faces_to_vtk(np.asarray(ff))
                try:
                    mesh = pv.PolyData(np.asarray(fv), faces_flat)
                except Exception:
                    mesh = pv.PolyData(np.asarray(fv), faces_flat.astype(np.int64, copy=False))

                plotter.add_mesh(mesh, color=fcol, name=fname, label=fname)
                legend_entries_faults.append((f"• {fname}", fcol))

        # points (optional, show even if no meshes)
        if show_input_data and fault_frame.fault_surface_points_df is not None:
            df_points = fault_frame.get_surface_points_for_element(fname)
            if df_points is not None and not df_points.empty:
                cloud = pv.PolyData(df_points[["X", "Y", "Z"]].values)
                plotter.add_points(
                    cloud,
                    color=fcol,
                    point_size=8,
                    render_points_as_spheres=True
                )

        # orientations (optional, show even if no meshes)
        if show_input_data and fault_frame.fault_orientations_df is not None:
            df_ori = fault_frame.get_orientations_for_element(fname)
            if df_ori is not None and not df_ori.empty:
                start = df_ori[["X", "Y", "Z"]].values
                direction = df_ori[["G_x", "G_y", "G_z"]].values

                # scale arrow relative to grid X-extent so it stays visible at any coordinate scale
                grid_obj = getattr(fault_frame, "grid", None)
                extent_vals = getattr(grid_obj, "extent", None) if grid_obj is not None else None
                scale = (extent_vals[1] - extent_vals[0]) * 0.05 if extent_vals is not None else 50.0
                for i in range(len(start)):
                    arrow = pv.Arrow(start=start[i], direction=direction[i], scale=scale)
                    plotter.add_mesh(arrow, color=fcol)

    # -------------------------
    # (C) Legend (Domains + Faults)
    # -------------------------
    flat_legend = []
    if legend_entries_domains:
        flat_legend.append(("Domains", "black"))
        flat_legend.extend(legend_entries_domains)

    if legend_entries_faults:
        flat_legend.append(("Faults", "black"))
        flat_legend.extend(legend_entries_faults)

    if flat_legend:
        plotter.add_legend(
            labels=flat_legend,
            size=(0.28, 0.28),
            loc="lower right",
            face="rectangle",
        )

    # -------------------------
    # (D) Bounds + camera
    # -------------------------
    if grid is not None:
        plotter.show_bounds(bounds=grid.extent, location="furthest", grid=True)

    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show:
        plotter.show()
    return plotter


def plot_fault_model_2D(
        fault_frame: FaultFrame,
        *,
        axis: Literal['x', 'y', 'z'] = "y",
        index: Optional[int] = None,
        show_results: bool = True,
        show_input_data: bool = True,
        show_fault_contours: bool = True,
) -> None:
    """
    2D slice plot for a FaultFrame.

    - If show_results=True AND a domain_map is present: plots domain map + fault isolines.
    - Otherwise: plots only input data (points) if available.

    Extent is computed from grid coordinates (grid.gridx/gridy/gridz) with half-cell padding,
    so imshow/contour align properly.
    """

    assert axis in ("x", "y", "z"), "Axis must be 'x', 'y', or 'z'."

    if fault_frame.grid is None:
        raise ValueError("FaultFrame has no grid associated.")

    grid = fault_frame.grid

    # --- safe attribute getter ---
    def _get_coord(obj, name):
        val = getattr(obj, name, None)
        if val is not None:
            return val

        sub = obj.grid
        if sub is not None:
            return getattr(sub, name, None)

        return None

    x = _get_coord(grid, "gridx")
    y = _get_coord(grid, "gridy")
    z = _get_coord(grid, "gridz")

    if x is None or y is None or z is None:
        raise ValueError("Grid must provide gridx, gridy, gridz coordinate arrays.")

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if index is None:
        index = {'x': len(x), 'y': len(y), 'z': len(z)}[axis] // 2

    # spacing for half-cell padded extent
    dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.0
    dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 1.0
    dz = (z[-1] - z[0]) / (len(z) - 1) if len(z) > 1 else 1.0

    # mapping to np.take axis (same as your inspiration)
    dim = {"x": 0, "y": 1, "z": 2}[axis]

    if axis == "x":
        extent = (y[0] - dy / 2, y[-1] + dy / 2, z[0] - dz / 2, z[-1] + dz / 2)
        Xc, Yc = np.meshgrid(y, z, indexing="ij")  # for contour coordinates
        xlabel, ylabel = "Y", "Z"
    elif axis == "y":
        extent = (x[0] - dx / 2, x[-1] + dx / 2, z[0] - dz / 2, z[-1] + dz / 2)
        Xc, Yc = np.meshgrid(x, z, indexing="ij")
        xlabel, ylabel = "X", "Z"
    else:  # "z"
        extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
        Xc, Yc = np.meshgrid(x, y, indexing="ij")
        xlabel, ylabel = "X", "Y"

    domain_map = fault_frame.domain_map
    faults = fault_frame.fault_elements or []

    has_result = show_results and domain_map is not None and np.size(domain_map) > 0
    if show_results and not has_result:
        warnings.warn(
            "show_results=True but no domain_map found. Plotting input data only.",
            UserWarning,
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---------------------------
    # RESULTS: domain map + isolines
    # ---------------------------
    if has_result:
        domain_slice = np.take(domain_map, index, axis=dim).T

        domain_values = np.unique(domain_slice)
        domain_values = domain_values[~np.isnan(domain_values)]

        if len(domain_values) > 0:
            base_cmap = plt.get_cmap("Set3")
            colors = base_cmap(np.linspace(0, 1, len(domain_values)))
            cmap = ListedColormap(colors)
            bounds = np.append(domain_values, domain_values[-1] + 1)
            norm = BoundaryNorm(bounds, cmap.N)

            im = ax.imshow(
                domain_slice,
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
                alpha=1.0,
            )
            cbar = plt.colorbar(im, ax=ax, ticks=domain_values)
            cbar.set_label("Fault Domain")

        # fault contours (only if requested)
        if show_fault_contours:
            handles = []
            for fault in faults:
                sf = fault.scalar_field
                sv = fault.scalar_value
                if sf is None or sv is None:
                    continue

                f_slice = np.take(sf, index, axis=dim).T
                fcol = fault.color or "black"

                CS = ax.contour(
                    Xc, Yc,
                    f_slice.T,
                    levels=[sv],
                    colors=[fcol],
                    linewidths=1.5,
                    linestyles="-",
                )

                # label directly on contour line
                ax.clabel(
                    CS,
                    fmt={sv: getattr(fault, "name", "fault")},
                    inline=True,
                    fontsize=15
                )

                handles.append(plt.Line2D([0], [0], color=fcol, lw=1.5, label=getattr(fault, "name", "fault")))

            if handles:
                by_label = {h.get_label(): h for h in handles}
                ax.legend(by_label.values(), by_label.keys(), title="Faults", loc="lower left")

    # ---------------------------
    # INPUT DATA ONLY (project all points)
    # ---------------------------

    if show_input_data:
        for fault in faults:
            # Preferred: dataframe getter (X,Y,Z)
            getter = fault_frame.get_surface_points_for_element
            if getter is not None:
                df_pts = getter(getattr(fault, "name", "fault"))

                if df_pts is None or getattr(df_pts, "empty", False):
                    continue

                fcol = fault.color or "black"
                if axis == "x":
                    ax.scatter(df_pts["Y"], df_pts["Z"], color=fcol, s=30,
                               edgecolors="black", linewidths=0.5, label=getattr(fault, "name", "fault"))
                elif axis == "y":
                    ax.scatter(df_pts["X"], df_pts["Z"], color=fcol, s=30,
                               edgecolors="black", linewidths=0.5, label=getattr(fault, "name", "fault"))
                else:  # z
                    ax.scatter(df_pts["X"], df_pts["Y"], color=fcol, s=30,
                               edgecolors="black", linewidths=0.5, label=getattr(fault, "name", "fault"))

            # ---- orientations as arrows (optional, if getter exists and data present) ----
            ori_getter = fault_frame.get_orientations_for_element
            if ori_getter is not None:
                df_ori = ori_getter(getattr(fault, "name", "fault"))
                if df_ori is not None and not getattr(df_ori, "empty", False):
                    if axis == "x":
                        xs, ys, u, v = df_ori["Y"], df_ori["Z"], df_ori["G_y"], df_ori["G_z"]
                    elif axis == "y":
                        xs, ys, u, v = df_ori["X"], df_ori["Z"], df_ori["G_x"], df_ori["G_z"]
                    else:  # z
                        xs, ys, u, v = df_ori["X"], df_ori["Y"], df_ori["G_x"], df_ori["G_y"]

                    ax.quiver(
                        xs, ys, u, v,
                        angles="xy", scale_units="xy", scale=0.03,
                        width=0.01, headwidth=3, headlength=4, headaxislength=3,
                        color=fcol, edgecolors="black", linewidths=0.5,
                        zorder=11,  # above imshow + points
                    )

        # optional: tidy legend (avoid duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = {lab: h for h, lab in zip(handles, labels)}
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), title="Input data", loc="lower left")

    # --- cosmetics ---
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{axis.upper()} Slice @ index {index}")
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.tight_layout()
    plt.show()
