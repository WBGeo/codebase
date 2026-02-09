import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from core.object_components import InputData_StructuralElements, GeomodelResults, MeshResults
import warnings

def _faces_to_vtk(faces_arr: np.ndarray) -> np.ndarray:
    """Convert (N,3) int triangle array to VTK flat face array."""
    faces_arr = np.asarray(faces_arr)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("faces must be (N, 3) triangle indices")
    return np.hstack([np.full((faces_arr.shape[0], 1), 3, dtype=np.int64), faces_arr.astype(np.int64)]).ravel()

def plot_structural_model_3D(
    frame,
    *,
    mesh_type="masked",              # "masked" | "unmasked" | "combined" (if present)
    fault_mesh_type="masked",        # same options for fault meshes
    show_surface_meshes=True,
    show_points=True,
    show_orientations=True,
    fault_opacity=0.35,
    notebook=False,
    show=True,
):
    """
    PyVista 3D plot of structural elements (per-domain meshes) + faults, with your preferred legend style.
    - frame.structural_groups[...] elements store meshes by domain via element.meshes_for_domain(domain_id)
      which returns a dict like {"masked": (V, F), "unmasked": ..., "combined": ...}
    - fault_frame.fault_elements[i].vertices / .edges are used for fault meshes.
    """
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(notebook=notebook)

    # ---------- plot structural meshes, points, orientations ----------
    legend_entries = []  # [(group_name, [(• elem, color), ...]), ...] in youngest->oldest order

    for group in frame.structural_groups:  # order preserved: youngest->oldest
        group_entries = []
        for elem in group.structural_elements:
            # legend entry
            group_entries.append((f"• {elem.name}", elem.color))

            # meshes (per domain)
            if show_surface_meshes:
                for mesh_elem in group.structural_elements:  # <--- renamed variable
                    plotter.add_mesh(
                        pv.PolyData(mesh_elem._vertices[mesh_type],
                                    np.insert(mesh_elem._edges[mesh_type], 0, 3, axis=1).ravel()),
                        color=mesh_elem.color, opacity=1.0, label=f"{group.name} | {mesh_elem.name}")

            # surface points
            if show_points and hasattr(frame, "get_surface_points_for_element"):
                df_points = frame.get_surface_points_for_element(elem.name)
                if df_points is not None and not df_points.empty:
                    cloud = pv.PolyData(df_points[["X", "Y", "Z"]].values.astype(np.float32))
                    plotter.add_points(cloud, color=elem.color, point_size=8, render_points_as_spheres=True)

            # orientations
            if show_orientations and getattr(frame, "orientations", None) is not None and hasattr(frame,
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
    if frame._fault_frame is not None:
        # add a "Faults" header in legend (bold/black achievable only approximately in pyvista legend)
        faults_header = ("Faults", "black")
        # we’ll append “Faults” header at the top with the flatten stage below
        fault_entries = []

        for fault in getattr(frame._fault_frame, "fault_elements", []):
            if fault_mesh_type not in getattr(fault, "vertices", {}) or fault_mesh_type not in getattr(fault, "edges", {}):
                fault_mesh_type = "unmasked"  # fallback if requested type not present
            fv = getattr(fault, "vertices", None)[fault_mesh_type]
            ff = getattr(fault, "edges", None)[fault_mesh_type]
            if fv is None or ff is None or len(fv) == 0 or len(ff) == 0:
                continue
            faces_flat = _faces_to_vtk(np.asarray(ff))
            try:
                fmesh = pv.PolyData(np.asarray(fv), faces_flat)
            except Exception:
                fmesh = pv.PolyData(np.asarray(fv), faces_flat.astype(np.int64, copy=False))
            fcolor = getattr(fault, "color", None) or "black"
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
    if getattr(frame, "grid", None) is not None:
        plotter.show_bounds(bounds=frame.grid.extent, location="furthest", grid=True)

    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show:
        plotter.show()
    return plotter


def plot_structural_model_2D(
    frame,
    *,
    axis='y',
    index=0,
    show_result=True,
    show_fault_contours=True,
    show_input_data=True,
):

    """
    2D slice of model with:
      - lithology block slice (if provided)
      - structural contours per group from per-domain scalar fields
      - fault contours from fault scalar fields at their scalar_value
      - legend matching your style (group bold + element lines)
    """
    warnings.simplefilter("always", UserWarning)

    assert axis in ('x', 'y', 'z'), "Axis must be 'x', 'y', or 'z'."

    dim = {'z': 0, 'y': 1, 'x': 2}[axis]

    x = frame.grid.gridx
    y = frame.grid.gridy
    z = frame.grid.gridz

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
        print(f"has_result: {has_result}")
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
                        if getattr(elem, "id", None):
                            id_to_color[elem.id] = elem.color
                            id_to_label[elem.id] = f"{group.name} | {elem.name}"

                sorted_ids = sorted(id_to_color)
                colors = [id_to_color[i] for i in sorted_ids]
                cmap = ListedColormap(colors)
                norm = BoundaryNorm(sorted_ids + [sorted_ids[-1] + 1], len(colors))
                ax.imshow(slice_lith, origin='lower', cmap=cmap, norm=norm, extent=extent, alpha=1)

    # ---- faults as contours from their scalar fields ----
    if show_fault_contours and frame.fault_frame is not None:
        for fault in frame.fault_frame.fault_elements:
            f_sf = getattr(fault, "scalar_field", None)
            f_sv = getattr(fault, "scalar_value", None)
            if f_sf is None or f_sv is None:
                continue
            f_slice = np.take(f_sf, index, axis=dim).T

            # if a result was already computed
            if show_result and frame.lith_block is not None:
                # this mask is the age mask defined by youngest group affected by this fault
                group_index = frame.fault_activity_verbose[fault.name]["youngest_group_index"] - 1
                fault_mask = np.take(frame.structural_groups[group_index].get_mask().T, index, axis=dim) if index >= 0 else np.ones_like(f_slice, dtype=bool)
                f_slice_masked = np.ma.array(f_slice, mask=fault_mask)
            else:
                # Fallback to unmasked contour if no result or mask available
                f_slice_masked = f_slice
            fcol = getattr(fault, "color", None) or "black"
            CSf = ax.contour(x_coords, y_coords, f_slice_masked.T, levels=[f_sv], colors=[fcol], linewidths=1.5, linestyles="-")
            ax.clabel(CSf, fmt={f_sv: f"Fault: {fault.name}"}, fontsize=7)

    # ---- input input_data (optional) ----
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
    if frame._fault_frame is not None and getattr(frame._fault_frame, "fault_elements", []):
        legend_handles.append(plt.Line2D([0], [0], color='black', lw=0, label=rf"$\bf{{Faults}}$"))
        for fault in frame._fault_frame.fault_elements:
            fcol = getattr(fault, "color", None) or "black"
            legend_handles.append(plt.Line2D([0], [0], color=fcol, lw=3, label=fault.name))

    by_label = {h.get_label(): h for h in legend_handles}
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)

    ax.set_xlabel(axis_labels[axis][0])
    ax.set_ylabel(axis_labels[axis][1])
    ax.set_title(f"{axis.upper()} Slice @ index {index}")
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.tight_layout()
    plt.show()


def plot_fault_model_3D(
    fault_frame,
    mesh_type="unmasked",
    show_surface_meshes=True,
    show_points=True,
    show_orientations=True,
    notebook=False,
    show=True
):
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(notebook=notebook)

    legend_entries = []
    for fault in getattr(fault_frame, "_fault_elements", []):
        # mesh
        if show_surface_meshes:
            fv = getattr(fault, "vertices", None)[mesh_type]
            ff = getattr(fault, "edges", None)[mesh_type]
            if fv is not None and ff is not None and len(fv) > 0 and len(ff) > 0:
                faces_flat = _faces_to_vtk(np.asarray(ff))
                try:
                    mesh = pv.PolyData(np.asarray(fv), faces_flat)
                except Exception:
                    mesh = pv.PolyData(np.asarray(fv), faces_flat.astype(np.int64, copy=False))
                plotter.add_mesh(mesh, color=fault.color, name=fault.name, label=fault.name)
                legend_entries.append((f"• {fault.name}", fault.color))

        # points
        if show_points and getattr(fault_frame, "_fault_surface_points_df", None) is not None:
            df_points = fault_frame.get_surface_points_for_element(fault.name)
            if df_points is not None and not df_points.empty:
                cloud = pv.PolyData(df_points[["X", "Y", "Z"]].values)
                plotter.add_points(cloud, color=fault.color, point_size=8, render_points_as_spheres=True)

        # orientations
        if show_orientations and getattr(fault_frame, "_fault_orientations_df", None) is not None:
            df_ori = fault_frame.get_orientations_for_element(fault.name)
            if df_ori is not None and not df_ori.empty:
                start = df_ori[["X", "Y", "Z"]].values
                direction = df_ori[["G_x", "G_y", "G_z"]].values
                scale = 50.0
                for i in range(len(start)):
                    arrow = pv.Arrow(start=start[i], direction=direction[i], scale=scale)
                    plotter.add_mesh(arrow, color=fault.color)

    if legend_entries:
        flat_legend = [("Faults", "black")]
        flat_legend.extend(legend_entries)

        plotter.add_legend(
            labels=flat_legend,
            size=(0.22, 0.22),
            loc="lower right",
            face="rectangle"
        )

    if getattr(fault_frame, "grid", None) is not None:
        plotter.show_bounds(bounds=fault_frame.grid.extent, location="furthest", grid=True)

    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show:
        plotter.show()
    return plotter


#TODO: Adapt mesh plotting to new Structure
def plot_mesh_3d(mesh_results: MeshResults, input_data: InputData_StructuralElements, colors=None, style="surface",
                 show_plotter=True) -> pv.Plotter:
    """
    Plot the mesh for process simulation in 3D.

    Args:
        mesh_results (MeshResults): The mesh to plot.
        input_data (InputData_StructuralElements): The input input_data for the structural geological model.
        colors (Optional(list)): List of colors to use for the different formations
        style (str): The style of the mesh to plot.
    """
    # Suppress the specific warning about points not being a float type
    warnings.filterwarnings("ignore",
                            message="Points is not a float type. This can cause issues when transforming or applying "
                                    "filters. Casting to ``np.float32``. Disable this by passing "
                                    "``force_float=False``.")

    formations = input_data.surface_points['formation'].unique()

    if input_data.faults is not None:
        n_faults = np.sum(input_data.faults)
    else:
        n_faults = 0

    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=not show_plotter)

    # Set default colors
    if colors is None:
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
                  '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']

        colors = ['#673ab7', '#34a853', '#ea4335', '#fbbc05', '#4285f4',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
                  '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']


    if input_data.faults is not None:

        data = input_data.mapping_object

        # Ensure all values are tuples
        for key in data:
            if isinstance(data[key], str):
                data[key] = (data[key],)

        fault_mask = []
        keys = list(data.keys())
        for i, key in enumerate(keys):
            fault_mask.extend([bool(input_data.faults[i])] * len(data[key]))

        fault_mask = ~np.array(fault_mask)

        masked_colors = [color for color, m in zip(colors, fault_mask) if m]
        masked_formations = [element for element, m in zip(formations, fault_mask) if m]

        # Add moose meshes to the plotter
        for i in range(len(formations) - np.sum(input_data.faults)):
            # plotter.add_mesh(mesh[0][i], show_edges=True, style='wireframe', color=masked_colors[i],
            #                  label=masked_formations[i])
            plotter.add_mesh(mesh_results.mesh[i], show_edges=True, style=style, color=masked_colors[i],
                             label=masked_formations[i])
    else:
        # Add moose meshes to the plotter
        for i in range(len(formations)):
            # plotter.add_mesh(mesh[0][i], show_edges=True, style='wireframe', color=colors[i],
            #                 label=formations[i])
            plotter.add_mesh(mesh_results.mesh[i], show_edges=True, style=style, color=colors[i],
                             label=formations[i])

    # Add basement with extra label
    basement_ID = int(len(formations) - n_faults)
    plotter.add_mesh(mesh_results.mesh[basement_ID], show_edges=True,
                     style=style,
                     color=colors[len(formations)])

    plotter.add_legend(size=(0.13, 0.13), loc='lower right', face='circle')

    # Set the bounds and grid of the plotter
    plotter.show_bounds(grid=True)

    # Set the camera position
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show_plotter:
        # Display the interactive plot
        plotter.show()

    return plotter

