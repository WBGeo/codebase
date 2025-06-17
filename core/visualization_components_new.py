import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches



def visualize_structural_frame(frame, show_surface_meshes=True, show_points=True,
                               show_orientations=True, notebook=False, show=True):
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(notebook=notebook)

    legend_entries = []

    # Build entries in order from youngest to oldest
    for group in frame.structural_groups:  # Order preserved: youngest to oldest
        group_entries = []
        for element in group.structural_elements:
            group_entries.append((f"• {element.name}", element.color))
        legend_entries.append((group.name, group_entries))

    # Build ordered legend entries (youngest to oldest, consistent with structural_frame)
    legend_entries = []
    for group in frame.structural_groups:  # Preserves order: youngest to oldest
        group_entries = []
        for element in group.structural_elements:
            group_entries.append((f"• {element.name}", element.color))

            # Plot surface mesh
            if show_surface_meshes and "masked" in element.vertices and "masked" in element.edges:
                vertices = element.vertices["masked"]
                edges = element.edges["masked"]
                if len(vertices) > 0 and len(edges) > 0:
                    faces = np.insert(edges, 0, 3, axis=1).ravel()
                    mesh = pv.PolyData(vertices, faces)
                    plotter.add_mesh(mesh, color=element.color, name=element.name, label=f"{group.name} | {element.name}")

            # Plot surface points
            if show_points and not frame.surface_points.empty:
                df_points = frame.get_surface_points_for_element(element.name)
                if not df_points.empty:
                    cloud = pv.PolyData(df_points[["X", "Y", "Z"]].values)
                    plotter.add_points(cloud, color=element.color, point_size=8, render_points_as_spheres=True)

            # Plot orientations
            if show_orientations and frame.orientations is not None:
                df_ori = frame.get_orientations_for_element(element.name)
                if df_ori is not None and not df_ori.empty:
                    start = df_ori[["X", "Y", "Z"]].values
                    direction = df_ori[["G_x", "G_y", "G_z"]].values
                    scale = 50.0
                    for i in range(len(start)):
                        arrow = pv.Arrow(start=start[i], direction=direction[i], scale=scale)
                        plotter.add_mesh(arrow, color=element.color)

        legend_entries.append((group.name, group_entries))

    # Sort entries by group and order of appearance
    legend_entries.sort(key=lambda x: (x[0], x[1]))

    # Flatten and reverse legend for PyVista's top-down rendering
    flat_legend = []
    for group_name, entries in reversed(legend_entries):  # Reversed: youngest group at top
        flat_legend.append((f"{group_name}", "black"))  # Neutral color for group name
        flat_legend.extend(entries)

    plotter.add_legend(
        labels=flat_legend,
        size=(0.22, 0.22),
        loc='lower right',
        face='rectangle'
    )

    # Add bounds/grid
    plotter.show_bounds(bounds=frame.grid.extent, location="furthest", grid=True)

    # Camera setup
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    # Return or show
    if show:
        plotter.show()
    return plotter

#%%

def plot_structural_slice(frame, lith_block=None, axis='z', index=0, show_scalar_contours=True, show_input_data=True):
    """
    Plot a 2D slice of the structural model using matplotlib.

    Args:
        frame: StructuralFrame containing grid, scalar fields and metadata.
        lith_block: 3D numpy array of lithology block IDs.
        axis: 'x', 'y', or 'z' indicating the slice direction.
        index: Index along the axis to slice.
        show_scalar_contours: Overlay scalar field contours if available.
        show_input_data: Plot surface points and orientations.
    """

    assert axis in ('x', 'y', 'z'), "Axis must be 'x', 'y', or 'z'."
    dim = {'z': 0, 'y': 1, 'x': 2}[axis]

    x = frame.grid.gridx
    y = frame.grid.gridy
    z = frame.grid.gridz

    if axis == 'x':
        extent = (y[0], y[-1], z[0], z[-1])
        x_coords, y_coords = np.meshgrid(y, z, indexing='ij')
    elif axis == 'y':
        extent = (x[0], x[-1], z[0], z[-1])
        x_coords, y_coords = np.meshgrid(x, z, indexing='ij')
    else:  # 'z'
        extent = (x[0], x[-1], y[0], y[-1])
        x_coords, y_coords = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lithology block slice
    id_to_color = {0: "#cccccc"}  # grey for basement
    id_to_label = {0: "Basement"}

    if lith_block is not None:
        slice_lith = np.take(lith_block, index, axis=dim)
        for group in frame.structural_groups:
            for elem in group.structural_elements:
                if elem.id:
                    id_to_color[elem.id] = elem.color
                    id_to_label[elem.id] = f"{group.name} | {elem.name}"

        sorted_ids = sorted(id_to_color)
        colors = [id_to_color[i] for i in sorted_ids]
        labels = [id_to_label[i] for i in sorted_ids]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(sorted_ids + [sorted_ids[-1] + 1], len(colors))

        ax.imshow(slice_lith, origin='lower', cmap=cmap, norm=norm,
                  extent=extent, alpha=0.6)

    # Plot scalar field contours
    if show_scalar_contours:
        for group in frame.structural_groups:
            if group.scalar_field is None:
                continue
            sf_slice = np.take(group.scalar_field, index, axis=dim)
            for elem in group.structural_elements:
                if elem.scalar_value is not None:
                    CS = ax.contour(x_coords, y_coords, sf_slice.T,
                                    levels=[elem.scalar_value],
                                    colors=[elem.color], linewidths=1.2)
                    ax.clabel(CS, fmt={elem.scalar_value: elem.name}, fontsize=7)



    # Plot input data
    if show_input_data:
        for group in frame.structural_groups:
            for elem in group.structural_elements:
                df_pts = frame.get_surface_points_for_element(elem.name)
                df_ori = frame.get_orientations_for_element(elem.name)

                # Extract correct 2D coordinates and vectors based on axis
                if axis == 'x':
                    xs = df_ori["Y"]
                    ys = df_ori["Z"]
                    u = df_ori["G_y"]
                    v = df_ori["G_z"]
                elif axis == 'y':
                    xs = df_ori["X"]
                    ys = df_ori["Z"]
                    u = df_ori["G_x"]
                    v = df_ori["G_z"]
                elif axis == 'z':
                    xs = df_ori["X"]
                    ys = df_ori["Y"]
                    u = df_ori["G_x"]
                    v = df_ori["G_y"]

                if not df_pts.empty:
                    if axis == 'x':
                        ax.scatter(df_pts["Y"], df_pts["Z"], color=elem.color, s=30, edgecolors='black', linewidths=0.5)
                    elif axis == 'y':
                        ax.scatter(df_pts["X"], df_pts["Z"], color=elem.color, s=30, edgecolors='black', linewidths=0.5)
                    else:  # 'z'
                        ax.scatter(df_pts["X"], df_pts["Y"], color=elem.color, s=30, edgecolors='black', linewidths=0.5)

                if df_ori is not None and not df_ori.empty:
                    if axis == 'x':
                        ax.quiver(xs, ys, u, v,
                                  angles='xy',
                                  scale_units='xy',
                                  scale=0.03,  # adjust for visibility
                                  width=0.01,
                                  headwidth=3,
                                  headlength=4,
                                  headaxislength=3,
                                  color=elem.color, edgecolors='black', linewidths=0.5)
                    elif axis == 'y':
                        ax.quiver(xs, ys, u, v,
                                  angles='xy',
                                  scale_units='xy',
                                  scale=0.03,  # adjust for visibility
                                  width=0.01,
                                  headwidth=3,
                                  headlength=4,
                                  headaxislength=3,
                                  color=elem.color, edgecolors='black', linewidths=0.5)
                    else:
                        ax.quiver(xs, ys, u, v,
                                  angles='xy',
                                  scale_units='xy',
                                  scale=0.03,  # adjust for visibility
                                  width=0.01,
                                  headwidth=3,
                                  headlength=4,
                                  headaxislength=3,
                                  color=elem.color,edgecolors='black', linewidths=0.5)

    # Axis labeling
    axis_labels = {
        'x': ("Y", "Z"),
        'y': ("X", "Z"),
        'z': ("X", "Y")
    }

    # Build legend handles from structural frame, not just lith block
    legend_handles = []

    for group in frame.structural_groups:
        # Add group label in black and bold
        legend_handles.append(
            plt.Line2D([0], [0], color='black', lw=0, label=rf"$\bf{{{group.name}}}$")
        )
        for elem in group.structural_elements:
            legend_handles.append(
                plt.Line2D([0], [0], color=elem.color, lw=3, label=f"{elem.name}")
            )

    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True
    )

    ax.set_xlabel(axis_labels[axis][0])
    ax.set_ylabel(axis_labels[axis][1])
    ax.set_title(f"{axis.upper()} Slice at Index {index}")
    ax.set_aspect("equal")

    # Legend outside
    if legend_handles:
        by_label = {h.get_label(): h for h in legend_handles}
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.tight_layout()
    plt.show()

