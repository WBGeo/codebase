import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
from core.object_components import InputData, GeomodelResults, MeshResults


def plot_2d(input_data: InputData, geomodel_results: GeomodelResults = None,
            show_results: bool = False, show_data: bool = True, colors=None, show_plot=True,
            direction="y", slice_int=None) -> (plt.Figure, plt.Axes):
    """
    Plot the input data and results in 2D.

    Args:
        input_data (InputData): The input data for the geological model.
        geomodel_results (GeomodelResults): The results of the geological model.
        show_results (bool): Whether to show the results.
        show_data (bool): Whether to show the input data.
        colors (Optional(list)): List of colors to use for the different formations.
        show_plot (bool): Whether to show the plot.
        direction (str): The direction of the slice, either 'x', 'y' or 'z'.
        slice_int (int): The index of the slice to plot.

    Returns:
        fig (plt.Figure): The figure object.
        ax (plt.Axes): The axis object
    """
    # TODO: Proper plotting options, directions etc - this is just a first draft

    # Set default colors
    if colors is None:
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
                  '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots()

    if direction == "x":
        extent = (float(input_data.extent[2]), float(input_data.extent[3]),
                  float(input_data.extent[4]), float(input_data.extent[5]))
    elif direction == "y":
        extent = (float(input_data.extent[0]), float(input_data.extent[1]),
                  float(input_data.extent[4]), float(input_data.extent[5]))
    elif direction == "z":
        extent = (float(input_data.extent[0]), float(input_data.extent[1]),
                  float(input_data.extent[2]), float(input_data.extent[3]))
    else:
        raise ValueError("Direction must be 'x', 'y' or 'z'.")

    # Set the x and y extent of the model
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Define unique formations
    formations = input_data.surface_points['formation'].unique()

    # Create a color map and norm
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(formations) + 1) - 0.5, len(formations))

    # # Plot the surface points
    # sc = ax.scatter(input_data.surface_points.X, input_data.surface_points.Z,
    #                 c=input_data.surface_points['formation'].apply(lambda x: np.where(formations == x)[0][0]),
    #                 cmap=cmap,
    #                 norm=norm,
    #                 edgecolors="black",
    #                 visible=show_data)
    #
    # # Plot the orientations if available
    # if input_data.orientations is not None:
    #     ax.quiver(input_data.orientations.X, input_data.orientations.Z,
    #               input_data.orientations.G_x, input_data.orientations.G_z,
    #               input_data.orientations['formation'].apply(lambda x: np.where(formations == x)[0][0]),
    #               angles='xy', scale_units='xy',
    #               cmap=cmap,
    #               norm=norm,
    #               edgecolors="black",
    #               linewidth=1,
    #               visible=show_data)
    # else:
    #     pass

    # Define coordinate mappings
    scatter_coords = {
        "x": ("Y", "Z"),
        "y": ("X", "Z"),
        "z": ("X", "Y")
    }

    quiver_coords = {
        "x": ("Y", "Z", "G_y", "G_z"),
        "y": ("X", "Z", "G_x", "G_z"),
        "z": ("X", "Y", "G_x", "G_y")
    }

    # Get the appropriate coordinates based on direction
    sc_x, sc_y = scatter_coords[direction]

    # Plot the surface points
    sc = ax.scatter(
        getattr(input_data.surface_points, sc_x),
        getattr(input_data.surface_points, sc_y),
        c=input_data.surface_points['formation'].apply(lambda x: np.where(formations == x)[0][0]),
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        visible=show_data
    )

    # Plot the orientations if available
    if input_data.orientations is not None:
        q_x, q_y, q_u, q_v = quiver_coords[direction]
        ax.quiver(
            getattr(input_data.orientations, q_x),
            getattr(input_data.orientations, q_y),
            getattr(input_data.orientations, q_u),
            getattr(input_data.orientations, q_v),
            input_data.orientations['formation'].apply(lambda x: np.where(formations == x)[0][0]),
            angles='xy', scale_units='xy',
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidth=1,
            visible=show_data
        )
    else:
        pass

    if geomodel_results is not None and show_results:

        # Reshape lithology block to resolution
        plot_block = np.round(geomodel_results.lith_block.reshape(geomodel_results.resolution), 0).astype(int)
        plot_block = plot_block

        # cut slice in the middle of the model
        if slice_int is None:
            slice_int = int(np.rint(input_data.resolution[1] / 2))

        if direction == "x":
            image = plot_block[slice_int, :, :].T
        elif direction == "y":
            image = plot_block[:, slice_int, :].T
        elif direction == "z":
            image = plot_block[:, :, slice_int].T
        else:
            raise ValueError("Direction must be 'x', 'y' or 'z'.")

        # Create a discrete color map for the lithology block
        if input_data.faults is not None:
            # retrieving correct colors for lithology block by removing fault colors
            filtered_colors = [color for color, flag in zip(colors, input_data.faults) if not flag]
            filtered_colors.extend(colors[len(input_data.faults):])
            cmap2 = mcolors.ListedColormap(filtered_colors)
        else:
            cmap2 = mcolors.ListedColormap(colors)

        # Make sure boundaries are one more than colors
        # specify boundaries based on values in lithology block
        # arr = np.unique(image) - 0.5
        arr = np.unique(geomodel_results.lith_block) - 0.5
        arr = np.append(arr, np.unique(arr)[-1] + 1)
        norm2 = mcolors.BoundaryNorm(arr, ncolors=len(arr) - 1)

        ax.imshow(image, origin='lower', zorder=-100, cmap=cmap2, norm=norm2, extent=extent)

        # TODO: Add contour solution here but this requires the scalar fields

    elif geomodel_results is None and show_results:
        raise ValueError("Can not show results without results data.")
    else:
        pass

    # Create a discrete colorbar
    cbar = plt.colorbar(sc, ticks=np.arange(len(formations)),
                        boundaries=np.arange(len(formations) + 1) - 0.5,
                        shrink=0.3)

    cbar.ax.set_yticklabels(formations)
    cbar.ax.invert_yaxis()

    # Grid lines and labels
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')

    if show_plot:
        plt.show()

    return fig, ax


def plot_3d(input_data: InputData, geomodel_results: GeomodelResults = None, show_results: bool = False, colors=None,
            surface_type="masked", show_plotter=True) -> pv.Plotter:
    """
    Plot the input data and results in 3D.

    Args:
        input_data (InputData): The input data for the geological model.
        geomodel_results (GeomodelResults): The results of the geological model.
        show_results (bool): Whether to show the results.
        colors (Optional(list)): List of colors to use for the different formations
        surface_type (str): The type of surface to plot, either 'masked', 'unmasked' or 'combined'.
        show_plotter (bool): Whether to show the plotter, or just return it
    """
    # Suppress the specific warning about points not being a float type
    warnings.filterwarnings("ignore",
                            message="Points is not a float type. This can cause issues when transforming or applying "
                                    "filters. Casting to ``np.float32``. Disable this by passing "
                                    "``force_float=False``.")

    pv.global_theme.allow_empty_mesh = True

    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=not show_plotter)

    # Set default colors
    if colors is None:
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
                  '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']

    formations = input_data.surface_points['formation'].unique()

    # Add the surface points and orientations to the plotter
    for i in range(len(input_data.surface_points.iloc[:, -1].unique())):
        # Get the current structural element
        current_element = input_data.surface_points.iloc[:, -1].unique()[i]

        # Add the surface points
        plotter.add_mesh(pv.PolyData(input_data.surface_points[input_data.surface_points.iloc[:, -1]
                                                               == current_element].iloc[:,
                                     :3].to_numpy()),
                         render_points_as_spheres=True,
                         point_size=10,
                         color=colors[i], label=formations[i])

        # Add the orientations if available
        if input_data.orientations is not None:
            points = pv.PolyData(input_data.orientations[input_data.orientations.iloc[:, -1]
                                                         == current_element].iloc[:, :3].to_numpy().astype(np.float32))

            points['vectors'] = input_data.orientations[input_data.orientations.iloc[:, -1]
                                                        == current_element].iloc[:, 3:6].to_numpy().astype(np.float32)

            # Create an arrow source and a Glyph object and add to plotter
            arrow = pv.Arrow()
            glyphs = points.glyph(orient='vectors', scale=False, geom=arrow,
                                  factor=int((input_data.extent[1] - input_data.extent[0]) / 100))
            plotter.add_mesh(glyphs, color=colors[i])

    if geomodel_results is not None and show_results:
        if surface_type == "masked":
            mesh_counter = 0
        elif surface_type == "unmasked":
            mesh_counter = 1
        elif surface_type == "combined":
            mesh_counter = 2
        else:
            raise ValueError("Surface type must be 'masked', 'unmasked' or 'combined'.")

            # Add the surface meshes
        for i in range(len(formations)):
            plotter.add_mesh(
                pv.PolyData(geomodel_results.surface_meshes_vertices[mesh_counter][i],
                            np.insert(geomodel_results.surface_meshes_edges[mesh_counter][i], 0, 3, axis=1).ravel()),
                color=colors[i])

    if geomodel_results is None and show_results:
        print("Can not show results without results data.")
    else:
        pass

    plotter.add_legend(size=(0.13, 0.13), loc='lower right', face='circle')

    # Set the bounds and grid of the plotter
    plotter.show_bounds(bounds=input_data.extent,
                        location="furthest",
                        grid=True)

    # Set the camera position
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show_plotter:
        # Display the interactive plot
        plotter.show()
    return plotter


def plot_mesh_3d(mesh_results: MeshResults, input_data: InputData, colors=None, style="surface",
                 show_plotter=True) -> pv.Plotter:
    """
    Plot the mesh for process simulation in 3D.

    Args:
        mesh_results (MeshResults): The mesh to plot.
        input_data (InputData): The input data for the structural geological model.
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
