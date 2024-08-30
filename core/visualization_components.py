import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
from core.object_components import InputData, GeomodelResults


def plot_2d(input_data: InputData, geomodel_results: GeomodelResults = None, show_results: bool = False, colors=None):
    """
    Plot the input data and results in 2D.

    Args:
        input_data (InputData): The input data for the geological model.
        geomodel_results (GeomodelResults): The results of the geological model.
        show_results (bool): Whether to show the results.
        colors (Optional(list)): List of colors to use for the different formations.
    """
    # TODO: Proper plotting options, directions etc - this is just a first draft

    # Set default colors
    if colors is None:
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd']

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots()

    # Set the x and y extent of the model
    ax.set_xlim(input_data.extent[0], input_data.extent[1])
    ax.set_ylim(input_data.extent[4], input_data.extent[5])

    # Define unique formations
    formations = input_data.surface_points['formation'].unique()

    # Create a color map and norm
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(formations) + 1) - 0.5, len(formations))

    # Plot the surface points
    sc = ax.scatter(input_data.surface_points.X, input_data.surface_points.Z,
                    c=input_data.surface_points['formation'].apply(lambda x: np.where(formations == x)[0][0]),
                    cmap=cmap,
                    norm=norm,
                    edgecolors="black")

    # Plot the orientations if available
    if input_data.orientations is not None:
        ax.quiver(input_data.orientations.X, input_data.orientations.Z,
                  input_data.orientations.G_x, input_data.orientations.G_z,
                  input_data.orientations['formation'].apply(lambda x: np.where(formations == x)[0][0]),
                  angles='xy', scale_units='xy',
                  cmap=cmap,
                  norm=norm,
                  edgecolors="black",
                  linewidth=1)
    else:
        pass

    if geomodel_results is not None and show_results:

        # Reshape lithology block to resolution
        plot_block = np.round(geomodel_results.lith_block.reshape(geomodel_results.resolution), 0).astype(int)
        plot_block = plot_block

        # cut slice in the middle of the model
        image = plot_block[:, int(np.rint(input_data.resolution[1] / 2)), :].T

        # Create a discrete color map for the lithology block
        if input_data.faults is not None:
            n_faults = len(np.where(input_data.faults)[0])
            cmap2 = mcolors.ListedColormap(colors[n_faults:])
        else:
            cmap2 = mcolors.ListedColormap(colors)

        # Make sure boundaries are one more that colors
        # specify boundaries based on values in lithology block
        arr = np.unique(image)-0.5
        arr = np.append(arr, np.unique(arr)[-1]+1)
        norm2 = mcolors.BoundaryNorm(arr, ncolors=len(arr)-1)

        ax.imshow(
            image,
            origin='lower',
            zorder=-100,
            cmap=cmap2,
            norm=norm2,
            extent=(float(geomodel_results.extent[0]), float(geomodel_results.extent[1]),
                    float(geomodel_results.extent[4]), float(geomodel_results.extent[5]))
        )

        # TODO: Add contour solution here but this requires the scalar fields

    elif geomodel_results is None and show_results:
        print("Can not show results without results data.")
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

    plt.show()


def plot_3d(input_data: InputData, geomodel_results: GeomodelResults = None, show_results: bool = False, colors=None):
    """
    Plot the input data and results in 3D.

    Args:
        input_data (InputData): The input data for the geological model.
        geomodel_results (GeomodelResults): The results of the geological model.
        show_results (bool): Whether to show the results.
        colors (Optional(list)): List of colors to use for the different formations
    """
    # Suppress the specific warning about points not being a float type
    warnings.filterwarnings("ignore",
                            message="Points is not a float type. This can cause issues when transforming or applying "
                                    "filters. Casting to ``np.float32``. Disable this by passing "
                                    "``force_float=False``.")

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Set default colors
    if colors is None:
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd']

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
                         color=colors[i])

        # Add the orientations if available
        if input_data.orientations is not None:
            points = pv.PolyData(input_data.orientations[input_data.orientations.iloc[:, -1]
                                                         == current_element].iloc[:, :3].to_numpy().astype(np.float32))

            points['vectors'] = input_data.orientations[input_data.orientations.iloc[:, -1]
                                                        == current_element].iloc[:, 3:6].to_numpy().astype(np.float32)

            # Create an arrow source and a Glyph object and add to plotter
            arrow = pv.Arrow()
            glyphs = points.glyph(orient='vectors', scale=False, geom=arrow, factor=100)
            plotter.add_mesh(glyphs, color=colors[i])

    if geomodel_results is not None and show_results:
        # Add the surface meshes
        for i in range(len(formations)):
            plotter.add_mesh(
                pv.PolyData(geomodel_results.surface_meshes_vertices[i],
                            np.insert(geomodel_results.surface_meshes_edges[i], 0, 3, axis=1).ravel()),
                color=colors[i])
    if geomodel_results is None and show_results:
        print("Can not show results without results data.")
    else:
        pass

    # Set the bounds and grid of the plotter
    plotter.show_bounds(bounds=input_data.extent,
                        location="furthest",
                        grid=True)

    # Display the interactive plot
    plotter.show()
