import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

def create_surface_grid(results_instance):
    """
    Find the surface with minimum vertices, sort the surface vertices into a grid,
    and then interpolate the z values for other surfaces based on the sorted grid.
    Also ensures the extent values from the results_instance are included in the grid.

    Args:
        results_instance (GeomodelResults): Geological model results containing surfaces.

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary with interpolated grids for each surface.
                       Each entry has a key of the form 'surface_{i}' and a value of the grid array.
               - int: Number of grid points in the x direction (n_gx).
               - int: Number of grid points in the y direction (n_gy).
    """
    # Get the extent from the results instance
    extent = results_instance.extent
    min_x, max_x, min_y, max_y, min_z, max_z = extent

    # Find the surface with the minimum number of vertices, index 2 for mesh type
    min_vertices_surface_index = np.argmin([len(vertices) for vertices in results_instance.surface_meshes_vertices[2]])
    print(min_vertices_surface_index)
    min_vertices_surface = results_instance.surface_meshes_vertices[2][min_vertices_surface_index]

    # Extract the x, y, z coordinates of the surface with minimum vertices
    x_min = min_vertices_surface[:, 0]
    y_min = min_vertices_surface[:, 1]
    z_min = min_vertices_surface[:, 2]

    # Extend the grid to include the full extent if necessary
    if max_x not in x_min:
        x_min = np.concatenate([x_min, [max_x]])
        y_min = np.concatenate([y_min, [max_y]])
        z_min = np.concatenate([z_min, [max_z]])
    if min_x not in x_min:
        x_min = np.concatenate([x_min, [min_x]])
        y_min = np.concatenate([y_min, [min_y]])
        z_min = np.concatenate([z_min, [min_z]])

    # Sort the vertices by x and y to create a grid-like structure
    sorted_indices = np.lexsort((x_min, y_min))  # Sort first by y, then by x
    x_min_sorted = x_min[sorted_indices]
    y_min_sorted = y_min[sorted_indices]

    # Reshape x and y to form a grid
    unique_x = np.unique(x_min_sorted)
    unique_y = np.unique(y_min_sorted)
    grid_x, grid_y = np.meshgrid(unique_x, unique_y)

    # Number of grids in x and y directions
    n_gx = len(unique_x)
    n_gy = len(unique_y)

    # Dictionary to store interpolated grids
    interpolated_surfaces = {}

    # Perform interpolation for each surface
    for i, vertices in enumerate(results_instance.surface_meshes_vertices[2]):
        # Get the x, y, z coordinates of the current surface
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        # Remove duplicates to avoid interpolation issues
        df = pd.DataFrame({'x': x, 'y': y, 'z': z}).drop_duplicates()
        x_cleaned = df['x'].values
        y_cleaned = df['y'].values
        z_cleaned = df['z'].values

        # Define the radial basis function interpolator
        rbf = Rbf(x_cleaned, y_cleaned, z_cleaned, function='multiquadric', epsilon=2, smooth=1e-5)

        # Perform interpolation on the grid
        z_interpolated = rbf(grid_x, grid_y)
        z_interpolated = np.round(z_interpolated)

        # Combine the grid with interpolated z values
        interpolated_grid = np.column_stack((grid_x.flatten(), grid_y.flatten(), z_interpolated.flatten()))

        # Store the interpolated grid
        interpolated_surfaces[f"surface_{i}"] = interpolated_grid

    return interpolated_surfaces, n_gx, n_gy



def sort_points_by_x_y(dataframes):
    """
    Sorts the points in each DataFrame by x and then by y.
    """
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].sort_values(by=["X", "Y"]).reset_index(drop=True)
    return dataframes

def sort_surfaces_by_z(dataframes):
    """
    Sorts the list of DataFrames by the average z-value of each surface.
    """
    avg_z_values = [df["Z"].mean() for df in dataframes]
    sorted_indices = np.argsort(avg_z_values)
    sorted_dataframes = [dataframes[i] for i in sorted_indices]
    return sorted_dataframes


def store_points_in_array(dataframes):
    """
    Stores x, y, z values of each surface into a NumPy array with the following format:
    - First n_gx * n_gy entries: x values of the surface
    - Next n_gx * n_gy entries: y values of the surface
    - Last n_gx * n_gy entries: z values of the surface
    Repeated for each surface in the dataset.

    Args:
    - dataframes: A list of pandas DataFrames, each representing a surface, with columns 'X', 'Y', and 'Z'.

    Returns:
    - output_array: A NumPy array where each row corresponds to a surface.
    - n_gx: Number of unique x values (grid size in x direction).
    - n_gy: Number of unique y values (grid size in y direction).
    """
    n_gx = len(dataframes[0]["X"].unique())
    n_gy = len(dataframes[0]["Y"].unique())
    n_gz = len(dataframes[0]["Z"].unique())

    # Ensure grids are consistent across surfaces
    for df in dataframes:
        if len(df["X"].unique()) != n_gx or len(df["Y"].unique()) != n_gy:
            raise ValueError("Inconsistent grid dimensions across surfaces.")

    # Number of surfaces
    n_surfaces = len(dataframes)
    n_points_per_surface = n_gx * n_gy

    # Initialize the output array
    output_array = np.zeros((n_surfaces, n_points_per_surface * 3))

    for i, df in enumerate(dataframes):
        # Sort the DataFrame by y and x to ensure consistency
        df_sorted = df.sort_values(by=["Y", "X"])

        # Extract sorted x, y, z values
        x_vals = df_sorted["X"].values
        y_vals = df_sorted["Y"].values
        z_vals = df_sorted["Z"].values

        # Store x, y, z values consecutively in the output array
        output_array[i, 0:n_points_per_surface] = x_vals
        output_array[i, n_points_per_surface:2 * n_points_per_surface] = y_vals
        output_array[i, 2 * n_points_per_surface:3 * n_points_per_surface] = z_vals

    return output_array, n_gx, n_gy, n_gz



