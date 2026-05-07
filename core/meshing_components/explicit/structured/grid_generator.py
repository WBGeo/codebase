import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from core.object_components import StructuralModelResults



def create_surface_grid(results_instance: StructuralModelResults, extent: List[float]) -> Tuple[Dict[str, NDArray[np.float64]], int, int]:
    """
    Find the surface with minimum vertices, sort the surface vertices into a grid,
    and then interpolate the z values for other surfaces based on the sorted grid.

    Args:
        results_instance (GeomodelResults): Geological model results containing surfaces.
        extent (list of float): extent of mesh

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary with interpolated grids for each surface.
                       Each entry has a key of the form 'surface_{i}' and a value of the grid array.
               - int: Number of grid points in the x direction (n_gx).
               - int: Number of grid points in the y direction (n_gy).
    """

    min_x, max_x, min_y, max_y, min_z, max_z = extent

    # Find the surface with the minimum number of vertices, index 2 for mesh type
    min_vertices_surface_index: int = np.argmin([len(vertices) for vertices in results_instance.surface_meshes_vertices[2]])
    min_vertices_surface: NDArray[np.float64] = results_instance.surface_meshes_vertices[2][min_vertices_surface_index]

    # Extract the x, y, z coordinates of the surface with minimum vertices
    x_min: NDArray[np.float64] = min_vertices_surface[:, 0]
    y_min: NDArray[np.float64] = min_vertices_surface[:, 1]
    z_min: NDArray[np.float64] = min_vertices_surface[:, 2]

    # Ensure the grid fully covers the model extent in both x and y
    corners: List[Tuple[float, float, float]]  = [
        (min_x, min_y, min_z),
        (min_x, max_y, min_z),
        (max_x, min_y, max_z),
        (max_x, max_y, max_z)
    ]

    for cx, cy, cz in corners:
        if not ((x_min == cx).any() and (y_min == cy).any()):
            x_min = np.concatenate([x_min, [cx]])
            y_min = np.concatenate([y_min, [cy]])
            z_min = np.concatenate([z_min, [cz]])


    # Sort the vertices by x and y to create a grid-like structure
    sorted_indices: NDArray[np.int64] = np.lexsort((x_min, y_min))
    x_min_sorted: NDArray[np.float64] = x_min[sorted_indices]
    y_min_sorted: NDArray[np.float64] = y_min[sorted_indices]

    # Reshape x and y to form a grid
    unique_x: NDArray[np.float64] = np.unique(x_min_sorted)
    unique_y: NDArray[np.float64] = np.unique(y_min_sorted)
    grid_x: NDArray[np.float64]
    grid_y: NDArray[np.float64]
    grid_x, grid_y = np.meshgrid(unique_x, unique_y)

    # Number of grids in x and y directions
    n_gx:int  = len(unique_x)
    n_gy:int = len(unique_y)

    # Dictionary to store interpolated grids
    interpolated_surfaces: Dict[str, NDArray[np.float64]] = {}

    # Perform interpolation for each surface
    for i, vertices in enumerate(results_instance.surface_meshes_vertices[2]):
        # Get the x, y, z coordinates of the current surface
        x: NDArray[np.float64] = vertices[:, 0]
        y: NDArray[np.float64] = vertices[:, 1]
        z: NDArray[np.float64] = vertices[:, 2]

        # Remove duplicates to avoid interpolation issues
        df: pd.DataFrame = pd.DataFrame({'x': x, 'y': y, 'z': z}).drop_duplicates()
        x_cleaned: NDArray[np.float64] = df['x'].values
        y_cleaned: NDArray[np.float64] = df['y'].values
        z_cleaned: NDArray[np.float64] = df['z'].values

        # Define the radial basis function interpolator
        rbf: Rbf = Rbf(x_cleaned, y_cleaned, z_cleaned, function='multiquadric', epsilon=2, smooth=1e-5)

        # Perform interpolation on the grid
        z_interpolated: np.ndarray  = rbf(grid_x, grid_y)
        z_interpolated = np.round(z_interpolated)

        # Combine the grid with interpolated z values
        interpolated_grid: NDArray[np.float64]= np.column_stack((grid_x.flatten(), grid_y.flatten(), z_interpolated.flatten()))

        # Store the interpolated grid
        interpolated_surfaces[f"surface_{i}"] = interpolated_grid

    return interpolated_surfaces, n_gx, n_gy



def sort_points_by_x_y(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Sorts the points in each DataFrame by x and then by y.
    Args:
        dataframes (list of pandas.DataFrame):
            A list of DataFrames, where each DataFrame represents a surface grid
            and contains the columns ["X", "Y", "Z"] corresponding to point
            coordinates.

    Returns:
        list of pandas.DataFrame:
            The input list of DataFrames, where each DataFrame is sorted in
            ascending order by "X" and then by "Y", with indices reset.
    """
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].sort_values(by=["X", "Y"]).reset_index(drop=True)
    return dataframes


def sort_surfaces_by_z(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Sorts the list of DataFrames by the average z-value of each surface.
        Args:
        dataframes (list of pandas.DataFrame):
            A list of DataFrames, where each DataFrame represents a surface grid
            and contains the columns ["X", "Y", "Z"] corresponding to point
            coordinates.

    Returns:
        list of pandas.DataFrame:
            The input list of DataFrames, where each DataFrame is sorted in
            ascending order by average "Z", with indices reset.
    """
    avg_z_values: List[float] = [df["Z"].mean() for df in dataframes]
    sorted_indices: NDArray[np.int64] = np.argsort(avg_z_values)
    sorted_dataframes: List[pd.DataFrame] = [dataframes[i] for i in sorted_indices]
    return sorted_dataframes


def store_points_in_array(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
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
    n_gx: int = len(dataframes[0]["X"].unique())
    n_gy: int = len(dataframes[0]["Y"].unique())
    n_gz: int = len(dataframes[0]["Z"].unique())

    # Ensure grids are consistent across surfaces
    for df in dataframes:
        if len(df["X"].unique()) != n_gx or len(df["Y"].unique()) != n_gy:
            raise ValueError("Inconsistent grid dimensions across surfaces.")

    # Number of surfaces
    n_surfaces: int = len(dataframes)
    n_points_per_surface: int = n_gx * n_gy

    # Initialize the output array
    output_array: NDArray[np.float64] = np.zeros((n_surfaces, n_points_per_surface * 3))

    for i, df in enumerate(dataframes):
        # Sort the DataFrame by y and x to ensure consistency
        df_sorted : pd.DataFrame = df.sort_values(by=["Y", "X"])

        # Extract sorted x, y, z values
        x_vals: NDArray[np.float64]= df_sorted["X"].values
        y_vals: NDArray[np.float64] = df_sorted["Y"].values
        z_vals: NDArray[np.float64] = df_sorted["Z"].values

        # Store x, y, z values consecutively in the output array
        output_array[i, 0:n_points_per_surface] = x_vals
        output_array[i, n_points_per_surface:2 * n_points_per_surface] = y_vals
        output_array[i, 2 * n_points_per_surface:3 * n_points_per_surface] = z_vals

    return output_array, n_gx, n_gy, n_gz



