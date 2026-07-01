import numpy as np
from numpy.typing import NDArray
from typing import List


def create_surfaces_with_grids_for_bottom_and_top(min_x: float, max_x: float, min_y: float, max_y: float, min_z: float,
    max_z: float, n_gx: int, n_gy: int) -> NDArray[np.floating]:

    """
    Generate two 3D surface grids (bottom and top) over a specified x-y range at given z levels.

    Args:
        - min_x (float): Minimum x-coordinate of the grid.
        - max_x (float): Maximum x-coordinate of the grid.
        - min_y (float): Minimum y-coordinate of the grid.
        - max_y (float): Maximum y-coordinate of the grid.
        - min_z (float): z-coordinate for the bottom surface.
        - max_z (float): z-coordinate for the top surface.
        - n_gx (int): Number of grid points along the x-axis.
        - n_gy (int): Number of grid points along the y-axis.

    Returns:
        numpy.ndarray: A 2x(3 * n_gx * n_gy) array containing:
            - Row 0: Flattened x, y, z values for the bottom surface (at min_z).
            - Row 1: Flattened x, y, z values for the top surface (at max_z).
    """

    # Generate the bottom (min_z) and top (max_z) surfaces with grid
    x_vals: NDArray[np.floating] = np.linspace(min_x, max_x, n_gx)
    y_vals: NDArray[np.floating] = np.linspace(min_y, max_y, n_gy)
    x_grid: NDArray[np.floating]
    y_grid: NDArray[np.floating]

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Surface 1: at min_z (bottom plane with grid)
    surface_1_x: NDArray[np.floating] = x_grid.flatten()
    surface_1_y: NDArray[np.floating] = y_grid.flatten()
    surface_1_z: NDArray[np.floating] = np.full_like(surface_1_x, min_z)

    # Surface 2: at max_z (top plane with grid)
    surface_2_x: NDArray[np.floating] = x_grid.flatten()
    surface_2_y: NDArray[np.floating] = y_grid.flatten()
    surface_2_z: NDArray[np.floating] = np.full_like(surface_2_x, max_z)

    # Create a 2-row array for bottom and top surfaces
    bottom_top_surfaces: NDArray[np.floating] = np.zeros((2, 3 * n_gx * n_gy))

    # Store x, y, z values for bottom surface in the first row
    bottom_top_surfaces[0, :n_gx * n_gy] = surface_1_x
    bottom_top_surfaces[0, n_gx * n_gy:2 * n_gx * n_gy] = surface_1_y
    bottom_top_surfaces[0, 2 * n_gx * n_gy:] = surface_1_z

    # Store x, y, z values for top surface in the second row
    bottom_top_surfaces[1, :n_gx * n_gy] = surface_2_x
    bottom_top_surfaces[1, n_gx * n_gy:2 * n_gx * n_gy] = surface_2_y
    bottom_top_surfaces[1, 2 * n_gx * n_gy:] = surface_2_z

    return bottom_top_surfaces


def create_intermediate_layers(bottom_top_surfaces: NDArray[np.floating], output_array: NDArray[np.floating], refinement_data: List[int],
    n_gx: int, n_gy: int) -> NDArray[np.floating]:
    """
    Creates intermediate layers of points between the bottom and top surfaces based on refinement data.

    Args:
        - bottom_top_surfaces (np.array): Array containing two rows. The first row is the bottom surface, and
                                        the second row is the top surface.
        - output_array (np.array): Array of shape (n_surfaces, 3 * n_gx * n_gy), containing multiple surfaces.
        - refinement_data (list): List of refinement data to scale z-differences between layers.
        - n_gx (int): Grid size in x direction.
        - n_gy (int): Grid size in y direction.

    Returns:
        np.array: Updated array with intermediate layers between the bottom and top surfaces.
    """

    # Subtract one from each refinement level so that the number of elements matches refinement_data
    refinement_data = [v - 1 for v in refinement_data]

    # Extract bottom surface x, y, z values
    x_bottom: NDArray[np.floating] = bottom_top_surfaces[0, :n_gx * n_gy]
    y_bottom: NDArray[np.floating] = bottom_top_surfaces[0, n_gx * n_gy: 2 * n_gx * n_gy]
    z_bottom: NDArray[np.floating] = bottom_top_surfaces[0, 2 * n_gx * n_gy:]

    # Extract top surface z values
    z_top: NDArray[np.floating] = bottom_top_surfaces[1, 2 * n_gx * n_gy:]

    # Check that the number of refinement levels is consistent with the number of surfaces
    n_surfaces: int = len(output_array)

    if len(refinement_data) != (n_surfaces + 1):
      raise ValueError(f"\033[91mError: Number of refinement input_data should be number of surfaces + 1. You have given {len(refinement_data)} instead of {len(output_array)+1}.\033[0m")


    # Initialize the updated output array. The size is (sum of number of refinements +number of layers + 2(bottom+top)* 4*n_gx*n_gy
    sum_rf: int =sum(refinement_data)+len(output_array)+2
    layer_size: int  = 4 * n_gx * n_gy
    updated_output_array: NDArray[np.floating] = np.zeros((sum_rf, layer_size))

    # Add the bottom surface to the updated output
    bottom_layer: NDArray[np.floating] = np.concatenate([x_bottom, y_bottom, z_bottom])

    # Create zero columns to assign surface_id
    zero_column: NDArray[np.floating] = np.zeros((n_gx * n_gy))

    # Concatenate the zero column to the bottom_layer
    updated_output_with_zeroes: NDArray[np.floating] = np.concatenate((bottom_layer, zero_column))

    # Append to the updated_output_array
    updated_output_array[0, :] = updated_output_with_zeroes  # Add to the first row

    # Handle the bottom surface to the first layer
    z_output_first_layer: NDArray[np.floating] = output_array[0, 2 * n_gx * n_gy:]

    # Size of divisions between bottom and the first intermediate layer
    z_step: NDArray[np.floating] = -(z_bottom - z_output_first_layer) / (refinement_data[0] + 1)

    for j in range(refinement_data[0]):

        z_values: NDArray[np.floating] = z_bottom + (j + 1) * z_step

        layer: NDArray[np.floating] = np.concatenate([x_bottom, y_bottom, z_values, zero_column])
        updated_output_array[j+1, :] = layer


    # Handle intermediate layers in output_array
    count: int =0

    current_layer_index: int = refinement_data[0]

    for i in range(len(output_array)-1):
        current_layer_index += 1
        count += 1

        Id_column: NDArray[np.integer] = np.full((n_gx * n_gy), count)

        z_output_current_layer: NDArray[np.integer] = output_array[i, 2 * n_gx * n_gy:]

        # Add current layer to the output_array
        current_layer: NDArray[np.integer] = np.concatenate([x_bottom, y_bottom, z_output_current_layer, Id_column])

        updated_output_array[current_layer_index, :] = current_layer
        # Calculate the divisions

        z_output_next_layer: NDArray[np.integer] = output_array[i + 1, 2 * n_gx * n_gy:]
        z_step = -(z_output_current_layer - z_output_next_layer) / (refinement_data[i+1] + 1)

        for j in range(refinement_data[i+1]):

            current_layer_index += 1

            z_values = z_output_current_layer + (j + 1) * z_step

            layer = np.concatenate([x_bottom, y_bottom, z_values, Id_column])

            updated_output_array[current_layer_index, :] = layer


    # Handle the last layer to the top surface
    z_output_last_layer: NDArray[np.integer] = output_array[-1, 2 * n_gx * n_gy:]
    z_step = -(z_output_last_layer - z_top) / (refinement_data[-1] + 1)

    # Add current layer to the output_array
    current_layer_index += 1
    count += 1

    Id_column = np.full((n_gx * n_gy), count)
    current_layer=np.concatenate([x_bottom, y_bottom,z_output_last_layer, Id_column])
    updated_output_array[current_layer_index, :]=current_layer

    for j in range(refinement_data[-1]):
        current_layer_index += 1
        z_values = z_output_last_layer + (j + 1) * z_step

        layer = np.concatenate([x_bottom, y_bottom, z_values, Id_column ])
        updated_output_array[current_layer_index, :] = layer


    # Add the top surface to the updated output
    current_layer_index += 1

    top_layer: NDArray[np.integer] = np.concatenate([x_bottom, y_bottom, z_top, Id_column ])

    updated_output_array[current_layer_index, :] = top_layer

    return updated_output_array

