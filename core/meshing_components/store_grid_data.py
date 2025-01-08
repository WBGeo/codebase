import numpy as np
import pandas as pd
import os

def create_surfaces_with_grids_for_bottom_and_top(min_x, max_x, min_y, max_y, min_z, max_z, n_gx, n_gy):
    # Generate the bottom (min_z) and top (max_z) surfaces with grid
    x_vals = np.linspace(min_x, max_x, n_gx)
    y_vals = np.linspace(min_y, max_y, n_gy)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Surface 1: at min_z (bottom plane with grid)
    surface_1_x = x_grid.flatten()
    surface_1_y = y_grid.flatten()
    surface_1_z = np.full_like(surface_1_x, min_z)

    # Surface 2: at max_z (top plane with grid)
    surface_2_x = x_grid.flatten()
    surface_2_y = y_grid.flatten()
    surface_2_z = np.full_like(surface_2_x, max_z)

    # Create a 2-row array for bottom and top surfaces
    bottom_top_surfaces = np.zeros((2, 3 * n_gx * n_gy))

    # Store x, y, z values for bottom surface in the first row
    bottom_top_surfaces[0, :n_gx * n_gy] = surface_1_x
    bottom_top_surfaces[0, n_gx * n_gy:2 * n_gx * n_gy] = surface_1_y
    bottom_top_surfaces[0, 2 * n_gx * n_gy:] = surface_1_z

    # Store x, y, z values for top surface in the second row
    bottom_top_surfaces[1, :n_gx * n_gy] = surface_2_x
    bottom_top_surfaces[1, n_gx * n_gy:2 * n_gx * n_gy] = surface_2_y
    bottom_top_surfaces[1, 2 * n_gx * n_gy:] = surface_2_z

    return bottom_top_surfaces


def read_refinement_file(file_path, expected_surfaces):
    """
    Reads refinement data from a file and checks if the number of data entries matches the expected number of surfaces + 1.

    Args:
        file_path (str): The path to the file containing the data.
        expected_surfaces (int): The expected number of surfaces.

    Returns:
        List of refinement data if valid, else prints an error.
    """
    if not os.path.exists(file_path):
        print("Error: Please provide a refinement file that contains the number of divisions between surfaces.")
        return None

    try:
        with open(file_path, 'r') as file:
            data = file.readlines()

        # Check if the number of data entries is correct
        if len(data) != expected_surfaces + 1:
            print(f"Error: Expected {expected_surfaces + 1} data entries, but got {len(data)}.")
            return None

        # Convert the data to a list (assuming each line represents a surface's data)
        refinement_data = [int(line.strip()) for line in data]

        return refinement_data

    except Exception as e:
        print(f"Error reading the file: {e}")
        return None



def create_intermediate_layers(bottom_top_surfaces, output_array, refinement_data, n_gx, n_gy):
    """
    Creates intermediate layers of points between the bottom and top surfaces based on refinement data.

    Args:
        bottom_top_surfaces (np.array): Array containing two rows. The first row is the bottom surface, and
                                        the second row is the top surface.
        output_array (np.array): Array of shape (n_surfaces, 3 * n_gx * n_gy), containing multiple surfaces.
        refinement_data (list): List of refinement data to scale z-differences between layers.
        n_gx (int): Grid size in x direction.
        n_gy (int): Grid size in y direction.

    Returns:
        np.array: Updated array with intermediate layers between the bottom and top surfaces.
    """
    # Extract bottom surface x, y, z values
    x_bottom = bottom_top_surfaces[0, :n_gx * n_gy]
    y_bottom = bottom_top_surfaces[0, n_gx * n_gy: 2 * n_gx * n_gy]
    z_bottom = bottom_top_surfaces[0, 2 * n_gx * n_gy:]
    # Extract top surface z values
    z_top = bottom_top_surfaces[1, 2 * n_gx * n_gy:]

    # Initialize the updated output array. The size is (sum of number of refinements +number of layers + 2(bottom+top)* 4*n_gx*n_gy
    sum_rf=sum(refinement_data)+len(output_array)+2
    layer_size = 4 * n_gx * n_gy
    updated_output_array = np.zeros((sum_rf, layer_size))

    # Add the bottom surface to the updated output
    bottom_layer = np.concatenate([x_bottom, y_bottom, z_bottom])
    # Create zero columns to assign surface_id
    zero_column = np.zeros((n_gx * n_gy))
    # Concatenate the zero column to the bottom_layer
    updated_output_with_zeroes = np.concatenate((bottom_layer, zero_column))

    # Append to the updated_output_array
    updated_output_array[0, :] = updated_output_with_zeroes  # Add to the first row

    # Handle the bottom surface to the first layer
    z_output_first_layer = output_array[0, 2 * n_gx * n_gy:]
    # Size of devisions between bottom and the first intermediate layer
    z_step = -(z_bottom - z_output_first_layer) / (refinement_data[0] + 1)
    for j in range(refinement_data[0]):
        z_values = z_bottom + (j + 1) * z_step
        layer = np.concatenate([x_bottom, y_bottom, z_values, zero_column])
        updated_output_array[j+1, :] = layer


    # Handle intermediate layers in output_array
    count=0
    current_layer_index = refinement_data[0]
    for i in range(len(output_array)-1):
        current_layer_index += 1
        count += 1
        Id_column = np.full((n_gx * n_gy), count)
        z_output_current_layer = output_array[i, 2 * n_gx * n_gy:]
        # Add current layer to the output_array
        current_layer=np.concatenate([x_bottom, y_bottom, z_output_current_layer, Id_column])
        updated_output_array[current_layer_index, :]=current_layer
        # Calculate the devisions
        z_output_next_layer = output_array[i + 1, 2 * n_gx * n_gy:]
        z_step = -(z_output_current_layer - z_output_next_layer) / (refinement_data[i+1] + 1)
        for j in range(refinement_data[i+1]):
            current_layer_index += 1
            z_values = z_output_current_layer + (j + 1) * z_step
            layer = np.concatenate([x_bottom, y_bottom, z_values, Id_column])
            updated_output_array[current_layer_index, :] = layer


    # Handle the last layer to the top surface
    z_output_last_layer = output_array[-1, 2 * n_gx * n_gy:]
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
    top_layer = np.concatenate([x_bottom, y_bottom, z_top, Id_column ])
    updated_output_array[current_layer_index, :] = top_layer

    return updated_output_array

