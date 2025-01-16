import numpy as np
import pandas as pd
from core.meshing_components.mesh_generation.grid_generator import create_surface_grid, sort_points_by_x_y
from core.meshing_components.mesh_generation.grid_generator import sort_surfaces_by_z, store_points_in_array
from core.meshing_components.mesh_generation.store_grid_data import create_surfaces_with_grids_for_bottom_and_top
from core.meshing_components.mesh_generation.store_grid_data import read_refinement_file, create_intermediate_layers
from core.meshing_components.mesh_generation.node_element_generator import adjust_z_values, create_hexahedral_elements_with_nodes
from core.object_components import MeshData

def create_mesh_data(extent, results_test, refine_file_path, z_threshold=0.1, tolerance=1):
    """
    Generates a geological mesh and returns a MeshData object.

    Args:
        extent (tuple): The spatial extent of the model.
        results_test (object): Object containing data to create the grid.
        refine_file_path (str): Path to the refinement file.
        z_threshold (float): Threshold for Z-value adjustment.
        tolerance (float): Distance tolerance for Z-value adjustment.

    Returns:
        MeshData: An instance of the MeshData class.
    """
    # Extract extent values
    min_x, max_x, min_y, max_y, min_z, max_z = extent

    # Create interpolated surfaces dictionary
    interpolated_surfaces, n_gx, n_gy = create_surface_grid(results_test)

    # Convert dictionary values (interpolated grids) to list of sorted DataFrames
    dataframes_list = [pd.DataFrame(grid, columns=["X", "Y", "Z"]) for grid in interpolated_surfaces.values()]
    sorted_dataframes = sort_points_by_x_y(dataframes_list)
    sorted_surfaces = sort_surfaces_by_z(sorted_dataframes)

    # Store points in array
    output_array = store_points_in_array(sorted_surfaces)
    output_array = np.array(output_array[0])

    # Create bottom and top surfaces
    bottom_top_surfaces = create_surfaces_with_grids_for_bottom_and_top(
        min_x, max_x, min_y, max_y, min_z, max_z, n_gx, n_gy
    )

    # Read refinement file
    refinement_data = read_refinement_file(refine_file_path, len(dataframes_list))

    # Create points between surfaces
    updated_output = create_intermediate_layers(
        bottom_top_surfaces, output_array, refinement_data, n_gx, n_gy
    )

    # Adjust Z values based on distance tolerance
    adjusted_array = adjust_z_values(updated_output, n_gx, n_gy, z_threshold, tolerance)

    # Create elements and assign surface IDs to them
    elements, nodes = create_hexahedral_elements_with_nodes(adjusted_array, n_gx, n_gy)

    # Create and return a MeshData instance
    return MeshData(
        elements=elements,
        nodes=nodes,
        n_gx=n_gx,
        n_gy=n_gy
    )

