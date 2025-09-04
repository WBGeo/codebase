import numpy as np
import pandas as pd
from core.meshing_components.explicit.structured.grid_generator import create_surface_grid, sort_points_by_x_y
from core.meshing_components.explicit.structured.grid_generator import sort_surfaces_by_z, store_points_in_array
from core.meshing_components.explicit.structured.store_grid_data import create_surfaces_with_grids_for_bottom_and_top
from core.meshing_components.explicit.structured.store_grid_data import create_intermediate_layers
from core.meshing_components.explicit.structured.node_element_generator import adjust_z_values, \
    create_hexahedral_elements_with_nodes
from core.object_components import MeshResults
from typing import Tuple
from core.object_components import GeomodelResults
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
import typing


RefinementData = typing.Annotated[Tuple[int, ...], AnnotatedScriptType(name='refinement_data', color='aqua', identifier='mesh::RefinementData')]


# Register this function as a component
@wbgeo_component(description='Provides structured mesh',
                 title='Creates Structured Mesh',  # The title shown in the GUI
                 color='#800000',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Mesh',
                 identifier='create_structured_mesh_data',  # a unique identifier
                 return_name='Mesh',  # the name for the returned-port
                 )  # inputs are handled via the method signature


def create_structured_mesh_data(geomodel_result: GeomodelResults, refinement_data: RefinementData =(25,21,16,5,6),
                                z_threshold: float =0.1, tolerance: float =1) -> MeshResults:
    """
    Generates a geological mesh and returns a MeshData object.

    Args:
        results_test (object): Object containing data to create the grid.
        refinement_data (list): list of refinement values.
        z_threshold (float): Threshold for Z-value adjustment.
        tolerance (float): Distance tolerance for Z-value adjustment.

    Returns:
        MeshResults: An instance of the MeshResults class.
    """


    # Extract extent values
    min_x, max_x, min_y, max_y, min_z, max_z = geomodel_result.extent

    # Create interpolated surfaces dictionary
    interpolated_surfaces, n_gx, n_gy = create_surface_grid(geomodel_result)

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
    # refinement_data = read_refinement_file(refine_file_path, len(dataframes_list))

    # Create points between surfaces
    updated_output = create_intermediate_layers(
        bottom_top_surfaces, output_array, refinement_data, n_gx, n_gy
    )

    # Adjust Z values based on distance tolerance
    adjusted_array = adjust_z_values(updated_output, n_gx, n_gy, z_threshold, tolerance)

    # Create elements and assign surface IDs to them
    elements, nodes = create_hexahedral_elements_with_nodes(adjusted_array, n_gx, n_gy)
    # Since in implicit mesh the numbering is reversed, here I also reverse them
    # Find unique values in the last column
    unique_values = np.unique(elements[:, -1])
    # Create a mapping: max value → 0, min value → max, etc.
    mapping = {val: i for i, val in enumerate(unique_values[::-1])}
    # Apply the mapping to the last column
    elements[:, -1] = np.vectorize(mapping.get)(elements[:, -1])

    # Create and return a MeshData instance
    return MeshResults(elements=elements,
                       nodes=nodes,
                       )
