import numpy as np
from core.object_components import InputData, GeomodelResults
from scipy.interpolate import RBFInterpolator
from core.utility.surface_mesh_extraction import marching_cubes
from core.grids.grid_classes import RegularGrid


#%%
def rbf_interpolator(input_data: InputData, kernel='linear', epsilon=1):
    """
    Compute a model based on input data using RBF interpolation

    Args:
        input_data (InputData): The input data for the structural geological model.
        kernel (str): The kernel to use for the RBF interpolation. Default is 'linear'.
        epsilon (float): The epsilon value for the RBF interpolation. Default is 1.

    Returns:
        resultsGeomodelResults: The results of the geological model.

    """
    # TODO: This does not consider faults
    # TODO: This does require two elements per group to make sense

    # Create a Grid instance
    grid = RegularGrid(input_data.extent, input_data.resolution)

    # Create mapping for replacing element names with ints
    unique_elements = sorted(set(element for elements in input_data.mapping_object.values() for element in elements))
    replacements = {element: i + 1 for i, element in enumerate(unique_elements)}

    # Separate data based on structural groups
    results = []
    results_scalars = []
    masks = []
    for key, value in input_data.mapping_object.items():
        structural_group_df = input_data.surface_points[
            input_data.surface_points['formation'].isin(list(input_data.mapping_object[key]))]
        structural_group_df.loc[
            structural_group_df['formation'].isin(list(input_data.mapping_object[key])), 'formation'] = \
            structural_group_df['formation'].replace(replacements)

        # TODO: Figure out good default settings and what other kernels are reasonable
        rbfi = RBFInterpolator(np.stack((structural_group_df['X'],
                                         structural_group_df['Y'],
                                         structural_group_df['Z']), axis=1),
                               structural_group_df['formation'], kernel=kernel,
                               epsilon=epsilon)

        # Interpolate the function on the grid
        rbf_res = rbfi(grid.grid_coordinates)

        # Reshape the result to resolution
        rbf_res = rbf_res.reshape(input_data.resolution).T

        # Save results, need to explicitly limit to maximum value as defined by replacement mapping
        max_value = max(replacements[element] for element in value)
        rbf_res[rbf_res > max_value] = max_value
        results.append(rbf_res.astype(int))
        results_scalars.append(rbf_res)

        # Create mask for values below the lowest integer value for stacking
        min_value = min(replacements[element] for element in value)
        mask = rbf_res >= min_value
        masks.append(mask)

    # Stack result based on stack
    combined_result = np.zeros(results[0].shape)

    # Iterate over the results and masks arrays
    for i in range(len(results) - 1, -1, -1):
        combined_result[masks[i]] = results[i][masks[i]]

    combined_result = combined_result.T

    # Reverse everything to match gempy, probably have to rewrite everything at some point
    max_val = int(np.max(combined_result))
    mapping = {i: max_val - i for i in range(max_val + 1)}

    # Apply the mapping to the array
    combined_result = np.vectorize(mapping.get)(combined_result)

    # Extract the surface meshes using marching cubes, does not consider faults as not possible atm
    mc_vertices, mc_edges = marching_cubes(combined_result, unique_elements, grid.spacing, input_data.extent)

    # Create a GeomodelResults instance
    results_instance = GeomodelResults(name=input_data.name,
                                       lith_block=combined_result.flatten(),
                                       surface_meshes_vertices=mc_vertices,
                                       surface_meshes_edges=mc_edges,
                                       grid=grid.grid_coordinates,
                                       extent=input_data.extent,
                                       resolution=input_data.resolution,
                                       mapping_object=input_data.mapping_object)

    return results_instance
