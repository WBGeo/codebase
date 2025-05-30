import numpy as np
from core.object_components import InputData, GeomodelResults
from scipy.interpolate import RBFInterpolator
from core.utility.surface_mesh_extraction import marching_cubes_per_element, marching_cubes
from core.grids.grid_classes import RegularGrid


#%%
def rbf_interpolator(input_data: InputData,
                     kernel='linear',
                     smoothing=0,
                     neighbors=None,
                     epsilon=1) -> GeomodelResults:
    """
    Compute a model based on input data using RBF interpolation

    Args:
        input_data (InputData): The input data for the structural geological model.
        kernel (str): The kernel to use for the RBF interpolation. Default is 'linear'.
        epsilon (float): The epsilon value for the RBF interpolation. Default is 1.
        mask_surfaces (bool): Whether to mask surfaces. Default is True.

    Returns:
        resultsGeomodelResults: The results of the geological model.

    """
    # Test validity of input data for this interpolation
    # Check for faults
    if input_data.faults is not None:
        raise ValueError("Interpolator can not handle faults in the current state")

    # Check if there are at least to values per key in mapping object
    if not all(len(value) >= 2 for value in input_data.mapping_object.values()):
        raise ValueError("Interpolator requires at least two elements per group in mapping object")

    # Create a Grid instance
    grid = RegularGrid(input_data.extent, input_data.resolution)

    # Create mapping for replacing element names with ints
    unique_elements = sorted(set(element for elements in input_data.mapping_object.values() for element in elements))
    replacements = {element: i + 1 for i, element in enumerate(unique_elements)}

    # Separate data based on structural groups
    results = []
    scalar_fields = []
    for key, value in input_data.mapping_object.items():
        structural_group_df = input_data.surface_points[
            input_data.surface_points['formation'].isin(list(input_data.mapping_object[key]))]
        structural_group_df.loc[
            structural_group_df['formation'].isin(list(input_data.mapping_object[key])), 'formation'] = \
            structural_group_df['formation'].replace(replacements)

        # Create RBF interpolator
        rbfi = RBFInterpolator(np.stack((structural_group_df['X'],
                                         structural_group_df['Y'],
                                         structural_group_df['Z']), axis=1),
                               structural_group_df['formation'], kernel=kernel,
                               smoothing=smoothing,
                               neighbors=neighbors,  # Use None to use all points
                               epsilon=epsilon)

        # Interpolate the function on the grid
        rbf_res = rbfi(grid.grid_coordinates)

        # Reshape the result to resolution
        rbf_res = rbf_res.reshape(input_data.resolution)

        # Store original scalar fields
        scalar_fields.append(rbf_res.copy())

        # Save results, need to explicitly limit to maximum value as defined by replacement mapping
        max_value = max(replacements[element] for element in value)
        rbf_res[rbf_res > max_value] = max_value
        results.append(rbf_res.astype(int).copy())

    # TODO: These will both not work with faults
    # Get indices for each lithological group
    lith_group_indices = np.arange(len(input_data.mapping_object.keys()))
    # Get scalar value per element grouped same way and order as mapping object
    scalar_values = [[replacements[element] for element in input_data.mapping_object[key]] for key in
                     input_data.mapping_object.keys()]

    # Create scalar fields masks
    masks = []
    masks.append(np.ones_like(scalar_fields[0], dtype=bool))
    for i in range(len(lith_group_indices) - 1):
        mask = scalar_fields[i] <= scalar_values[i][-1]
        masks.append(mask)

    # Stack result based on stack
    combined_result = np.zeros_like(results[0])
    # Iterate over the results and masks arrays
    for i in range(len(results)):
        combined_result[masks[i]] = results[i][masks[i]]

    # replace all negative values with zeros in combined result
    combined_result[combined_result < 0] = 0

    # Reverse everything to match gempy, probably have to rewrite everything at some point
    max_val = int(np.max(combined_result))
    mapping = {i: max_val - i for i in range(max_val + 1)}

    # Apply the mapping to the array
    combined_result = np.vectorize(mapping.get)(combined_result)

    # Extract surface meshes
    mc_vertices_masked = []
    mc_edges_masked = []
    mc_vertices_all = []
    mc_edges_all = []
    mc_vertices_combined = []
    mc_edges_combined = []

    for idx in lith_group_indices:
        for i in range(len(scalar_values[idx])):
            # masked version
            vertices, edges = marching_cubes_per_element(scalar_fields[idx], scalar_values[idx][i],
                                                         grid.spacing, input_data.extent,
                                                         mask=masks[idx])
            mc_vertices_masked.append(vertices)
            mc_edges_masked.append(edges)

            # complete version going through unconformities
            vertices, edges = marching_cubes_per_element(scalar_fields[idx], scalar_values[idx][i],
                                                         grid.spacing, input_data.extent,
                                                         mask=None)
            mc_vertices_all.append(vertices)
            mc_edges_all.append(edges)

    # Extract surface meshes for structured meshing from combined block
    mc_vertices_combined, mc_edges_combined = marching_cubes(combined_result, unique_elements, grid.spacing,
                                                             input_data.extent)

    # Combine all meshes
    mc_vertices = [mc_vertices_masked, mc_vertices_all, mc_vertices_combined]
    mc_edges = [mc_edges_masked, mc_edges_all, mc_edges_combined]

    # Create a GeomodelResults instance
    results_instance = GeomodelResults(name=input_data.name,
                                       lith_block=combined_result.flatten(),
                                       surface_meshes_vertices=mc_vertices,
                                       surface_meshes_edges=mc_edges,
                                       grid=grid.grid_coordinates,
                                       extent=input_data.extent,
                                       resolution=input_data.resolution,
                                       mapping_object=input_data.mapping_object,
                                       scalar_fields=scalar_fields)

    return results_instance
