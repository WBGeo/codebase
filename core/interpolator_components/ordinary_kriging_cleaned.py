import numpy as np
from core.object_components import InputData, GeomodelResults
from pykrige.ok3d import OrdinaryKriging3D
from core.utility.surface_mesh_extraction import marching_cubes_per_element
from skimage import measure
from core.grids.grid_classes import RegularGrid


#%%
def ordinary_kriging_interpolator_cleaned(input_data: InputData,
                                          var_model="gaussian",
                                          var_sill=1,
                                          var_range=500,
                                          var_nugget=0,
                                          anisotropy_scaling_z=0.3,
                                          mask_surfaces=True) -> GeomodelResults:
    """
    Compute a model based on input data using kriging interpolation

    Args:
        input_data (InputData): The input data for the geological model.
        var_model (str): The variogram model to use. Default is 'gaussian'.
        var_sill (float): The sill of the variogram. Default is 1.
        var_range (float): The range of the variogram. Default is 500.
        var_nugget (float): The nugget of the variogram. Default is 0.
        anisotropy_scaling_z (float): The scaling factor for the z-axis. Default is 0.3.
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
    results_scalars = []
    masks = []
    scalar_fields = []
    for key, value in input_data.mapping_object.items():
        structural_group_df = input_data.surface_points[
            input_data.surface_points['formation'].isin(list(input_data.mapping_object[key]))]
        structural_group_df.loc[
            structural_group_df['formation'].isin(list(input_data.mapping_object[key])), 'formation'] = \
            structural_group_df['formation'].replace(replacements)

        # perform Ordinary Kriging per structural group
        ok3d = OrdinaryKriging3D(
            structural_group_df['X'], structural_group_df['Y'], structural_group_df['Z'],
            structural_group_df['formation'], variogram_model=var_model,
            variogram_parameters=[var_sill, var_range, var_nugget],
            anisotropy_scaling_z=anisotropy_scaling_z
        )
        k3d1, ss3d = ok3d.execute("grid",
                                  grid.gridx,
                                  grid.gridy,
                                  grid.gridz)

        # Store original scalar fields
        scalar_fields.append(k3d1.T.copy())

        # Save results, need to explicitly limit to maximum value as defined by replacement mapping
        max_value = max(replacements[element] for element in value)
        k3d1[k3d1 > max_value] = max_value
        results.append(k3d1.astype(int).T.copy())

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

    print(results[0].shape)
    print(masks[0].shape)

    # TODO: This doest not give the correct result. The block is weirdly inverted
    # Stack result based on stack
    combined_result = np.zeros_like(results[0])
    # Iterate over the results and masks arrays
    for i in range(len(results) - 1, -1, -1):
        combined_result[masks[i]] = results[i][masks[i]]

    # Reverse everything to match gempy, probably have to rewrite everything at some point
    max_val = int(np.max(combined_result))
    mapping = {i: max_val - i for i in range(max_val + 1)}
    # Apply the mapping to the array
    combined_result = np.vectorize(mapping.get)(combined_result)

    # Extract surface meshes
    mc_vertices = []
    mc_edges = []
    for idx in lith_group_indices:
        for i in range(len(scalar_values[idx])):
            if mask_surfaces:
                vertices, edges = marching_cubes_per_element(scalar_fields[idx], scalar_values[idx][i],
                                                             grid.spacing, input_data.extent,
                                                             mask=masks[idx])
            else:
                vertices, edges = marching_cubes_per_element(scalar_fields[idx], scalar_values[idx][i],
                                                             grid.spacing, input_data.extent,
                                                             mask=None)

            mc_vertices.append(vertices)
            mc_edges.append(edges)

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
