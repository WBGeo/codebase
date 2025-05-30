import numpy as np
from core.object_components import InputData, GeomodelResults
from pykrige.ok3d import OrdinaryKriging3D
from core.utility.surface_mesh_extraction import marching_cubes_per_element, marching_cubes
from skimage import measure
from core.grids.grid_classes import RegularGrid


#%%
def ordinary_kriging_interpolator(input_data: InputData,
                                  var_model="gaussian",
                                  var_sill=1,
                                  var_range=500,
                                  var_nugget=0,
                                  anisotropy_scaling_z=0.3) -> GeomodelResults:
    """
    Compute a model based on input data using kriging interpolation

    Args:
        input_data (InputData): The input data for the geological model.
        var_model (str): The variogram model to use. Default is 'gaussian'.
        var_sill (float): The sill of the variogram. Default is 1.
        var_range (float): The range of the variogram. Default is 500.
        var_nugget (float): The nugget of the variogram. Default is 0.
        anisotropy_scaling_z (float): The scaling factor for the z-axis. Default is 0.3.

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

    # plot slice of mask in y direction
    # 0 alles true
    # 1 nur oben true
    # import matplotlib.pyplot as plt
    # mask = masks[1].T
    # mask = mask[:, 25, :]
    # plt.imshow(mask, origin='lower')
    # plt.show()

    # plot slice of result in y direction
    # 1 ist untere grupe
    # 0 ist obere gruppe
    # result=results[1].T
    # result = result[:, 25, :]
    # plt.imshow(result, origin='lower')
    # plt.show()

    # Stack result based on stack
    combined_result = np.zeros_like(results[0])
    # Iterate over the results and masks arrays
    for i in range(len(results)):
        combined_result[masks[i]] = results[i][masks[i]]

    # TODO: Why is there a negative 1 value in this
    # replace all minus ones with zeros in combined result
    combined_result[combined_result == -1] = 0

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
