import numpy as np
from core.object_components import InputData, GeomodelResults
from pykrige.ok3d import OrdinaryKriging3D
from skimage import measure


#%%
def ordinary_kriging_interpolator(input_data: InputData):
    """
    Compute a model based on input data using kriging interpolation

    Args:
        input_data (InputData): The input data for the geological model.

    Returns:
        resultsGeomodelResults: The results of the geological model.

    """
    # TODO: This does not consider faults
    # TODO: This does require two elements per group to make sense

    # Based on input data create regular grid - this can be outsourced to a separate function
    dx = (input_data.extent[1] - input_data.extent[0]) / input_data.resolution[0]
    dy = (input_data.extent[3] - input_data.extent[2]) / input_data.resolution[1]
    dz = (input_data.extent[5] - input_data.extent[4]) / input_data.resolution[2]
    gridx = np.linspace(input_data.extent[0] + dx / 2, input_data.extent[1] - dx / 2, input_data.resolution[0])
    gridy = np.linspace(input_data.extent[2] + dy / 2, input_data.extent[3] - dy / 2, input_data.resolution[1])
    gridz = np.linspace(input_data.extent[4] + dz / 2, input_data.extent[5] - dz, input_data.resolution[2])

    # gempy way to get coordinates, for some reason this does not blow memory
    coords = gridx, gridy, gridz
    g = np.meshgrid(*coords, indexing="ij")
    grid = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")

    # Create mapping for replacing element names with ints
    unique_elements = sorted(set(element for elements in input_data.mapping_object.values() for element in elements))
    replacements = {element: i + 1 for i, element in enumerate(unique_elements)}

    # Separate data based on structural groups
    results = []
    masks = []
    for key, value in input_data.mapping_object.items():
        structural_group_df = input_data.surface_points[
            input_data.surface_points['formation'].isin(list(input_data.mapping_object[key]))]
        structural_group_df.loc[
            structural_group_df['formation'].isin(list(input_data.mapping_object[key])), 'formation'] = \
            structural_group_df['formation'].replace(replacements)

        # perform kriging per structural group
        # TODO: Set reasonable default variogram model and parameters
        ok3d = OrdinaryKriging3D(
            structural_group_df['X'], structural_group_df['Y'], structural_group_df['Z'],
            structural_group_df['formation'], variogram_model="gaussian",
            variogram_parameters=[1, 500, 0],
            anisotropy_scaling_z=0.3
        )
        k3d1, ss3d = ok3d.execute("grid", gridx, gridy, gridz)

        # Save results, need to explicitly limit to maximum value as defined by replacement mapping
        max_value = max(replacements[element] for element in value)
        k3d1[k3d1 > max_value] = max_value
        results.append(k3d1.astype(int))

        # Create mask for values below the lowest integer value for stacking
        min_value = min(replacements[element] for element in value)
        mask = k3d1 >= min_value
        masks.append(mask)

    # Create combined result
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
    mc_vertices = []
    mc_edges = []
    block = combined_result
    for i in range(0, len(unique_elements)):
        verts, faces, _, _ = measure.marching_cubes(block, i,
                                                    spacing=(dx, dy, dz))
        mc_vertices.append(verts)
        mc_edges.append(faces)

    # Create a GeomodelResults instance
    results_instance = GeomodelResults(name=input_data.name,
                                       lith_block=combined_result.flatten(),
                                       surface_meshes_vertices=mc_vertices,
                                       surface_meshes_edges=mc_edges,
                                       grid=grid,
                                       extent=input_data.extent,
                                       resolution=input_data.resolution)

    return results_instance
