import numpy as np
from core.object_components import InputData, GeomodelResults
from pykrige.ok3d import OrdinaryKriging3D
import matplotlib.pyplot as plt

# TODO: This is just for testing here
import pandas as pd
import os

from core.object_components import InputData
from core.visualization_components import plot_2d, plot_3d
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator


#%%
def kriging_interpolator(input_data: InputData):
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
    gridx = np.linspace(input_data.extent[0] + dx/2, input_data.extent[1] - dx/2, input_data.resolution[0])
    gridy = np.linspace(input_data.extent[2] + dy/2, input_data.extent[3] - dy/2, input_data.resolution[1])
    gridz = np.linspace(input_data.extent[4] + dz/2, input_data.extent[5] - dz, input_data.resolution[2])

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
        ok3d = OrdinaryKriging3D(
            structural_group_df['X'], structural_group_df['Y'], structural_group_df['Z'],
            structural_group_df['formation'], variogram_model="gaussian",
            variogram_parameters=[4, 500, 0],
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

    # TODO: Actually do meshing
    a = []
    b = []

    # Create a GeomodelResults instance
    results_instance = GeomodelResults(lith_block=combined_result.flatten(),
                                       surface_meshes_vertices=a,
                                       surface_meshes_edges=b,
                                       grid=grid,
                                       extent=input_data.extent,
                                       resolution=input_data.resolution)

    return results_instance


#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 2: no faults, no unconformities, 2 stratigraphic series

# # Component 1: input data
# data_test = InputData(name='Model 2',
#                       extent=np.array([0, 1000, 0, 1000, 0, 1000]),
#                       resolution=np.array([40, 40, 40]),
#                       surface_points=pd.read_csv(
#                           cwd + "/examples/data/model2_surface_points_df.csv"),
#                       orientations=pd.read_csv(
#                           cwd + "/examples/data/model2_orientations_df.csv"),
#                       mapping_object={"Strat_Series": ('rock2', 'rock1')}
#                       )

data_test = InputData(name='Model 12',
                      extent=np.array([0, 2000, 0, 1000, 0, 1000]),
                      resolution=np.array([100, 50, 50]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model12_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/model12_orientations_df.csv"),
                      mapping_object={
                          "Strat_Series1": ('rock4', 'rock3'),
                          "Strat_Series2": ('rock2', 'rock1')},
                      )

#%%

# Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)

#%%
results_test2 = universal_cokriging_interpolator(data_test)

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
results_test = kriging_interpolator(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
# plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)

#%%

