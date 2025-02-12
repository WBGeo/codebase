import numpy as np
from core.object_components import InputData, GeomodelResults
from LoopStructural import GeologicalModel
from core.utility.surface_mesh_extraction import marching_cubes
import pandas as pd


#%%
def loop_structural_interpolator(input_data: InputData, interpolator_type="FDI"):
    """
    Compute a model based on input data using loop structural.

    Args:
        input_data (InputData): The input data for the geological model.
        interpolator_type (str): The type of interpolator to use. Defaults to "FDI".

    Returns:
        resultsGeomodelResults: The results of the geological model.

    """
    # TODO: This does not consider faults, seems to fail for slim models

    # Based on input data create regular grid - this can be outsourced to a separate function
    dx = (input_data.extent[1] - input_data.extent[0]) / input_data.resolution[0]
    dy = (input_data.extent[3] - input_data.extent[2]) / input_data.resolution[1]
    dz = (input_data.extent[5] - input_data.extent[4]) / input_data.resolution[2]
    spacing = (dx, dy, dz)
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

    # Convert surface_points to loopstructural input DataFrame
    surface_points_temp = input_data.surface_points.copy()
    orientations_temp = input_data.orientations.copy()

    # map formation names to feature names (structural groups)
    surface_points_temp['feature_name'] = surface_points_temp['formation'].map(
        lambda x: next((k for k, v in input_data.mapping_object.items() if x in v), x))

    # Map unique formation strings to unique int values
    formation_mapping = {formation: idx for idx, formation in enumerate(surface_points_temp['formation'].unique())}

    surface_points_temp['val'] = surface_points_temp['formation'].map(formation_mapping)

    # Reorder columns if necessary
    surface_points_temp = surface_points_temp[['X', 'Y', 'Z', 'val', 'feature_name']]
    surface_points_temp['tx'] = np.nan
    surface_points_temp['ty'] = np.nan
    surface_points_temp['tz'] = np.nan

    orientations_temp['feature_name'] = "Strat_Series"
    orientations_temp['val'] = orientations_temp['formation'].map(formation_mapping)

    orientations_temp = orientations_temp.rename(columns={'G_x': 'tx', 'G_y': 'ty', 'G_z': 'tz'})
    orientations_temp = orientations_temp[['X', 'Y', 'Z', 'val', 'feature_name', 'tx', 'ty', 'tz']]

    # Create final combined df for loopstructural
    data_combined = pd.concat([surface_points_temp, orientations_temp], ignore_index=True)

    # Create a GeologicalModel instance
    model = GeologicalModel(input_data.extent[::2], input_data.extent[1::2])
    model.set_model_data(data_combined)

    # Set stratigraphic column
    stratigraphic_column = {}
    for series, rocks in input_data.mapping_object.items():
        stratigraphic_column[series] = {}
        for i, rock in enumerate(rocks):
            stratigraphic_column[series][rock] = {"min": i, "max": i + 1, "id": i}

    model.set_stratigraphic_column(stratigraphic_column)

    features = input_data.mapping_object.keys()
    strat_features = []

    for feature in features:
        strat = model.create_and_add_foliation(
            feature,
            interpolatortype=interpolator_type,  # try changing this to 'PLI'
            nelements=1e4,  # try changing between 1e3 and 5e4
            buffer=0,
            solver="cg",
            # npw=1,
            # gpw=100000,
            # regularisation=1,
            damp=True,
        )
        strat_features.append(strat)

    # Set grid
    # regular_grid = model.regular_grid(input_data.resolution, shuffle=False, rescale=True)
    regular_grid = grid

    results_sf = []
    for feature in input_data.mapping_object.keys():
        sf = model.evaluate_feature_value(feature, regular_grid, scale=True)
        results_sf.append(sf.reshape(input_data.resolution))

    # Replacing with lith_block ID and unconformity masking
    combined_result = np.zeros(results_sf[0].shape) + len(formation_mapping)

    groups_dict = {key: [formation_mapping[val] for val in values] for key, values in input_data.mapping_object.items()}


    counter = len(input_data.mapping_object)-1
    for key, values in reversed(groups_dict.items()):
        for j in reversed(values):
            combined_result[results_sf[counter] <= j] = j
        counter -= 1

    combined_result = combined_result.reshape(input_data.resolution)

    # Extract the surface meshes using marching cubes, does not consider faults as not possible atm
    mc_vertices, mc_edges = marching_cubes(combined_result, unique_elements, spacing, input_data.extent)

    # # Create a GeomodelResults instance
    results_instance = GeomodelResults(name=input_data.name,
                                       lith_block=combined_result.flatten(),
                                       surface_meshes_vertices=mc_vertices,
                                       surface_meshes_edges=mc_edges,
                                       grid=grid,
                                       extent=input_data.extent,
                                       resolution=input_data.resolution,
                                       mapping_object=input_data.mapping_object)

    return results_instance
