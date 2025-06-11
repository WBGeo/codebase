import numpy as np
from core.object_components import InputData, GeomodelResults
from LoopStructural import GeologicalModel
from core.utility.surface_mesh_extraction import marching_cubes_per_element, marching_cubes
import pandas as pd
from core.grids.grid_classes import RegularGrid


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

    # Create a Grid instance
    grid = RegularGrid(input_data.extent, input_data.resolution)

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
    # TODO: The scaling between these values actually matters for the interpolation
    formation_mapping = {formation: idx for idx, formation in enumerate(surface_points_temp['formation'].unique())}

    surface_points_temp['val'] = surface_points_temp['formation'].map(formation_mapping)

    # Reorder columns if necessary
    surface_points_temp = surface_points_temp[['X', 'Y', 'Z', 'val', 'feature_name']]
    surface_points_temp['gx'] = np.nan
    surface_points_temp['gy'] = np.nan
    surface_points_temp['gz'] = np.nan

    # TODO: Location of these orientations seems to matter (a lot) for a loopstructural model
    # map formation names to feature names (structural groups)
    orientations_temp['feature_name'] = orientations_temp['formation'].map(
        lambda x: next((k for k, v in input_data.mapping_object.items() if x in v), x))

    orientations_temp['val'] = orientations_temp['formation'].map(formation_mapping)

    orientations_temp = orientations_temp.rename(columns={'G_x': 'gx', 'G_y': 'gy', 'G_z': 'gz'})
    orientations_temp = orientations_temp[['X', 'Y', 'Z', 'val', 'feature_name', 'gx', 'gy', 'gz']]

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
    regular_grid = grid.grid_coordinates

    scalar_fields = []
    for feature in input_data.mapping_object.keys():
        sf = model.evaluate_feature_value(feature, regular_grid, scale=True)
        scalar_fields.append(sf.reshape(input_data.resolution))

    # TODO: These will both not work with faults
    # Get indices for each lithological group
    lith_group_indices = np.arange(len(input_data.mapping_object.keys()))
    groups_dict = {key: [formation_mapping[val] for val in values] for key, values in input_data.mapping_object.items()}
    # Get scalar value per element grouped same way and order as mapping object
    scalar_values = [np.array(v) for v in groups_dict.values()]

    # Create scalar fields masks
    masks = []
    masks.append(np.ones_like(scalar_fields[0], dtype=bool))
    for i in range(len(lith_group_indices) - 1):
        mask = scalar_fields[i] >= scalar_values[i][-1] # TODO: Gefahr
        masks.append(mask)

    # TODO: This works but I am a little confused about order and orientation of the scalar fields
    # Replacing with lith_block ID and unconformity masking
    combined_result = np.zeros(scalar_fields[0].shape) + len(formation_mapping)

    counter = len(input_data.mapping_object)-1
    for key, values in reversed(groups_dict.items()):
        for j in reversed(values):
            combined_result[scalar_fields[counter] <= j] = j
        counter -= 1

    combined_result = combined_result.reshape(input_data.resolution)

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

    # # Create a GeomodelResults instance
    results_instance = GeomodelResults(name=input_data.name,
                                       lith_block=combined_result.flatten(),
                                       surface_meshes_vertices=mc_vertices,
                                       surface_meshes_edges=mc_edges,
                                       grid=grid.grid_coordinates,
                                       extent=input_data.extent,
                                       resolution=input_data.resolution,
                                       mapping_object=input_data.mapping_object)

    return results_instance
