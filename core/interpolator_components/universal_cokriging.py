import numpy as np
import gempy as gp
from core.object_components import InputData, GeomodelResults
from skimage import measure
from core.utility.conversions import element_list_from_dict
from core.utility.model_cleaning import remove_outliers_3d
from core.utility.surface_mesh_extraction import marching_cubes_per_element


def universal_cokriging_interpolator(input_data: InputData,
                                    mask_surfaces=True) -> GeomodelResults:
    """
    Compute a model based on input data using universal co-kriging interpolation (gempy)

    Args:
        input_data (InputData): The input data for the geological model.
        mask_surfaces (bool): Whether to mask surfaces. Default is True.

    Returns:
        GeomodelResults: The results of the geological model.

    """
    # Test validity of input data for this interpolation
    # Check for orientations
    if input_data.orientations is None:
        raise ValueError("Interpolator requires orientations in the input data")

    # Create a structural frame
    # How this will look in the end depends mainly on how our input data component looks
    structural_frame = gp.data.structural_frame.StructuralFrame.from_data_tables(
        gp.data.surface_points.SurfacePointsTable.from_arrays(x=input_data.surface_points.X.to_numpy(),
                                                              y=input_data.surface_points.Y.to_numpy(),
                                                              z=input_data.surface_points.Z.to_numpy(),
                                                              names=input_data.surface_points.formation.to_numpy(),
                                                              nugget=np.zeros(len(input_data.surface_points)),
                                                              name_id_map=None),
        gp.data.orientations.OrientationsTable.from_arrays(x=input_data.orientations.X.to_numpy(),
                                                           y=input_data.orientations.Y.to_numpy(),
                                                           z=input_data.orientations.Z.to_numpy(),
                                                           G_x=input_data.orientations.G_x.to_numpy(),
                                                           G_y=input_data.orientations.G_y.to_numpy(),
                                                           G_z=input_data.orientations.G_z.to_numpy(),
                                                           names=input_data.orientations.formation.to_numpy(),
                                                           nugget=np.zeros(len(input_data.orientations)),
                                                           name_id_map=None)
    )

    # Create a GeoModel instance
    model_instance = gp.create_geomodel(
        project_name=input_data.name,
        extent=input_data.extent,
        resolution=input_data.resolution,
        structural_frame=structural_frame
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=model_instance,
        mapping_object=input_data.mapping_object
    )

    # Set faults
    if input_data.faults is not None:
        gp.set_is_fault(
            model_instance,
            fault_groups=np.array(list(input_data.mapping_object.keys()))[input_data.faults].tolist()
        )

    # Define fault relations
    if input_data.fault_relations is not None and input_data.faults is not None:
        model_instance.structural_frame.fault_relations = input_data.fault_relations
    elif input_data.fault_relations is not None and input_data.faults is None:
        print("Fault relations defined but no faults defined in input data.")
    elif input_data.fault_relations is None and input_data.faults is not None:
        # Default behavior, young affects everything below
        relations = np.zeros((len(model_instance.structural_frame.structural_groups),
                              len(model_instance.structural_frame.structural_groups)))
        for i in np.where(np.array(input_data.faults))[0]:
            relations[i, i + 1:] = 1
        model_instance.structural_frame.fault_relations = relations.astype(bool)
    else:
        pass

    print(model_instance.input_data_descriptor)

    # Compute the geological model
    gp.compute_model(model_instance)

    # Back transform vertices and get the dc meshes
    # dc_vertices_transformed = [model_instance.input_transform.apply_inverse(mesh.vertices) for mesh in
    #                            model_instance.solutions.dc_meshes]
    # dc_edges = [mesh.edges for mesh in model_instance.solutions.dc_meshes]

    block = model_instance.solutions.raw_arrays.lith_block.reshape(input_data.resolution)

    # Remove small isolated patches
    cleaned_block = remove_outliers_3d(block)

    cleaned_lith_block = cleaned_block.reshape(model_instance.solutions.raw_arrays.lith_block.shape)

    # extract scalar field values at surface points
    scalar_values = model_instance.solutions.raw_arrays.scalar_field_at_surface_points

    # Extract the surface meshes using marching cubes, does not consider faults as not possible atm
    # TODO: Does include faults now but I need to test with multiple structural groups with multiple faults
    mc_vertices = []
    mc_edges = []
    if input_data.faults is not None:

        # Get indices of fault and non_fault groups
        lith_group_indices = [i for i, fault in enumerate(input_data.faults) if not fault]
        fault_group_indices = [i for i, fault in enumerate(input_data.faults) if fault]

        # for i in np.unique(model_instance.solutions.raw_arrays.fault_block)[:-1]:
        for idx in fault_group_indices:

            # Use Scalar fields
            fault_block = model_instance.solutions.raw_arrays.scalar_field_matrix[idx].reshape(input_data.resolution)

            for i in range(len(scalar_values[idx])):
                vertices, edges = marching_cubes_per_element(fault_block,
                                                             scalar_values[idx][i],
                                                             spacing=(model_instance.grid.regular_grid.dx,
                                                                      model_instance.grid.regular_grid.dy,
                                                                      model_instance.grid.regular_grid.dz),
                                                             extent=input_data.extent,
                                                             mask=None)

                mc_vertices.append(vertices)
                mc_edges.append(edges)
    else:
        # Get indices non_fault groups (all groups)
        lith_group_indices = np.arange(len(input_data.mapping_object.keys()))

    # Create scalar fields masks
    masks = []
    masks.append(np.ones_like(model_instance.solutions.raw_arrays.scalar_field_matrix[0].reshape(input_data.resolution),
                              dtype=bool))
    for idx in lith_group_indices:
        mask = model_instance.solutions.raw_arrays.scalar_field_matrix[idx].reshape(input_data.resolution) <= \
               scalar_values[idx][-1]
        masks.append(mask)

    # for i in range(len(lith_group_indices)):
    #     mask = model_instance.solutions.raw_arrays.scalar_field_matrix[i].reshape(input_data.resolution) <= \
    #            scalar_values[i][-1]
    #     masks.append(mask)

    # Extract meshes for lithological elements
    counter = 0 # for masks
    for idx in lith_group_indices:
        scalar_field = model_instance.solutions.raw_arrays.scalar_field_matrix[idx].reshape(input_data.resolution)

        for i in range(len(scalar_values[idx])):
            if mask_surfaces:
                vertices, edges = marching_cubes_per_element(scalar_field,
                                                             scalar_values[idx][i],
                                                             spacing=(model_instance.grid.regular_grid.dx,
                                                                      model_instance.grid.regular_grid.dy,
                                                                      model_instance.grid.regular_grid.dz),
                                                             extent=input_data.extent,
                                                             mask=masks[counter])
            else:
                vertices, edges = marching_cubes_per_element(scalar_field,
                                                             scalar_values[idx][i],
                                                             spacing=(model_instance.grid.regular_grid.dx,
                                                                      model_instance.grid.regular_grid.dy,
                                                                      model_instance.grid.regular_grid.dz),
                                                             extent=input_data.extent,
                                                             mask=None)

            mc_vertices.append(vertices)
            mc_edges.append(edges)
        counter += 1

    # Reorder everything correctly if faults exist
    if input_data.faults is not None:

        bool_list = element_list_from_dict(input_data.mapping_object, input_data.faults)

        true_count = sum(bool_list)

        # Split arr_list into two parts
        true_elements_vertices = mc_vertices[:true_count]
        false_elements_vertices = mc_vertices[true_count:]
        true_elements_edges = mc_edges[:true_count]
        false_elements_edges = mc_edges[true_count:]

        # Create a new list to store reordered elements
        mc_vertices = []
        mc_edges = []

        # Iterator for both true and false elements
        true_idx, false_idx = 0, 0

        # Populate reordered_list based on bool_list
        for is_true in bool_list:
            if is_true:
                mc_vertices.append(true_elements_vertices[true_idx] + [input_data.extent[0], input_data.extent[2],
                                                                       input_data.extent[4]])
                mc_edges.append(true_elements_edges[true_idx])
                true_idx += 1
            else:
                mc_vertices.append(false_elements_vertices[false_idx] + [input_data.extent[0], input_data.extent[2],
                                                                         input_data.extent[4]])
                mc_edges.append(false_elements_edges[false_idx])
                false_idx += 1

    scalar_fields = model_instance.solutions.raw_arrays.scalar_field_matrix
    print(type(scalar_fields))

    # Create a GeomodelResults instance
    results_instance = GeomodelResults(name=input_data.name,
                                       # lith_block=model_instance.solutions.raw_arrays.lith_block,
                                       lith_block=cleaned_lith_block,
                                       # surface_meshes_vertices=dc_vertices_transformed,
                                       surface_meshes_vertices=mc_vertices,
                                       # surface_meshes_edges=dc_edges,
                                       surface_meshes_edges=mc_edges,
                                       grid=model_instance.grid.regular_grid.values,
                                       extent=model_instance.grid.regular_grid.extent,
                                       resolution=model_instance.grid.regular_grid.resolution,
                                       mapping_object=input_data.mapping_object,
                                       scalar_fields=scalar_fields)

    return results_instance
