import numpy as np
import gempy as gp
from core.object_components import InputData, GeomodelResults


def universal_cokriging_interpolator(input_data: InputData):
    """
    Compute a model based on input data using universal co-kriging interpolation (gempy)

    Args:
        input_data (InputData): The input data for the geological model.

    Returns:
        GeomodelResults: The results of the geological model.

    """
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
        model_instance.structural_frame.fault_relations = relations
    else:
        pass

    # Compute the geological model
    gp.compute_model(model_instance)

    # Back transform vertices
    dc_vertices_transformed = [model_instance.input_transform.apply_inverse(mesh.vertices) for mesh in
                               model_instance.solutions.dc_meshes]
    dc_edges = [mesh.edges for mesh in model_instance.solutions.dc_meshes]

    # Reorder meshes to append faults at end
    # TODO: I have now strictly ordered by age, lets see how this works with multiple weird faults

    # new version, for case all faults are ordered at beginning - I doubt it
    # n_faults = len(np.where(input_data.faults)[0])
    # n_elements = len(dc_vertices_transformed)
    #
    # dc_edges2 = np.empty(n_elements, dtype=object)
    # dc_edges2[0:n_elements-n_faults] = dc_edges[n_faults:n_elements]
    # dc_edges2[n_elements-n_faults:] = dc_edges[0:n_faults]
    #
    # dc_vertices_transformed2 = np.empty(n_elements, dtype=object)
    # dc_vertices_transformed2[0:n_elements-n_faults] = dc_vertices_transformed[n_faults:n_elements]
    # dc_vertices_transformed2[n_elements-n_faults:] = dc_vertices_transformed[0:n_faults]

    # New version, if elements in meshes are ordered by age - this one should work in general
    # if input_data.faults is not None:
    #     # indices of faults
    #     indices_to_cut = np.where(np.array(input_data.faults))[0]
    #
    #     # reorder meshes
    #     dc_edges = np.array(dc_edges, dtype=object)
    #     elements_to_append = dc_edges[indices_to_cut]
    #     dc_edges = np.delete(dc_edges, indices_to_cut)
    #     dc_edges = np.concatenate((dc_edges, elements_to_append))
    #
    #     dc_vertices_transformed = np.array(dc_vertices_transformed, dtype=object)
    #     elements_to_append = dc_vertices_transformed[indices_to_cut]
    #     dc_vertices_transformed = np.delete(dc_vertices_transformed, indices_to_cut)
    #     dc_vertices_transformed = np.concatenate((dc_vertices_transformed, elements_to_append))

    # Create a GeomodelResults instance
    results_instance = GeomodelResults(lith_block=model_instance.solutions.raw_arrays.lith_block,
                                       surface_meshes_vertices=dc_vertices_transformed,
                                       surface_meshes_edges=dc_edges,
                                       grid=model_instance.grid.regular_grid.values,
                                       extent=model_instance.grid.regular_grid.extent,
                                       resolution=model_instance.grid.regular_grid.resolution)

    return results_instance
