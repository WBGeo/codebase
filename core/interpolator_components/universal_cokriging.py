import numpy as np
import gempy as gp
from core.object_components import InputData, GeomodelResults
from skimage import measure


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

    # Extract the surface meshes using marching cubes, does not consider faults as not possible atm
    # TODO: Does not include faults as of now
    # mc_vertices = []
    # mc_edges = []
    # block = model_instance.solutions.raw_arrays.lith_block.reshape(input_data.resolution)
    # print(block.min(), block.max())
    # for i in range(block.min(), block.max()):
    #     verts, faces, _, _ = measure.marching_cubes(block, i,
    #                                                 spacing=(model_instance.grid.regular_grid.dx,
    #                                                          model_instance.grid.regular_grid.dy,
    #                                                          model_instance.grid.regular_grid.dz))
    #     mc_vertices.append(verts)
    #     mc_edges.append(faces)

    # Create a GeomodelResults instance
    results_instance = GeomodelResults(name=input_data.name,
                                       lith_block=model_instance.solutions.raw_arrays.lith_block,
                                       surface_meshes_vertices=dc_vertices_transformed,
                                       # surface_meshes_vertices=mc_vertices,
                                       surface_meshes_edges=dc_edges,
                                       # surface_meshes_edges=mc_edges,
                                       grid=model_instance.grid.regular_grid.values,
                                       extent=model_instance.grid.regular_grid.extent,
                                       resolution=model_instance.grid.regular_grid.resolution,
                                       mapping_object=input_data.mapping_object)

    return results_instance
