import numpy as np
from scipy.interpolate import RegularGridInterpolator
from core.object_components import InputData, GeomodelResults
from core.utility.conversions import normalize_vectors


def get_surface_mesh_gradients(geo_model_results, norm=True):
    """
    Get the gradient vector field at the surface mesh vertices
    Args:
        geo_model_results (GeomodelResults): The results of the geological model.
        norm (bool): Normalize the gradient vectors. Default is True.
    Returns:
        points_list (list): The surface mesh vertices per element
        vectors_list (list): The gradient vector field at the surface mesh vertices per element
    """
    scalar_fields = geo_model_results.scalar_fields

    mesh_type=1

    points_list = []
    vectors_list = []

    #TODO: This is a temporary solution to get the group index for each element
    element_counter = 0  # Global index counter
    group_pointer = []
    for group, element in enumerate(geo_model_results.mapping_object.values()):  # key_id as the first index
        for value_id, value in enumerate(element):  # value_id for each item within a key
            group_pointer.append(group)
            element_counter += 1

    for i in range(element_counter):

        # Compute gradients on the regular grid
        grad = np.gradient(scalar_fields[group_pointer[i]].reshape(geo_model_results.resolution), axis=(0, 1, 2))

        # Reshape the grid coordinates to match the 3D structure
        x = geo_model_results.grid[:, 0].reshape(geo_model_results.resolution)
        y = geo_model_results.grid[:, 1].reshape(geo_model_results.resolution)
        z = geo_model_results.grid[:, 2].reshape(geo_model_results.resolution)

        # Sort along each axis and get sorted indices
        x_sorted_idx = np.argsort(x[:, 0, 0])  # Sort only along the first axis
        y_sorted_idx = np.argsort(y[0, :, 0])  # Sort along the second axis
        z_sorted_idx = np.argsort(z[0, 0, :])  # Sort along the third axis

        # Apply sorting to coordinates
        x_sorted = x[x_sorted_idx, 0, 0]
        y_sorted = y[0, y_sorted_idx, 0]
        z_sorted = z[0, 0, z_sorted_idx]

        # Apply sorting to gradients (along the respective axis)
        grad_x_sorted = grad[0][x_sorted_idx, :, :]
        grad_y_sorted = grad[1][:, y_sorted_idx, :]
        grad_z_sorted = grad[2][:, :, z_sorted_idx]

        # Modify interpolators to avoid bounds error
        interp_grad_x = RegularGridInterpolator(
            (x_sorted, y_sorted, z_sorted), grad_x_sorted, bounds_error=False, fill_value=None
        )
        interp_grad_y = RegularGridInterpolator(
            (x_sorted, y_sorted, z_sorted), grad_y_sorted, bounds_error=False, fill_value=None
        )
        interp_grad_z = RegularGridInterpolator(
            (x_sorted, y_sorted, z_sorted), grad_z_sorted, bounds_error=False, fill_value=None
        )

        new_points = geo_model_results.surface_meshes_vertices[mesh_type][i]

        # Interpolate gradients at new points
        grad_x_new = interp_grad_x(new_points)
        grad_y_new = interp_grad_y(new_points)
        grad_z_new = interp_grad_z(new_points)

        # Combine into a gradient vector field
        # new_gradients = np.vstack([grad_x_new, grad_y_new, grad_z_new]).T

        # Create a PyVista dataset
        points = np.column_stack((new_points[:, 0], new_points[:, 1], new_points[:, 2]))  # Combine coordinates
        vectors = np.column_stack((grad_x_new, grad_y_new, grad_z_new))  # Combine vector components

        # Apply normalization
        if norm:
            vectors = normalize_vectors(vectors)

        points_list.append(points)
        vectors_list.append(vectors)

    return points_list, vectors_list
