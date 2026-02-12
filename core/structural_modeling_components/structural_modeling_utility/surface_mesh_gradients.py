import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from core.object_components import StructuralModelResults
from core.utility.conversions import normalize_vectors


def get_surface_mesh_gradients(result, norm=True, mesh_type="unmasked", return_faults=True):
    """
    Returns
    -------
    gradients_dict : dict
        {element.name: {"points": (N,3), "vectors": (N,3)}}
    gradients_faults_dict : dict
        {fault_element.name: {"points": (N,3), "vectors": (N,3)}}  (empty if no faults or return_faults=False)
    """
    mesh_counter = {"masked": 0, "unmasked": 1, "combined": 2}.get(mesh_type, 1)

    grid = result.structural_frame.grid
    res = grid.resolution

    # grid coords reshaped
    x = grid.grid_coordinates[:, 0].reshape(res)
    y = grid.grid_coordinates[:, 1].reshape(res)
    z = grid.grid_coordinates[:, 2].reshape(res)

    # sort axes for interpolator
    x_sorted_idx = np.argsort(x[:, 0, 0])
    y_sorted_idx = np.argsort(y[0, :, 0])
    z_sorted_idx = np.argsort(z[0, 0, :])

    x_sorted = x[x_sorted_idx, 0, 0]
    y_sorted = y[0, y_sorted_idx, 0]
    z_sorted = z[0, 0, z_sorted_idx]

    def make_interpolators(sf):
        grad = np.gradient(sf)
        gx = grad[0][x_sorted_idx, :, :]
        gy = grad[1][:, y_sorted_idx, :]
        gz = grad[2][:, :, z_sorted_idx]

        ix = RegularGridInterpolator((x_sorted, y_sorted, z_sorted), gx, bounds_error=False, fill_value=None)
        iy = RegularGridInterpolator((x_sorted, y_sorted, z_sorted), gy, bounds_error=False, fill_value=None)
        iz = RegularGridInterpolator((x_sorted, y_sorted, z_sorted), gz, bounds_error=False, fill_value=None)
        return ix, iy, iz

    def interpolate_on_mesh(element, ix, iy, iz):
        pts = element.get_mesh(mesh_type)[0]  # keep your convention
        vecs = np.column_stack((ix(pts), iy(pts), iz(pts)))
        if norm:
            vecs = normalize_vectors(vecs)
        return {
            "points": np.asarray(pts),
            "vectors": np.asarray(vecs),
        }

    # --- structural elements ---
    gradients_dict = {}
    for group in result.structural_frame.structural_groups:
        ix, iy, iz = make_interpolators(group.scalar_field)
        for element in group.structural_elements:
            gradients_dict[element.name] = interpolate_on_mesh(element, ix, iy, iz)

    # --- faults (optional) ---
    gradients_faults_dict = {}
    fault_frame = getattr(result.structural_frame, "fault_frame", None)

    if return_faults and fault_frame is not None:
        for fault_element in fault_frame.fault_elements:
            ix, iy, iz = make_interpolators(fault_element.scalar_field)
            gradients_faults_dict[fault_element.name] = interpolate_on_mesh(fault_element, ix, iy, iz)

    return gradients_dict, gradients_faults_dict


def plot_surface_mesh_gradients(structural_model_result,
                                gradients_dict,
                                gradients_faults_dict=None,
                                mesh_type="unmasked",
                                scale_factor=50):
    """
    Plot the gradient vector field at the surface mesh vertices
    Args:
        structural_model_result (StructuralModelResults): The results of the geological model.
        gradients_dict (dict): A dictionary containing the gradient vectors for each structural element.
        gradients_faults_dict (dict, optional): A dictionary containing the gradient vectors for each fault element. Default is None.
        mesh_type (str): The type of surface mesh to use. Default is "unmasked".
        scale_factor (float): The scale factor for the arrows. Default is 50.
    """

    # Plotting the gradient vector field
    plotter = pv.Plotter()

    for i, element in enumerate(gradients_dict.keys()):
        pdata = pv.PolyData(gradients_dict[element]['points'])
        pdata["vectors"] = gradients_dict[element]['vectors']  # Add vector field

        # Create arrow glyphs
        arrows = pdata.glyph(orient="vectors", scale="vectors", factor=scale_factor)

        # Plot the arrows
        plotter.add_mesh(arrows, color=structural_model_result.structural_frame.get_element_by_name(element).color)

        plotter.add_mesh(
            pv.PolyData(structural_model_result.structural_frame.get_element_by_name(element).get_mesh(mesh_type)[0],
                        np.insert(structural_model_result.structural_frame.get_element_by_name(element)
                                  .get_mesh(mesh_type)[1], 0, 3, axis=1).ravel()),
            color=structural_model_result.structural_frame.get_element_by_name(element).color, style="wireframe")

    if gradients_faults_dict is not None:
        for i, fault_element in enumerate(gradients_faults_dict.keys()):
            pdata = pv.PolyData(gradients_faults_dict[fault_element]['points'])
            pdata["vectors"] = gradients_faults_dict[fault_element]['vectors']  # Add vector field

            # Create arrow glyphs
            arrows = pdata.glyph(orient="vectors", scale="vectors", factor=scale_factor)

            # Plot the arrows
            plotter.add_mesh(arrows,
                             color=structural_model_result.structural_frame.fault_frame.get_element_by_name(
                                 fault_element).color)

            plotter.add_mesh(
                pv.PolyData(
                    structural_model_result.structural_frame.fault_frame.get_element_by_name(fault_element).get_mesh(
                        mesh_type)[0],
                    np.insert(structural_model_result.structural_frame.fault_frame.get_element_by_name(fault_element)
                              .get_mesh(mesh_type)[1], 0, 3, axis=1).ravel()),
                color=structural_model_result.structural_frame.fault_frame.get_element_by_name(fault_element).color,
                style="wireframe")

    plotter.show()
