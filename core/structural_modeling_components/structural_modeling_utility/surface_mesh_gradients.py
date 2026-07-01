"""
Gradient computation and visualization for surface meshes.

Provides utilities for interpolating scalar-field gradients onto surface mesh
vertices and plotting the resulting vector field.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from core.object_components import StructuralModelResults
from core.structural_modeling_components.structural_objects.structural_objects import GeoMeshType, FaultMeshType


def normalize_vectors(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize an array of vectors to unit length.

    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (N, 3).

    Returns
    -------
    np.ndarray
        Row-normalized array of shape (N, 3). Zero-norm rows are left unchanged.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def get_surface_mesh_gradients(
    result: StructuralModelResults,
    norm: bool = True,
    mesh_type: GeoMeshType = GeoMeshType.UNMASKED,
    return_faults: bool = True,
    fault_mesh_type: Optional[FaultMeshType] = None,
) -> Tuple[Dict[str, Dict[str, NDArray[np.float64]]], Dict[str, Dict[str, NDArray[np.float64]]]]:
    """
    Compute gradient vectors at surface mesh vertices by interpolating scalar field gradients.

    Parameters
    ----------
    result : StructuralModelResults
        Computed structural model results.
    norm : bool, default True
        Whether to normalize gradient vectors to unit length.
    mesh_type : GeoMeshType, default GeoMeshType.UNMASKED
        Mesh type to use for structural elements.
    return_faults : bool, default True
        Whether to also compute gradients for fault elements.
    fault_mesh_type : FaultMeshType, optional
        Mesh type to use for fault elements. Defaults to ``mesh_type`` if not set.

    Returns
    -------
    gradients_dict : dict
        ``{element.name: {"points": (N, 3), "vectors": (N, 3)}}``
    gradients_faults_dict : dict
        ``{fault_element.name: {"points": (N, 3), "vectors": (N, 3)}}``
        Empty if no faults or ``return_faults=False``.
    """
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

    def interpolate_on_mesh(element, ix, iy, iz, mt):
        pts = element.get_mesh(mt)[0]
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
            gradients_dict[element.name] = interpolate_on_mesh(element, ix, iy, iz, mesh_type)

    # --- faults (optional) ---
    gradients_faults_dict = {}
    fault_frame = getattr(result.structural_frame, "fault_frame", None)
    _fault_mesh_type = fault_mesh_type if fault_mesh_type is not None else mesh_type

    if return_faults and fault_frame is not None:
        for fault_element in fault_frame.fault_elements:
            ix, iy, iz = make_interpolators(fault_element.scalar_field)
            gradients_faults_dict[fault_element.name] = interpolate_on_mesh(
                fault_element, ix, iy, iz, _fault_mesh_type
            )

    return gradients_dict, gradients_faults_dict


def plot_surface_mesh_gradients(
    structural_model_result: StructuralModelResults,
    gradients_dict: Dict[str, Dict[str, NDArray[np.float64]]],
    gradients_faults_dict: Optional[Dict[str, Dict[str, NDArray[np.float64]]]] = None,
    mesh_type: GeoMeshType = GeoMeshType.UNMASKED,
    scale_factor: float = 50,
) -> None:
    """
    Plot gradient vector fields at surface mesh vertices using PyVista.

    Parameters
    ----------
    structural_model_result : StructuralModelResults
        The computed structural model results (used for mesh and color lookup).
    gradients_dict : dict
        Gradient data for structural elements, as returned by
        :func:`get_surface_mesh_gradients`.
    gradients_faults_dict : dict, optional
        Gradient data for fault elements. If None, faults are not plotted.
    mesh_type : GeoMeshType, default GeoMeshType.UNMASKED
        Mesh type used to retrieve element meshes for wireframe overlay.
    scale_factor : float, default 50
        Arrow scale factor for gradient glyphs.
    """
    plotter = pv.Plotter()

    for i, element in enumerate(gradients_dict.keys()):
        pdata = pv.PolyData(gradients_dict[element]['points'])
        pdata["vectors"] = gradients_dict[element]['vectors']

        arrows = pdata.glyph(orient="vectors", scale="vectors", factor=scale_factor)
        plotter.add_mesh(arrows, color=structural_model_result.structural_frame.get_element_by_name(element).color)

        plotter.add_mesh(
            pv.PolyData(structural_model_result.structural_frame.get_element_by_name(element).get_mesh(mesh_type)[0],
                        np.insert(structural_model_result.structural_frame.get_element_by_name(element)
                                  .get_mesh(mesh_type)[1], 0, 3, axis=1).ravel()),
            color=structural_model_result.structural_frame.get_element_by_name(element).color, style="wireframe")

    if gradients_faults_dict is not None:
        for i, fault_element in enumerate(gradients_faults_dict.keys()):
            pdata = pv.PolyData(gradients_faults_dict[fault_element]['points'])
            pdata["vectors"] = gradients_faults_dict[fault_element]['vectors']

            arrows = pdata.glyph(orient="vectors", scale="vectors", factor=scale_factor)
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
