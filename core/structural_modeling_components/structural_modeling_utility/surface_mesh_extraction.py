"""
Surface mesh extraction utilities for structural geological modeling.

Thin wrappers around scikit-image marching cubes for extracting isosurfaces
from scalar fields, with optional domain masking.
"""
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from skimage import measure


def _grid_origin(
    spacing: Tuple[float, float, float],
    extent: Tuple[float, ...],
) -> NDArray[np.float64]:
    """
    World coordinates of array index (0, 0, 0) on a cell-centred grid.

    ``skimage.measure.marching_cubes`` returns vertices relative to the first
    array sample. Grid values are evaluated at cell centres (see
    ``RegularGrid``), so the first sample sits half a cell inside the extent.
    """
    return np.array([
        extent[0] + spacing[0] / 2,
        extent[2] + spacing[1] / 2,
        extent[4] + spacing[2] / 2,
    ], dtype=float)


def marching_cubes(
    block: NDArray[np.float64],
    elements: List[float],
    spacing: Tuple[float, float, float],
    extent: Tuple[float, ...],
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.int64]]]:
    """
    Extract isosurfaces for multiple isovalue levels from a scalar field.

    Parameters
    ----------
    block : np.ndarray
        3D scalar field array.
    elements : sequence of float
        Isovalues at which to extract surfaces.
    spacing : tuple[float, float, float]
        Voxel spacing in (x, y, z).
    extent : tuple[float, ...]
        Model extent as (xmin, xmax, ymin, ymax, zmin, zmax). The block is
        assumed cell-centred on this extent; vertices are offset by
        ``extent.min + spacing / 2`` to world coordinates.

    Returns
    -------
    mc_vertices : list[np.ndarray]
        Vertex arrays in world coordinates, one per isovalue.
    mc_edges : list[np.ndarray]
        Triangle index arrays, one per isovalue.
    """
    origin = _grid_origin(spacing, extent)
    mc_vertices, mc_edges = [], []
    for level in elements:
        verts, faces, _, _ = measure.marching_cubes(block, float(level), spacing=spacing)
        mc_vertices.append(verts + origin)
        mc_edges.append(faces)
    return mc_vertices, mc_edges


def marching_cubes_per_element(
    block: NDArray[np.float64],
    element: float,
    spacing: Tuple[float, float, float],
    extent: Tuple[float, ...],
    mask: Optional[NDArray[np.bool_]],
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Extract the surface mesh for a single isovalue from a scalar field.

    Parameters
    ----------
    block : np.ndarray
        3D scalar field array.
    element : float
        Isovalue of the structural element to extract.
    spacing : tuple[float, float, float]
        Voxel spacing in (x, y, z).
    extent : tuple[float, ...]
        Model extent as (xmin, xmax, ymin, ymax, zmin, zmax). The block is
        assumed cell-centred on this extent; vertices are offset by
        ``extent.min + spacing / 2`` to world coordinates.
    mask : np.ndarray or None
        Boolean mask restricting the extraction region. Pass None for no masking.

    Returns
    -------
    vertices : np.ndarray
        Vertex array in world coordinates, shape (N, 3).
        Empty (shape (0, 3)) if the isovalue is not crossed in the masked region.
    edges : np.ndarray
        Triangle index array, shape (M, 3).
        Empty (shape (0, 3)) if the isovalue is not crossed in the masked region.
    """
    # If the scalar field does not cross the iso-value anywhere in the (masked)
    # domain — e.g. a formation absent from one fault block — return empty arrays
    # so the caller can combine or skip them gracefully.
    try:
        verts, edges, _, _ = measure.marching_cubes(block, element, spacing=spacing, mask=mask)
    except RuntimeError:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices = verts + _grid_origin(spacing, extent)
    return vertices, edges
