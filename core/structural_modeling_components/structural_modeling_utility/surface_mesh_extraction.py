import numpy as np
from skimage import measure


def marching_cubes(block, elements, spacing, extent):
    # If elements looks like [0,1,2,...] we’re fine either way.
    # Otherwise we should respect the IDs provided.
    mc_vertices, mc_edges = [], []
    for level in elements:
        verts, faces, _, _ = measure.marching_cubes(block, float(level), spacing=spacing)
        mc_vertices.append(verts + [extent[0], extent[2], extent[4]])
        mc_edges.append(faces)
    return mc_vertices, mc_edges


def marching_cubes_per_element(block, element, spacing, extent, mask):
    """
    Extract the surface meshes using marching cubes
    Args:
        block (np.array): The block to extract the surface meshes from.
        element (float): Scalar value of unique structural element in model
        spacing (tuple): The spacing between grid points in the block.
        extent (tuple): The extent of the model.
        mask (np.array): The mask for the group of elements.

    Returns:
        vertices (np.array): Vertices of the surface meshes.
        edges (np.array): Edges of the surface meshes.
    """

    # Extract the surface meshes using marching cubes.
    # If the scalar field does not cross the iso-value anywhere in the (masked)
    # domain — e.g. a formation that is absent from one fault block — return
    # empty arrays so the caller can combine or skip them gracefully.
    try:
        verts, edges, _, _ = measure.marching_cubes(block, element, spacing=spacing, mask=mask)
    except RuntimeError:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices = verts + [extent[0], extent[2], extent[4]]
    return vertices, edges
