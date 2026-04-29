import numpy as np
from skimage import measure


def _extrapolate_pad(block):
    """Pad block by 1 cell on all 6 faces using linear extrapolation.

    This extends the scalar field trends beyond the model boundary so that
    marching cubes naturally continues each isosurface through the extent wall
    at the correct angle, without creating artificial flat caps.
    """
    padded = np.pad(block, 1, mode='edge')
    # Extrapolate linearly from the two outermost real cells on each face.
    # X faces
    padded[0, 1:-1, 1:-1] = 2 * block[0, :, :] - block[1, :, :]
    padded[-1, 1:-1, 1:-1] = 2 * block[-1, :, :] - block[-2, :, :]
    # Y faces
    padded[1:-1, 0, 1:-1] = 2 * block[:, 0, :] - block[:, 1, :]
    padded[1:-1, -1, 1:-1] = 2 * block[:, -1, :] - block[:, -2, :]
    # Z faces
    padded[1:-1, 1:-1, 0] = 2 * block[:, :, 0] - block[:, :, 1]
    padded[1:-1, 1:-1, -1] = 2 * block[:, :, -1] - block[:, :, -2]
    return padded


def _pad_block(block, mask=None):
    """Pad block by 1 cell on all 6 faces using linear extrapolation.

    The mask (if given) is padded with True so the padding region is included
    in masked marching cubes runs, allowing surfaces to exit through the extent.
    """
    padded_block = _extrapolate_pad(block)
    padded_mask = np.pad(mask, 1, mode='edge') if mask is not None else None
    return padded_block, padded_mask


def _clip_to_extent(verts, extent):
    """Clip vertex coordinates to the model extent."""
    np.clip(verts[:, 0], extent[0], extent[1], out=verts[:, 0])
    np.clip(verts[:, 1], extent[2], extent[3], out=verts[:, 1])
    np.clip(verts[:, 2], extent[4], extent[5], out=verts[:, 2])
    return verts


def marching_cubes(block, elements, spacing, extent):
    # If elements looks like [0,1,2,...] we're fine either way.
    # Otherwise we should respect the IDs provided.
    padded_block, _ = _pad_block(block)
    # Shift origin back by one spacing to account for the added padding layer.
    origin = [extent[0] - spacing[0], extent[2] - spacing[1], extent[4] - spacing[2]]
    mc_vertices, mc_edges = [], []
    for level in elements:
        verts, faces, _, _ = measure.marching_cubes(padded_block, float(level), spacing=spacing)
        verts = _clip_to_extent(verts + origin, extent)
        mc_vertices.append(verts)
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
    padded_block, padded_mask = _pad_block(block, mask)
    # Shift origin back by one spacing to account for the added padding layer.
    origin = [extent[0] - spacing[0], extent[2] - spacing[1], extent[4] - spacing[2]]
    try:
        verts, edges, _, _ = measure.marching_cubes(padded_block, element, spacing=spacing, mask=padded_mask)
    except RuntimeError:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices = _clip_to_extent(verts + origin, extent)
    return vertices, edges