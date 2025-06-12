from skimage import measure

def marching_cubes_new(block, elements, spacing, extent):
    """
    Extract the surface meshes using marching cubes.
    Args:
        block (np.array): The block to extract the surface meshes from.
        elements (list[int]): IDs of unique structural elements in model.
        spacing (tuple): The spacing between grid points in the block.
        extent (tuple): The extent of the model.

    Returns:
        mc_vertices (list): Vertices of the surface meshes.
        mc_edges (list): Edges of the surface meshes.
    """
    mc_vertices = []
    mc_edges = []
    for elem_id in elements:
        verts, faces, _, _ = measure.marching_cubes(block, elem_id, spacing=spacing)
        mc_vertices.append(verts + [extent[0], extent[2], extent[4]])
        mc_edges.append(faces)
    return mc_vertices, mc_edges


def marching_cubes(block, elements, spacing, extent):
    """
    Extract the surface meshes using marching cubes
    Args:
        block (np.array): The block to extract the surface meshes from.
        elements (list): IDs of unique structural elements in model
        spacing (tuple): The spacing between grid points in the block.

    Returns:
        mc_vertices (list): Vertices of the surface meshes.
        mc_edges (list): Edges of the surface meshes.
    """

    # Extract the surface meshes using marching cubes
    mc_vertices = []
    mc_edges = []
    for i in range(0, len(elements)):
        verts, faces, _, _ = measure.marching_cubes(block, i,
                                                    spacing=spacing)
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

    # Extract the surface meshes using marching cubes
    verts, edges, _, _ = measure.marching_cubes(block, element, spacing=spacing, mask=mask)
    vertices = verts + [extent[0], extent[2], extent[4]]
    return vertices, edges
