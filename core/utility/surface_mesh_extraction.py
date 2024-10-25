from skimage import measure


def marching_cubes(block, elements, spacing):
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
        mc_vertices.append(verts)
        mc_edges.append(faces)
    return mc_vertices, mc_edges
