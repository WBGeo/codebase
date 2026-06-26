import numpy as np
import pandas as pd
from core.meshing_components.explicit.structured.grid_generator import create_surface_grid, sort_points_by_x_y
from core.meshing_components.explicit.structured.grid_generator import sort_surfaces_by_z, store_points_in_array
from core.meshing_components.explicit.structured.store_grid_data import create_surfaces_with_grids_for_bottom_and_top
from core.meshing_components.explicit.structured.store_grid_data import create_intermediate_layers
from core.meshing_components.explicit.structured.node_element_generator import adjust_z_values, \
    create_hexahedral_elements_with_nodes
from core.object_components import StructuralModelResults, MeshResults, ExtentData
from core.structural_modeling_components.structural_objects.structural_objects import GeoMeshType
import typing
from typing import List, Tuple, Dict
from scipy.spatial import cKDTree
from numpy.typing import NDArray
import meshio
from types import SimpleNamespace
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
from core.meshing_components.geometry.Nodes import Nodes

# Rsample while preserving Z
def resample_preserve_z_nearest(points: NDArray[np.floating], nx: int, ny: int, extent: List[float]) -> NDArray[np.floating]:
    """
    Resample the point cloud to an nx × ny grid while preserving original Z values
    by assigning each grid node the Z of the nearest original point.

    Args
    ----------
    points : (N, 3) ndarray
        Cropped original surface points (X, Y, Z)
    nx, ny : int
        Grid resolution in X and Y
    extent : (min_x, max_x, min_y, max_y, min_z, max_z)

    Returns
    -------
    (nx*ny, 3) ndarray
        Structured grid [X, Y, Z]
    """

    if points.size == 0:
        raise ValueError("No points provided to resample_preserve_z_nearest")

    min_x, max_x, min_y, max_y, _, _ = extent

    xi: NDArray[np.floating] = np.linspace(min_x, max_x, nx)
    yi: NDArray[np.floating] = np.linspace(min_y, max_y, ny)

    XI: NDArray[np.floating]
    YI: NDArray[np.floating]
    XI, YI = np.meshgrid(xi, yi)

    grid_xy: NDArray[np.floating] = np.column_stack([XI.ravel(), YI.ravel()])

    tree: cKDTree = cKDTree(points[:, :2])
    _, idx = tree.query(grid_xy, k=1)

    Z: NDArray[np.floating] = points[idx, 2]

    return np.column_stack([grid_xy, Z])



RefinementData = typing.Annotated[List[int], AnnotatedScriptType(name='refinement_data', color='aqua', identifier='mesh::RefinementData', controlled='List|R')]

MeshDev = typing.Annotated[
    Tuple[int, int],
    AnnotatedScriptType(name='mesh_division', color='pink', identifier='mesh::MeshDev', controlled='Tuple|M1|M2')
]

# Prepare geological model's results to be used in creating structured meshing
def prepare_surface_vertices_from_geomodel(geomodel_result: StructuralModelResults) -> SimpleNamespace:
    """
    Extract and organize surface mesh vertices from a geological model result
    to be used for creating structure meshing.

    Args:
        geomodel_result (StructuralModelResults): Result object of a geological model containing the structural frame,
            grid definition, and structural elements with associated surface meshes.

    Returns:
        SimpleNamespace:
            An object with the following attributes:
                - extent (array-like):
                    The spatial extent of the model domain
                    [xmin, xmax, ymin, ymax, zmin, zmax].
                - surface_meshes_vertices (list):
                    A list of length three, where:
                        * index 0: empty list (reserved for legacy compatibility)
                        * index 1: empty list (reserved for legacy compatibility)
                        * index 2: list of NumPy arrays containing combined
                          surface vertices for each structural element
    """

    frame = geomodel_result.structural_frame

    combined_vertices: List[np.ndarray] = []

    # colors: List[str] = []  # hex color per element, same order as combined_vertices
    for group in frame.structural_groups:

        for elem in group.structural_elements:
            try:
                verts, _ = elem.get_mesh(GeoMeshType.COMBINED)

            except KeyError:
                continue

            if verts is not None and len(verts) > 0:
                combined_vertices.append(np.asarray(verts))
                # colors.append(elem.color)

    # Legacy code accesses surface_meshes_vertices[2] for the "combined" mesh list.
    # Indices 0 and 1 are padded with empty lists to preserve that indexing.
    return SimpleNamespace(
        extent=frame.grid.extent,
        surface_meshes_vertices=[[], [], combined_vertices],
        # colors=colors,
    )


# Register this function as a component
@wbgeo_component(description='Provides structured mesh',
                 title='Create Structured Mesh',  # The title shown in the GUI
                 color='#cc9999',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Meshing',
                 identifier='create_structured_mesh_data',  # a unique identifier
                 return_name='Mesh',  # the name for the returned-port
                 )  # inputs are handled via the method signature

def create_structured_mesh_data(geomodel_result: StructuralModelResults,
                                refinement_data: RefinementData = (25,21,16,5,6),  #TODO: This needs a proper default
                                z_threshold: float =0.1,
                                mesh_division: typing.Optional[MeshDev] = None,
                                tolerance: float =1,
                                extent: typing.Optional[ExtentData] = None) -> MeshResults:
    """
    Generates a geological mesh and returns a MeshData object.

    Args:
        geomodel_result :Results of geological modeling.
        refinement_data (list): list of refinement values.
        z_threshold (float): Threshold for Z-value adjustment.
        mesh_division (MeshDev): Resolution in x and y directions.
        tolerance (float): Distance tolerance for Z-value adjustment.
        extent (ExtentData): extent of mesh (min_x, max_x, min_y,max_y, min_z, max_z)

    Returns:
        MeshResults:
            MeshResults object containing:
                - nodes: array of shape (n_nodes, 4) with columns
                  [node_id, x, y, z]
                - elements_structured: array of shape (n_elements, 10) with columns
                  [element_id, node0, ..., node7, block_id]
    """

    # Extent
    if extent is None or len(extent) == 0:
        extent = np.asarray(geomodel_result.structural_frame.grid.extent, dtype=float)

    else:
        extent = np.asarray(extent, dtype=float)

    min_x, max_x, min_y, max_y, min_z, max_z = extent
    geomodel_adapter = prepare_surface_vertices_from_geomodel(geomodel_result)  # TODO: HOTFIX — see adapter definition above

    # Create surface grids
    if mesh_division and len(mesh_division) == 2:
        n_gx: int
        n_gy: int
        n_gx, n_gy = mesh_division
        raw_surfaces: Dict[str, NDArray[np.floating]] = create_surface_grid(geomodel_adapter , extent)[0]

        interpolated_surfaces: Dict[str, NDArray[np.floating]] = {
            name: grid
            for name, grid in raw_surfaces.items()
        }

    else:
        interpolated_surfaces, n_gx, n_gy = create_surface_grid(
            geomodel_adapter , extent
        )

    dataframes_list: List[pd.DataFrame]  = [
        pd.DataFrame(grid, columns=["X", "Y", "Z"])
        for grid in interpolated_surfaces.values()
    ]

    # Crop surfaces to extent ONLY
    nx: int
    ny: int
    nx, ny = mesh_division if mesh_division else (n_gx, n_gy)
    cropped_surfaces: List[pd.DataFrame] = []

    for df in dataframes_list:
        pts: NDArray[np.floating] = df.to_numpy()

        mask: NDArray[np.bool_] = (
            (pts[:, 0] >= min_x) & (pts[:, 0] <= max_x) &
            (pts[:, 1] >= min_y) & (pts[:, 1] <= max_y) &
            (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z)
        )

        cropped: NDArray[np.floating] = pts[mask]

        if cropped.size == 0:
            continue

        grid: NDArray[np.floating]  = resample_preserve_z_nearest(cropped, nx, ny, extent)
        cropped_surfaces.append(pd.DataFrame(grid, columns=["X", "Y", "Z"]))

    # Sorting & stacking
    sorted_dataframes: List[pd.DataFrame] = sort_points_by_x_y(cropped_surfaces)
    sorted_surfaces: List[pd.DataFrame] = sort_surfaces_by_z(sorted_dataframes)


    output_array_tuple = store_points_in_array(sorted_surfaces)
    output_array: NDArray[np.floating] = np.asarray(output_array_tuple[0])

    # Create bottom and top surfaces
    bottom_top_surfaces: NDArray[np.floating]  = create_surfaces_with_grids_for_bottom_and_top(
        min_x, max_x, min_y, max_y, min_z, max_z, n_gx, n_gy
    )


    # Create points between surfaces
    updated_output: NDArray[np.floating]  = create_intermediate_layers(
        bottom_top_surfaces, output_array, refinement_data, n_gx, n_gy
    )

    # Adjust Z values based on distance tolerance
    adjusted_array: NDArray[np.floating]  = adjust_z_values(updated_output, n_gx, n_gy, z_threshold, tolerance)

    # Create elements and assign surface IDs to them
    elements: NDArray[np.integer]
    nodes: NDArray[np.floating]
    elements, nodes = create_hexahedral_elements_with_nodes(adjusted_array, n_gx, n_gy)


    # -----------------------------------
    # Convert structured elements to meshio format
    # -----------------------------------

    # Extract connectivity (8 nodes per hex)
    hexa_cells = elements[:, 1:9].astype(int)

    # Extract block IDs (last column)
    hexa_blocks = elements[:, -1].astype(int)

    # Group by block ID
    unique_blocks = np.unique(hexa_blocks)

    cells: List[meshio.CellBlock] = []
    cell_data_list: List[np.ndarray] = []

    for blk in unique_blocks:
        mask = hexa_blocks == blk
        elems_blk = hexa_cells[mask]

        cells.append(meshio.CellBlock("hexahedron", elems_blk))
        cell_data_list.append(np.full(len(elems_blk), blk, dtype=int))

    cell_data = {"block_id": cell_data_list}
    #------------------------------------
    #.Add node boundaries
    #------------------------------------
    nodes_obj = Nodes(nodes)

    point_sets = nodes_obj.nodes_on_boundaries()
    # -----------------------------------
    # Return mesh
    # -----------------------------------

    return MeshResults(
        nodes=nodes[:, 1:4] if nodes.shape[1] == 4 else nodes,  # remove node_id column if present
        elements=cells,
        cell_data=cell_data,
        point_sets=point_sets
    )

