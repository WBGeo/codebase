import numpy as np
import typing
from typing import List
import meshio
from core.object_components import MeshResults, StructuralModelResults, ExtentData
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
from numpy.typing import NDArray
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType, wbgeo_type, wbgeo_inspector, \
  InspectorHelper
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d


def classify_boundary_nodes(nodes, extent, tol_ratio=1e-4):

    nodes = np.asarray(nodes)

    xmin, xmax, ymin, ymax, zmin, zmax = extent

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    tolx = dx * tol_ratio
    toly = dy * tol_ratio
    tolz = dz * tol_ratio

    boundary_groups = {
        "left": [],
        "right": [],
        "front": [],
        "back": [],
        "bottom": [],
        "top": [],
    }

    for i, (x, y, z) in enumerate(nodes):

        if abs(x - xmin) <= tolx:
            boundary_groups["left"].append(i)

        if abs(x - xmax) <= tolx:
            boundary_groups["right"].append(i)

        if abs(y - ymin) <= toly:
            boundary_groups["front"].append(i)

        if abs(y - ymax) <= toly:
            boundary_groups["back"].append(i)

        if abs(z - zmin) <= tolz:
            boundary_groups["bottom"].append(i)

        if abs(z - zmax) <= tolz:
            boundary_groups["top"].append(i)

    return boundary_groups

def build_point_sets(boundary_groups, n_nodes):

    point_sets = {}

    for name, node_ids in boundary_groups.items():
        if len(node_ids) == 0:
            continue

        arr = np.asarray(node_ids, dtype=int)

        # optional safety check
        if np.any(arr >= n_nodes):
            raise ValueError(f"{name} contains invalid node index")

        point_sets[name] = arr

    return point_sets




@wbgeo_component(
    description='Provides structured implicit mesh as unstructured',
    title='Create Structured Implicit Mesh',
    color='#cc9999',
    border_color='#000000',
    group='Meshing',
    identifier='create_structured_implicit_mesh_meshio',
    return_name='Mesh',
)
def create_implicit_structured_mesh(geomodel_result: StructuralModelResults, extent: ExtentData = []) -> MeshResults:
    """
    Create a structured hexahedral mesh and return as meshio points, cells, and cell_data.
    Block IDs are reversed so the highest block becomes 0.
    """

    # Get extent
    if extent is None or len(extent) == 0:
        extent_arr = np.asarray(geomodel_result.structural_frame.grid.extent, dtype=float)
    else:
        extent_arr = np.asarray(extent, dtype=float)
    xmin, xmax, ymin, ymax, zmin, zmax = extent_arr

    # Resolution
    nx, ny, nz = geomodel_result.structural_frame.grid.resolution

    # Reshape grid + block IDs
    grid: NDArray[np.floating] = np.asarray(geomodel_result.structural_frame.grid.grid_coordinates).reshape((nx, ny, nz, 3))
    block_ids: NDArray[np.integer] = np.asarray(geomodel_result.structural_frame.lith_block).reshape((nx, ny, nz))

    # Mask nodes inside extent
    inside: NDArray[np.bool_] = (
        (grid[..., 0] >= xmin) & (grid[..., 0] <= xmax) &
        (grid[..., 1] >= ymin) & (grid[..., 1] <= ymax) &
        (grid[..., 2] >= zmin) & (grid[..., 2] <= zmax)
    )

    # Map old node indices to new node IDs
    old_to_new: NDArray[np.integer] = -np.ones((nx, ny, nz), dtype=int)
    kept_nodes: NDArray[np.integer] = np.argwhere(inside)
    for new_id, (i, j, k) in enumerate(kept_nodes):
        old_to_new[i, j, k] = new_id

    # Keep node coordinates only
    points: NDArray[np.floating] = grid[inside]

    # Build hexahedral elements
    hexa_cells: List[List[int]] = []
    hexa_blocks: List[int] = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corners = [
                    (i, j, k),
                    (i+1, j, k),
                    (i+1, j+1, k),
                    (i, j+1, k),
                    (i, j, k+1),
                    (i+1, j, k+1),
                    (i+1, j+1, k+1),
                    (i, j+1, k+1),
                ]
                if not all(inside[c] for c in corners):
                    continue

                node_ids = [old_to_new[c] for c in corners]

                # Assign block ID based on central node
                block = int(block_ids[i, j, k])
                hexa_cells.append(node_ids)
                hexa_blocks.append(block)

    hexa_cells = np.array(hexa_cells, dtype=int)
    hexa_blocks = np.array(hexa_blocks, dtype=int)

    # Reverse block IDs: max_block becomes 0
    #max_block = hexa_blocks.max()
    #hexa_blocks = max_block - hexa_blocks

    # Group elements by block ID
    unique_blocks = np.unique(hexa_blocks)
    cells: List[meshio.CellBlock] = []
    cell_data_list: List[np.ndarray] = []

    for blk in unique_blocks:
        mask = hexa_blocks == blk
        elems_blk = hexa_cells[mask]
        cells.append(meshio.CellBlock("hexahedron", elems_blk))
        cell_data_list.append(np.full(len(elems_blk), blk, dtype=int))

    cell_data = {"block_id": cell_data_list}
    # -----------------------------------
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)

    extent_mesh = np.array([x_min, x_max, y_min, y_max, z_min, z_max])

    boundary_groups_n = classify_boundary_nodes(points, extent_mesh)
    point_sets = build_point_sets(boundary_groups_n, len(points))

    return MeshResults(
        nodes=points,
        elements=cells,
        cell_data=cell_data,
        point_sets=point_sets
    )

@wbgeo_component(identifier='wbgeo::inspect_implicit_mesh_3d',
                 title='Plot Implicit Mesh in 3D',
                 description='...')
@wbgeo_inspector()
async def inspect_implicit_mesh_3d(
    mesh_results: MeshResults, _inspector: InspectorHelper):
  # load structural_model_result from the execution trace
  structural_model_result = await (await _inspector.trace(StructuralModelResults)).get_value()
  # and call the render function with both the mesh_results and the object from our trace
  plot_mesh_3d(mesh_results, structural_model_result, "surface", True)
