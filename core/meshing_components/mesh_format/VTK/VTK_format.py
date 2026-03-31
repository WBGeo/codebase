import meshio
import pyvista as pv
from typing import Union, List, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import io
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
import tempfile
from core.object_components import MeshResults

# meshio → PyVista cell type mapping
MESHIO_TO_VTK = {
    "vertex": pv.CellType.VERTEX,
    "line": pv.CellType.LINE,
    "triangle": pv.CellType.TRIANGLE,
    "quad": pv.CellType.QUAD,
    "tetra": pv.CellType.TETRA,
    "tetra10": pv.CellType.QUADRATIC_TETRA,
    "hexahedron": pv.CellType.HEXAHEDRON,
    "hexahedron20": pv.CellType.QUADRATIC_HEXAHEDRON,
}


class VTKInputs:
    """
    Build a PyVista UnstructuredGrid for legacy VTK (.vtk) export.
    """

    def __init__(
        self,
        nodes,
        elements: List[meshio.CellBlock],
    ) -> None:

        # Normalize nodes to NumPy
        self.nodes = np.asarray(nodes, dtype=float)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be Nx3 coordinates.")

        if not isinstance(elements, list):
            raise TypeError("elements_array must be List[meshio.CellBlock]")

        self.elements = elements

    def create_mesh(self) -> pv.UnstructuredGrid:
        """
        Create a PyVista UnstructuredGrid suitable for VTK export.

        Returns:
        pyvista.UnstructuredGrid: Single unstructured grid with `RegionId` cell data.

        Raises:
        ValueError: If an unsupported element or cell type is encountered.
        """

        cells = []
        cell_types = []
        region_ids = []

        # 🔑 RegionId comes from CellBlock order
        for region_id, cb in enumerate(self.elements_array, start=1):

            if cb.type not in MESHIO_TO_VTK:
                raise ValueError(f"Unsupported cell type: {cb.type}")

            vtk_type = MESHIO_TO_VTK[cb.type]

            for elem in cb.data:
                cells.append(len(elem))
                cells.extend(elem.tolist())
                cell_types.append(vtk_type)
                region_ids.append(region_id)

        grid = pv.UnstructuredGrid(
            np.asarray(cells, dtype=np.int32),
            np.asarray(cell_types, dtype=np.uint8),
            self.nodes_array,
        )

        # ✅ Physical groups preserved
        grid.cell_data["RegionId"] = np.asarray(region_ids, dtype=np.int32)
        return grid



@wbgeo_component(
    title="Download Mesh as VTK",
    description="Export Mesh to single VTK",
    group="Export",
    identifier="wbgeo::expert_mesh_results_vtk",
)
def export_mesh_results_to_vtk(mesh: MeshResults) -> BasicallyABufferedFile:
    """
    Export the mesh to a single legacy VTK (.vtk) file.

    This exporter Collects all mesh blocks from the WBGeo mesh and ensures each block has a `RegionId` cell array.
    It also afely merges all blocks into one UnstructuredGrid and returns the result as a downloadable VTK file

    Args:
    mesh (MeshResults):
        WBGeo mesh object containing one or more mesh blocks.

    Returns:
    io.BytesIO: In-memory `.vtk` file ready for download.
    """
    from core.object_components import Exporters
    import pyvista as pv
    import tempfile
    import io
    import os
    import numpy as np

    exporters = Exporters(**mesh.__dict__)
    multiblock = exporters.mesh

    if not isinstance(multiblock, pv.MultiBlock):
        raise TypeError("Expected a PyVista MultiBlock mesh")

    grids = []

    for block_id, block in enumerate(multiblock, start=1):

        if block is None or block.n_cells == 0:
            continue

        # 🔑 Ensure RegionId exists
        if "RegionId" not in block.cell_data:
            block = block.copy()
            block.cell_data["RegionId"] = np.full(
                block.n_cells, block_id, dtype=np.int32
            )

        grids.append(block)

    if not grids:
        raise RuntimeError("No valid mesh blocks found")

    # 🔥 SAFE merge (keeps cell data)
    grid = grids[0].copy()
    for g in grids[1:]:
        grid = grid.merge(g, merge_points=False)

    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as tmp:
        tmp_path = tmp.name
        grid.save(tmp_path)

    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    buf.filename = "mesh_export.vtk"
    os.remove(tmp_path)

    return buf

