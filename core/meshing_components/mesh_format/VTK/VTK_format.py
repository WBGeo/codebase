import meshio
import pyvista as pv
from typing import Union, List, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import numpy as np
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
import pyvista as pv
from typing import Annotated
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

    def __init__(self, nodes, elements: List[meshio.CellBlock]) -> None:
        # Normalize nodes to NumPy
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be Nx3 coordinates.")
        if not isinstance(elements, list):
            raise TypeError("elements must be List[meshio.CellBlock]")

        self.elements = elements  # Use this consistently

    def create_mesh(self) -> pv.UnstructuredGrid:
        """
        Create a PyVista UnstructuredGrid suitable for VTK export.
        """
        cells = []
        cell_types = []
        region_ids = []

        # 🔑 RegionId comes from CellBlock order
        for region_id, cb in enumerate(self.elements, start=1):  # use self.elements here
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
            self.nodes  # use self.nodes here
        )

        # ✅ Physical groups preserved
        grid.cell_data["RegionId"] = np.asarray(region_ids, dtype=np.int32)
        return grid


# We have one singular export component now
# @wbgeo_component(
#     title="Download Mesh as VTK",
#     description="Export Mesh to single VTK",
#     group="Export",
#     identifier="wbgeo::expert_mesh_results_vtk",
# )
def export_mesh_results_to_vtk(mesh: MeshResults) -> BasicallyABufferedFile:
    """
    Export a WBGeo MeshResults object to a legacy VTK (.vtk) file
    using the VTKInputs class.

    The mesh is written to a temporary file and returned as an in-memory buffer.
    """
    import tempfile
    import io
    import os

    # Create VTKInputs object
    vtk_in = VTKInputs(nodes=mesh.nodes, elements=mesh.elements)
    grid = vtk_in.create_mesh()

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as tmp:
        tmp_path = tmp.name
        grid.save(tmp_path)

    # Read back into memory buffer
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    buf.filename = "mesh_export.vtk"

    # Cleanup
    os.remove(tmp_path)

    return buf
