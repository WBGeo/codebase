import meshio
from typing import Union, List
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import pyvista as pv
from numpy.typing import NDArray
import io
import os
import importlib
import tempfile
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
from typing import Annotated


class AnsysInputs:
    """
    Class for exporting meshes to ANSYS-compatible format.

    This class converts WBGeo mesh data into a `meshio.Mesh`
    suitable for export to ANSYS (.msh) format.
    Supported element types include: triangle, quad, tetra, hexahedron, pyramid and wedge
    Both unstructured and structured meshes are supported.

    Notes
    -----
    - Only element types supported by ANSYS are exported.
    - Unsupported element types are skipped with a warning.

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
        self.elements_block: List[meshio.CellBlock] = elements

        self.elements = elements

    def create_mesh(self) -> meshio.Mesh:
        """
        Create a meshio Mesh object compatible with ANSYS.

        Returns:
        meshio.Mesh: Mesh object ready for export to ANSYS format.

        Raises:
        ValueError: If no supported element types are found for export.
        """


        points : NDArray[np.float64]= self.nodes

        supported_types: str[str] = {"triangle", "quad", "tetra", "hexahedron", "pyramid", "wedge"}

        cells: List[tuple[str, : NDArray[np.intt64]]] = []
        for block in self.elements_block:
            if block.type in supported_types:
                cells.append((block.type, block.data))
            else:
                print(f"⚠️ Skipping unsupported ANSYS cell type: {block.type}")

        if not cells:
            raise ValueError("No valid element types for ANSYS export")

        mesh: meshio.Mesh = meshio.Mesh(points=points, cells=cells)

        return mesh



@wbgeo_component(
    title="Download Mesh as Ansys",
    description="Export Mesh to Ansys",
    group="Export",
    identifier="wbgeo::expert_mesh_results_ansys",
)
def export_mesh_results_to_ansys(mesh) -> BasicallyABufferedFile:
    """
    Export the mesh to ANSYS (.msh) format.
    """
    import importlib
    import tempfile
    import io
    import os

    # Lazy import to avoid circular dependency
    obj_module = importlib.import_module("core.object_components")
    MeshResults = getattr(obj_module, "MeshResults")
    Exporters = getattr(obj_module, "Exporters")

    if not isinstance(mesh, MeshResults):
        raise TypeError(f"Expected a MeshResults instance, got {type(mesh)}")

    exporters = Exporters(**mesh.__dict__)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        tmp_path = tmp.name
        exporters.export_ansys(tmp_path)

    # Read into memory
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    buf.filename = "mesh_export_ansys.msh"

    # Cleanup
    os.remove(tmp_path)

    return buf
