import logging
import meshio
from typing import List
import numpy as np
from numpy.typing import NDArray
import io
import os
import tempfile
from py_api_wbgeo.nodesapi import BasicallyABufferedFile
from core.object_components import MeshResults

logger = logging.getLogger(__name__)



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

    def __init__(self, nodes, elements: List[meshio.CellBlock],) -> None:

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
                logger.warning("Skipping unsupported ANSYS cell type: %s", block.type)

        if not cells:
            raise ValueError("No valid element types for ANSYS export")

        mesh: meshio.Mesh = meshio.Mesh(points=points, cells=cells)

        return mesh




class AnsysInputs:
    """
    Class for exporting meshes to ANSYS-compatible format.

    Notes
    -----
    - Supported element types: triangle, quad, tetra, hexahedron, pyramid, wedge.
    - Unsupported element types are skipped with a warning.
    """

    def __init__(self, nodes, elements: List[meshio.CellBlock]) -> None:

        # Normalize nodes to NumPy
        self.nodes = np.asarray(nodes, dtype=float)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be Nx3 coordinates.")

        if not isinstance(elements, list):
            raise TypeError("elements must be List[meshio.CellBlock]")

        self.elements_block: List[meshio.CellBlock] = elements


    def create_mesh(self) -> meshio.Mesh:
        """
        Create a meshio Mesh object compatible with ANSYS.
        """

        points: NDArray[np.float64] = self.nodes


        supported_types: set[str] = {"triangle", "quad", "tetra", "hexahedron", "pyramid", "wedge"}

        cells: List[tuple[str, NDArray[np.int64]]] = []

        for block in self.elements_block:

            if block.type in supported_types:
                cells.append((block.type, block.data))

            else:
                logger.warning("Skipping unsupported ANSYS cell type: %s", block.type)

        if not cells:
            raise ValueError("No valid element types for ANSYS export")

        return meshio.Mesh(points=points, cells=cells)


def export_mesh_results_to_ansys(mesh: MeshResults) -> BasicallyABufferedFile:
    """
    Export a WBGeo MeshResults object to ANSYS (.msh) format using AnsysInputs.
    """

    # Ensure the mesh has nodes and elements
    if not hasattr(mesh, "nodes") or not hasattr(mesh, "elements"):
        raise ValueError("MeshResults must have 'nodes' and 'elements' attributes.")

    # Create ANSYS mesh
    ansys_mesh = AnsysInputs(mesh.nodes, mesh.elements).create_mesh()

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        tmp_path = tmp.name
        ansys_mesh.write(tmp_path)

    # Read into memory buffer
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())
    buf.filename = "mesh_export_ansys.msh"

    # Cleanup temporary file
    os.remove(tmp_path)

    return buf
