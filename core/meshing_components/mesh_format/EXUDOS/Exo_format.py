import meshio
from typing import Union, List, Dict
from numpy.typing import NDArray
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import pyvista as pv
import io
import os
from py_api_wbgeo.nodesapi import wbgeo_component
import tempfile


class ExosInputs:
    """
    Class for exporting unstructured volumetric meshes to ExodusL format.
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



    def create_mesh(self) -> meshio.Mesh:
        """
        Create a meshio Mesh object suitable for Exodus export.


        Returns:
        meshio.Mesh: Mesh object containing points, cell connectivity, and optional point sets for boundary conditions.
        """
        # Create the meshio.Mesh object
        mesh = meshio.Mesh(
            points=self.nodes,
            cells=self.elements
        )
        return mesh



@wbgeo_component(
    title="Download Mesh as Exodus",
    description="Export Mesh to Exodus",
    group="Export",
    identifier="wbgeo::expert_mesh_results_exodus",
)
def export_mesh_results_to_exodus(mesh: "MeshResults") -> io.BytesIO:
    """
    Export a WBGeo MeshResults object to an Exodus (.exo) file.

    The mesh is written to a temporary Exodus file using the internal
    exporter and then returned as an in-memory buffer for download
    through the WBGeo interface.

    Args:
    mesh: MeshResults object containing the mesh to be exported.

    Returns:
    io.BytesIO: In-memory buffer containing the Exodus `.exo` file.
    """
    from core.object_components import Exporters

    # Create exporters from MeshResults
    exporters = Exporters(**mesh.__dict__)

    # Write to a real temporary file (REQUIRED for Exodus)
    with tempfile.NamedTemporaryFile(suffix=".exo", delete=False) as tmp:
        tmp_path = tmp.name
        exporters.export_exodus(tmp_path)

    # Read back into memory
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    # Name for WBGeo download
    buf.filename = "mesh_export_exodus.exo"
    # Remove the temporary file immediately
    os.remove(tmp_path)
    return buf
