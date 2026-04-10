import meshio
from typing import List
import numpy as np
import io
import os
import tempfile

from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
from core.object_components import MeshResults
# ⚠️ Avoid top-level import if it creates cycles
# from core.object_components import MeshResults


class ExosInputs:
    """
    Class for exporting unstructured volumetric meshes to Exodus format.
    """

    def __init__(self, nodes, elements: List[meshio.CellBlock]) -> None:
        # Normalize nodes to NumPy
        self.nodes = np.asarray(nodes, dtype=float)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be Nx3 coordinates.")

        if not isinstance(elements, list):
            raise TypeError("elements must be List[meshio.CellBlock]")

        self.elements = elements

    def create_mesh(self) -> meshio.Mesh:
        """
        Create a meshio Mesh object suitable for Exodus export.
        """
        return meshio.Mesh(
            points=self.nodes,
            cells=self.elements
        )


@wbgeo_component(
    title="Download Mesh as Exodus",
    description="Export Mesh to Exodus",
    group="Export",
    identifier="wbgeo::expert_mesh_results_exodus",
)
def export_mesh_results_to_exodus(mesh: MeshResults) -> BasicallyABufferedFile:
    """
    Export a WBGeo MeshResults object to an Exodus (.exo) file.

    The mesh is written to a temporary Exodus file and returned
    as an in-memory buffer for download.
    """

    exo_in = ExosInputs(mesh.nodes, mesh.elements)
    meshio_mesh = exo_in.create_mesh()

    # Write to temporary file (Exodus requires real file)
    with tempfile.NamedTemporaryFile(suffix=".exo", delete=False) as tmp:
        tmp_path = tmp.name
        meshio_mesh.write(tmp_path, file_format="exodus")

    # Read file into memory buffer
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    buf.filename = "mesh_export_exodus.exo"

    # Clean up temporary file
    os.remove(tmp_path)

    return buf
