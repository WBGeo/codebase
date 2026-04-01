import meshio
from typing import Union, List, Dict
from numpy.typing import NDArray
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import pyvista as pv
import io
import os
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
import tempfile
import importlib
from typing import Annotated


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







def export_mesh_results_to_exodus(mesh) -> BasicallyABufferedFile:
    """
    Export a WBGeo MeshResults object to an Exodus (.exo) file.
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

    with tempfile.NamedTemporaryFile(suffix=".exo", delete=False) as tmp:
        tmp_path = tmp.name
        exporters.export_exodus(tmp_path)

    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    buf.filename = "mesh_export_exodus.exo"
    os.remove(tmp_path)
    return buf
