import meshio
import pyvista as pv
from typing import Union, List, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray
import os
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import io
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
import tempfile
from core.object_components import MeshResults


class VTUInputs:
    """
    Class for constructing VTU (.vtu) meshes using meshio.
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
        Creates a VTU mesh using the meshio library and saves it to the specified output file.

        Returns:
            meshio.Mesh: The VTU mesh object.
        """
        tags_per_block: List[List[int]] = []
        for cell_block in self.elements_block:
            num_cells:int  = len(cell_block.data)
            tag: int  = len(tags_per_block) + 1
            tags_per_block.append([tag] * num_cells)
        mesh = meshio.Mesh(
            points=self.nodes,
            cells=[(cb.type, cb.data) for cb in self.elements_block],
            cell_data={"gmsh:physical": tags_per_block}
        )
        return mesh


@wbgeo_component(
    title="Download Mesh as VTU",
    description="Export Mesh to VTU",
    group="Export",
    identifier="wbgeo::expert_mesh_results_vtu",
)
def export_mesh_results_to_vtu(mesh: MeshResults) -> BasicallyABufferedFile:
    """
    Export the given MeshResults object to a VTU (.vtu) file.

    The mesh is written to a temporary file using the WBGeo Exporters
    interface (required by VTU), then read back into memory and returned
    as a downloadable file.

    Args:
    mesh (MeshResults): WBGeo mesh object containing nodes, elements, and metadata to be exported.

    Returns:
    io.BytesIO: In-memory buffer containing the VTU file, ready for download.

    Notes
    -----
    - Supports both structured and unstructured meshes.
    - The exported VTU file is suitable for visualization in ParaView
      and PyVista.
    """
    from core.object_components import Exporters

    # Create exporters from MeshResults
    exporters = Exporters(**mesh.__dict__)

    # Write to a real temporary file (REQUIRED for VTU)
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as tmp:
        tmp_path = tmp.name
        exporters.export_vtu(tmp_path)

    # Read back into memory
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    # Name for WBGeo download
    buf.filename = "mesh_export_vtu.vtu"
    # Remove the temporary file immediately
    os.remove(tmp_path)
    return buf
