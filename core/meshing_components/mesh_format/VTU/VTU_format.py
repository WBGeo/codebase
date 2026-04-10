import meshio
import numpy as np
import tempfile
import io
import os
from typing import List
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
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
    Export a MeshResults object to a VTU (.vtu) file.
    """
    if not hasattr(mesh, "nodes") or not hasattr(mesh, "elements"):
        raise ValueError("MeshResults must have 'nodes' and 'elements'")

    # Create meshio.Mesh from VTUInputs
    vtu_mesh = VTUInputs(mesh.nodes, mesh.elements)
    meshio_mesh = vtu_mesh.create_mesh()

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as tmp:
        tmp_path = tmp.name
        meshio_mesh.write(tmp_path)

    # Read back into memory
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())
    buf.filename = "mesh_export.vtu"

    # Cleanup
    os.remove(tmp_path)
    return buf
