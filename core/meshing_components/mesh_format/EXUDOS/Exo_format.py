import meshio
from typing import List, Optional, Dict
import numpy as np
import io
import os
import tempfile

from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
from core.object_components import MeshResults


class ExosInputs:
    """
    Class for exporting unstructured volumetric meshes to Exodus format.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: List[meshio.CellBlock],
        cell_data: Optional[Dict[str, List[np.ndarray]]] = None,
        point_sets: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        # --- nodes ---
        self.nodes = np.asarray(nodes, dtype=float)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be Nx3 coordinates.")

        # --- elements ---
        if not isinstance(elements, list):
            raise TypeError("elements must be List[meshio.CellBlock]")

        self.elements = elements

        # --- optional data ---
        self.cell_data = cell_data if cell_data is not None else {}
        self.point_sets = point_sets if point_sets is not None else {}


    # =====================================================
    # 🧱 Mesh creation
    # =====================================================
    def create_mesh(self) -> meshio.Mesh:
        """
        Create a meshio Mesh object suitable for Exodus export.
        """
        return meshio.Mesh(
            points=self.nodes,
            cells=self.elements,
            point_sets=self.point_sets,
            #cell_data=self.cell_data
        )


# =========================================================
# 🚀 WBGeo Export Component
# =========================================================
@wbgeo_component(
    title="Download Mesh as Exodus",
    description="Export Mesh to Exodus",
    group="Export",
    identifier="wbgeo::expert_mesh_results_exodus",
)
def export_mesh_results_to_exodus(mesh: MeshResults) -> BasicallyABufferedFile:
    """
    Export a WBGeo MeshResults object to an Exodus (.exo) file.
    """

    exo_in = ExosInputs(
        mesh.nodes,
        mesh.elements,
        point_sets=mesh.point_sets,
        #cell_data=mesh.cell_data
    )

    meshio_mesh = exo_in.create_mesh()

    # --- Exodus requires real file ---
    with tempfile.NamedTemporaryFile(suffix=".exo", delete=False) as tmp:
        tmp_path = tmp.name
        meshio_mesh.write(tmp_path, file_format="exodus")

    # --- Load into memory ---
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    buf.filename = "mesh_export_exodus.exo"

    # --- Cleanup ---
    os.remove(tmp_path)

    return buf
