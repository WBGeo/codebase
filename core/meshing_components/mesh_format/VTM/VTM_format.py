import numpy as np
import pyvista as pv
import meshio
from typing import Union, List, Optional, Dict
from numpy.typing import NDArray
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
from typing import Annotated
from core.object_components import MeshResults
import io


from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
class VTMInputs:
    """
    Class for constructing and exporting VTK MultiBlock (.vtm) meshes.

    This class converts node and element information into a PyVista

    Each cell group (cell block or surface ID) is exported as a separate
    block in the MultiBlock dataset, making the output suitable for
    visualization and post-processing in ParaView and PyVista.
    """
    def __init__(
        self,
        nodes,
        elements: List[meshio.CellBlock],
        output_filename: Optional[str] = None,
    ) -> None:

        # Normalize nodes to NumPy
        self.nodes = np.asarray(nodes, dtype=float)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be Nx3 coordinates.")

        if not isinstance(elements, list):
            raise TypeError("elements_array must be List[meshio.CellBlock]")
        self.elements_block: List[meshio.CellBlock] = elements
        self.output_filename: Optional[str] = output_filename

        self.elements = elements



    def _cell_block_type_to_vtk(self, meshio_type: str) -> int:
        """
        Map a meshio cell type to its corresponding VTK cell type ID.

        Args:
        meshio_type (str): Cell type string used by meshio (e.g. 'triangle', 'tetra').

        Returns:
        int: VTK cell type ID.

        Raises:
        ValueError: If the meshio cell type is not supported.
        """
        mapping: Dict[str, int] = {
            "vertex": 1,         # VTK_VERTEX
            "line": 3,           # VTK_LINE
            "triangle": 5,       # VTK_TRIANGLE
            "quad": 9,           # VTK_QUAD
            "tetra": 10,         # VTK_TETRA
            "hexahedron": 12,    # VTK_HEXAHEDRON
            "wedge": 13,         # VTK_WEDGE
            "pyramid": 14,       # VTK_PYRAMID
        }

        if meshio_type not in mapping:
            raise ValueError(f"Unsupported meshio cell type: {meshio_type}")
        return mapping[meshio_type]


    def create_mesh(self) -> pv.MultiBlock:
        """
        Construct a PyVista MultiBlock mesh from the input nodes and elements.

        Returns:
            pv.MultiBlock: MultiBlock dataset containing one UnstructuredGrid per block.
        """
        multi_block: pv.MultiBlock = pv.MultiBlock()

        for idx, cell_block in enumerate(self.elements_block):
            vtk_cell_type: int = self._cell_block_type_to_vtk(cell_block.type)
            cells_flat: List[int] = []
            cell_types: List[int] = []

            for cell in cell_block.data:
                cells_flat.append(len(cell))
                cells_flat.extend(cell)
                cell_types.append(vtk_cell_type)

            cells_flat: NDArray[np.int32] = np.array(cells_flat, dtype=np.int32)
            cell_types: NDArray[np.uint8] = np.array(cell_types, dtype=np.uint8)
            grid: pv.UnstructuredGrid = pv.UnstructuredGrid(cells_flat, cell_types, self.nodes)
            multi_block[f"Block_{idx}_{cell_block.type}"] = grid

        # Save MultiBlock properly (this writes .vtm + referenced .vtu files)
        if self.output_filename:
            multi_block.save(self.output_filename)  # ✅ correct for MultiBlock

        return multi_block


    def plot_mesh(self) -> None:
        """
        Visualize the exported VTM mesh using PyVista.

        The method reads the `.vtm` file from disk and displays it
        with edges enabled.

        Raises:
        ValueError: If no output filename was specified during initialization.
        """
        if not self.output_filename:
            raise ValueError("No output filename specified for plot.")

        mesh: pv.MultiBlock = pv.read(self.output_filename)
        mesh.plot(show_edges=True)

@wbgeo_component(
    title="Download Mesh as VTM ZIP",
    description="Export Mesh to VTM inside a ZIP (like Exporters.export_vtm)",
    group="Export",
    identifier="wbgeo::expert_mesh_results_vtm_zip",
)
def export_mesh_results_to_vtm(mesh: MeshResults) -> BasicallyABufferedFile:
    """
    Export the given MeshResults object as a VTK MultiBlock (.vtm) dataset
    and package it in a ZIP archive, saving exactly like Exporters.export_vtm.

    Args:
        mesh (MeshResults): WBGeo mesh object containing 'nodes' and 'elements'.

    Returns:
        io.BytesIO: In-memory ZIP archive containing the `.vtm` file and all sub-files.
    """
    import tempfile
    import zipfile
    import os

    if not hasattr(mesh, "nodes") or not hasattr(mesh, "elements"):
        raise ValueError("MeshResults must have 'nodes' and 'elements' attributes.")

    # Build a meaningful filename
    mesh_name = getattr(mesh, "name", "mesh").replace(" ", "_")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Full path for the .vtm file
        vtm_path = os.path.join(tmp_dir, f"{mesh_name}.vtm")

        # Create MultiBlock and save to disk (this will save .vtm + referenced .vtu files automatically)
        vtm_mesh = VTMInputs(mesh.nodes, mesh.elements, output_filename=vtm_path)
        vtm_mesh.create_mesh()  # ✅ saves .vtm + sub-files automatically

        # Create ZIP archive of the entire temporary folder
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, tmp_dir)
                    zf.write(full_path, arcname)

        zip_buffer.seek(0)
        zip_buffer.filename = f"{mesh_name}.vtm.zip"
        return zip_buffer
