import numpy as np
import pyvista as pv
import meshio
from typing import Union, List, Optional, Dict
from numpy.typing import NDArray
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes

import importlib
import tempfile
import zipfile
import io
import os
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
                cell_types: NDArray[np.unit8] = np.array(cell_types, dtype=np.uint8)
                grid: pv.UnstructuredGrid = pv.UnstructuredGrid(cells_flat, cell_types, self.nodes)
                multi_block[f"Block_{idx}_{cell_block.type}"] = grid

        # Save the MultiBlock to disk correctly
        if self.output_filename:
            pv.save_meshio(self.output_filename, multi_block)  # saves .vtm and .vtm sub-files

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
    title="Download Mesh as VTM",
    description="Export selected mesh to VTM",
    group="Export",
    identifier="wbgeo::expert_mesh_results_vtm",
)
def export_mesh_results_to_vtm(mesh) -> BasicallyABufferedFile:
    """
    Export the given MeshResults object as a VTK MultiBlock (.vtm) dataset.

    The mesh is first written to a temporary directory using the WBGeo
    Exporters interface. Since a VTM file may consist of multiple
    sub-files, the entire directory is packaged into a ZIP archive
    and returned as a downloadable file.

    Args:
    mesh: WBGeo mesh object containing nodes, elements, and metadata to be exported.

    Returns:
    io.BytesIO: In-memory ZIP archive containing the `.vtm` file and all associated sub-files.

    Notes
    -----
    - The ZIP archive is required because `.vtm` files reference
      additional files stored alongside the main VTM file.
    - The output is compatible with ParaView and PyVista.
    """
    # Dynamically import Exporters and MeshResults
    obj_module = importlib.import_module("core.object_components")
    MeshResults = getattr(obj_module, "MeshResults")
    Exporters = getattr(obj_module, "Exporters")

    # Optional runtime type check
    if not isinstance(mesh, MeshResults):
        raise TypeError(f"Expected a MeshResults instance, got {type(mesh)}")

    # Create exporters from MeshResults
    exporters = Exporters(**mesh.__dict__)

    # Build a safe mesh name
    mesh_name = getattr(mesh, "name", "mesh").replace(" ", "_")

    # Write to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        vtm_path = os.path.join(tmp_dir, f"{mesh_name}.vtm")
        exporters.export_vtm(vtm_path)

        # Package all files into a ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, tmp_dir)
                    zf.write(full_path, arcname)

    buf.seek(0)
    buf.filename = f"{mesh_name}.vtm.zip"
    return buf
