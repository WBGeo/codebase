import logging
import meshio
from typing import List, Optional, Dict, Tuple, Set, TYPE_CHECKING
from numpy.typing import NDArray
import numpy as np
from collections import defaultdict
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
from core.object_components import MeshResults

logger = logging.getLogger(__name__)

class STLInputs:
    """
    Class for exporting unstructured volumetric meshes to STL format.

    Notes
    -----
    - Only unstructured meshes with tetrahedral and/or triangular
      elements are supported.
    """
    def __init__(self, nodes, elements: List[meshio.CellBlock],) -> None:

        # Normalize nodes to NumPy
        self.nodes = np.asarray(nodes, dtype=float)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be Nx3 coordinates.")

        if not isinstance(elements, list):
            raise TypeError("elements_array must be List[meshio.CellBlock]")
        self.elements_block: List[meshio.CellBlock] = elements

        self.elements = elements # must be set before writing STL
        self.cell_data: Dict[str, List[List[int]]] = self._generate_cell_data()
        self.output_filename: Optional[str] = None


    def _generate_cell_data(self)-> Dict[str, List[List[int]]]:
        """
        Generate physical group tags for each CellBlock.

        Each CellBlock is assigned a unique physical group ID,
        which is later used to identify surfaces and interfaces.

        Returns:
        dict: Dictionary containing "gmsh:physical" tags per cell block.
        """

        tags_per_block: List[List[int]] = []

        for i, cell_block in enumerate(self.elements_block):

            num_cells = len(cell_block.data)
            tags_per_block.append([i + 1] * num_cells)

        return {"gmsh:physical": tags_per_block}


    def _get_tet_faces(self, tet:  NDArray[np.integer]) -> List[Tuple[int, int, int]]:
        """
        Return the four triangular faces of a tetrahedral element.

        Args:
        tet (ndarray): Array of four node indices defining a tetrahedron.

        Returns:
        list of tuple: List of sorted node-index triplets representing the tetrahedral faces.
        """

        return [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]


    def _extract_and_save_triangle_groups(self) -> Set[int]:
        """
        Extract surface triangle groups and write them to STL files.

        Each physical group of surface triangles is written to a
        separate STL file.

        Returns:
        set of int: Set of node indices that belong to surface triangles.
        """

        triangle_groups: Dict[int, List[ NDArray[np.integer]]] = defaultdict(list)
        triangle_node_set: Set[int] = set()

        # You must define self.cells and self.cell_data before using them here!
        for cell_block, phys_list in zip(self.elements_block, self.cell_data["gmsh:physical"]):

            if cell_block.type == "triangle":

                for tri, phys in zip(cell_block.data, phys_list):
                    triangle_groups[phys].append(tri)

        for phys_id, tris in triangle_groups.items():

            tris:  NDArray[np.integer] = np.array(tris)
            triangle_node_set.update(tris.flatten())

            filename = f"{self.output_filename.replace('.stl','')}_tri_group_{phys_id}.stl"
            meshio.Mesh(points=self.nodes, cells=[("triangle", tris)]).write(filename)

        return triangle_node_set


    def _extract_interface_faces_by_group(self, triangle_node_set: Set[int]) -> Dict[Tuple[int, int], List[Tuple[int, int, int]]]:
        """
        Extract interface faces between tetrahedra of different
        physical groups.

        Args:
        triangle_node_set (set of int): Node indices belonging to external surface triangles.
            These faces are excluded from interface extraction.

        Returns:
        dict: Dictionary mapping pairs of physical group IDs (group1, group2) to lists of triangular interface faces.

        Notes
        -----
        Only faces shared by exactly two tetrahedra with different
        physical tags are considered interfaces.
    """

        tets: List[NDArray[np.integer]] = []
        physical_groups: List[int] = []

        for cell_block, phys_list in zip(self.elements_block, self.cell_data["gmsh:physical"]):

            if cell_block.type == "tetra":
                tets.extend(cell_block.data)
                physical_groups.extend(phys_list)

        face_map: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = defaultdict(list)

        for tet_id, (tet, phys) in enumerate(zip(tets, physical_groups)):

            for face in self._get_tet_faces(tet):
                face_map[face].append((tet_id, phys))

        interface_faces_by_group: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = defaultdict(list)

        for face, entries in face_map.items():

            if len(entries) == 2:
                (_, phys1), (_, phys2) = entries

                if phys1 != phys2 and not all(node in triangle_node_set for node in face):
                    group_pair: Tuple[int, int] = tuple(sorted((phys1, phys2)))
                    interface_faces_by_group[group_pair].append(face)

        return interface_faces_by_group


    def create_mesh(self) -> None:
        """
        Generate STL files for surface and interface geometries.

        For unstructured meshes:
            - Surface triangle groups are exported to STL.
            - Internal interface faces between different physical
            regions are extracted and exported to separate STL files.

        """

        for cell_block in self.elements:

            # If the cell block is hexahedral (8 nodes per element)
            if cell_block.type == "hexahedron":
                raise ValueError(f"STL cannot be created for hexahedral elements ")

        triangle_node_set: Set[int] = self._extract_and_save_triangle_groups()
        interface_faces_by_group = self._extract_interface_faces_by_group(triangle_node_set)

        for (phys1, phys2), faces in interface_faces_by_group.items():

                filename = f"{self.output_filename.replace('.stl','')}_interface_{phys1}_{phys2}.stl"
                face_array :  NDArray[np.integer]= np.array(faces)

                meshio.Mesh(points=self.nodes, cells=[("triangle", face_array)]).write(filename)
                logger.info("Interface faces between group %s and %s written to %s", phys1, phys2, filename)


def export_mesh_results_to_stl(mesh: MeshResults) -> BasicallyABufferedFile:
    """
    Export a WBGeo MeshResults object to STL format.

    The mesh is written to separate STL files for surfaces and interfaces,
    then zipped into a single in-memory archive for download.
    """

    import tempfile
    import zipfile
    import io
    import os

    # Create STLInputs object
    stl_in = STLInputs(nodes=mesh.nodes, elements=mesh.elements)
    mesh_name = getattr(mesh, "name", "mesh").replace(" ", "_")

    # Use temporary directory for STL files
    with tempfile.TemporaryDirectory() as tmp_dir:

        stl_in.output_filename = os.path.join(tmp_dir, f"{mesh_name}.stl")
        stl_in.create_mesh()  # generates STL files in tmp_dir

        # Zip all STL files
        buf = io.BytesIO()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

            for root, _, files in os.walk(tmp_dir):

                for file in files:

                    full_path = os.path.join(root, file)

                    arcname = os.path.relpath(full_path, tmp_dir)

                    zf.write(full_path, arcname)

    buf.seek(0)

    buf.filename = f"{mesh_name}.stl.zip"

    return buf
