import meshio
import pyvista as pv
from typing import Union, List, Optional
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes


import meshio
import numpy as np
from collections import defaultdict

class STLInputs:
    def __init__(self, nodes_array, elements_array):
        self.nodes_array = np.array(nodes_array)
        self.elements_block = elements_array
        self.cell_data = self._generate_cell_data()

        self.output_filename = None # You can set this later as well
        # Add other needed attributes here, e.g. self.cells, self.cell_data, self.nodes, etc.
    def _generate_cell_data(self):
        """
        Assigns unique physical group tags to each CellBlock.
        """
        tags_per_block = []
        for i, cell_block in enumerate(self.elements_block):
            num_cells = len(cell_block.data)
            tags_per_block.append([i + 1] * num_cells)
        return {"gmsh:physical": tags_per_block}


    def _get_tet_faces(self, tet):
        return [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]

    def _extract_and_save_triangle_groups(self):
        triangle_groups = defaultdict(list)
        triangle_node_set = set()

        # You must define self.cells and self.cell_data before using them here!
        for cell_block, phys_list in zip(self.elements_block, self.cell_data["gmsh:physical"]):
            if cell_block.type == "triangle":
                for tri, phys in zip(cell_block.data, phys_list):
                    triangle_groups[phys].append(tri)

        for phys_id, tris in triangle_groups.items():
            tris = np.array(tris)
            triangle_node_set.update(tris.flatten())
            filename = f"{self.output_filename.replace('.stl','')}_tri_group_{phys_id}.stl"
            meshio.Mesh(points=self.nodes_array, cells=[("triangle", tris)]).write(filename)

        return triangle_node_set

    def _extract_interface_faces_by_group(self, triangle_node_set):
        tets = []
        physical_groups = []

        for cell_block, phys_list in zip(self.elements_block, self.cell_data["gmsh:physical"]):
            if cell_block.type == "tetra":
                tets.extend(cell_block.data)
                physical_groups.extend(phys_list)

        face_map = defaultdict(list)

        for tet_id, (tet, phys) in enumerate(zip(tets, physical_groups)):
            for face in self._get_tet_faces(tet):
                face_map[face].append((tet_id, phys))

        interface_faces_by_group = defaultdict(list)

        for face, entries in face_map.items():
            if len(entries) == 2:
                (_, phys1), (_, phys2) = entries
                if phys1 != phys2 and not all(node in triangle_node_set for node in face):
                    group_pair = tuple(sorted((phys1, phys2)))
                    interface_faces_by_group[group_pair].append(face)

        return interface_faces_by_group

    def create_mesh(self):
        if self.nodes_array.shape[1] == 3:
            triangle_node_set = self._extract_and_save_triangle_groups()
            interface_faces_by_group = self._extract_interface_faces_by_group(triangle_node_set)
            for (phys1, phys2), faces in interface_faces_by_group.items():
                filename = f"{self.output_filename.replace('.stl','')}_interface_{phys1}_{phys2}.stl"
                face_array = np.array(faces)
                meshio.Mesh(points=self.nodes_array, cells=[("triangle", face_array)]).write(filename)
                print(f"Interface faces between group {phys1} and {phys2} written to {filename}")
        else:
            print('STL cannot be created for structured mesh')



    def plot_mesh(self):
        """
        Visualizes the VTU mesh using PyVista.

        This method reads the VTU file and visualizes the nodes and elements of the mesh.
        """
        # Load the VTU file using PyVista
        vtu_mesh = pv.read(self.output_filename)
        # Plot the mesh
        vtu_mesh.plot(show_edges=True, color=True)

