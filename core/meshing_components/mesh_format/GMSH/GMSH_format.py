import meshio
import pyvista as pv
from typing import Union, List, Optional
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes



class GMSHInputs:
    def __init__(self, nodes_array, elements_array):
        self.nodes_array = np.array(nodes_array)
        self.elements_block = elements_array
        self.cell_data = self._generate_cell_data()

        self.output_filename = None # You can set this later as well

    @classmethod
    def from_msh(cls, filename):
        """
        Create a GMSHInputs object by reading a GMSH .msh file.

        Args:
            filename (str): Path to the .msh file

        Returns:
            GMSHInputs: Instance populated with mesh data
        """
        mesh = meshio.read(filename)

        # mesh.cell_data is a dict: { "gmsh:physical": [...], "gmsh:geometrical": [...] }
        physical = mesh.cell_data.get("gmsh:physical", [])
        geometrical = mesh.cell_data.get("gmsh:geometrical", [])

        cell_data = {
            "gmsh:physical": physical,
            "gmsh:geometrical": geometrical
        }

        return cls(
            nodes_array=mesh.nodes_array,
            elements_blocks=mesh.elements_block,
            output_filename=filename,
            cell_data=cell_data
        )

    def _generate_cell_data(self):
        """
        Assigns unique physical group tags to each CellBlock and returns a dictionary
        with 'gmsh:physical' and an empty 'gmsh:geometrical'.
        """
        physical_tags = []
        for i, cell_block in enumerate(self.elements_block):
            num_cells = len(cell_block.data)
            physical_tags.append(np.full(num_cells, i + 1, dtype=int))

        return {
            "gmsh:physical": physical_tags,
            "gmsh:geometrical": physical_tags
        }

    def create_mesh(self):
        """
        Creates a GMSH mesh using the meshio library and saves it to the specified output file.
        """
        if self.nodes_array.shape[1]  == 3:

            n_points = self.nodes_array.shape[0]
            dim_tags = np.zeros((n_points, 2), dtype=int)
            node_tag_map = {}
            gmsh_element_dimensions = {
                "vertex": 0,
                "line": 1,
                "triangle": 2,
                "quad": 2,
                "tetra": 3,
                "hexahedron": 3,
            }

            for cell_block_index, cell_block in enumerate(self.elements_block):
                tag = self.cell_data['gmsh:physical'][cell_block_index][0]
                element_type = cell_block.type
                dim = gmsh_element_dimensions.get(element_type)

                if dim is None:
                    raise ValueError(f"Unsupported element type: {element_type}")

                node_indices = np.unique(cell_block.data.flatten())
                for node_id in node_indices:
                    dim_tags[node_id] = [dim, tag]

            mesh = meshio.Mesh(
                points=self.nodes_array,
                cells=[(cb.type, cb.data) for cb in self.elements_block],
                cell_data=self.cell_data,
                point_data={"gmsh:dim_tags": dim_tags}
            )
            return mesh
        else:
            return None




    def plot_mesh(self):
        """
        Visualizes the VTU mesh using PyVista.

        This method reads the VTU file and visualizes the nodes and elements of the mesh.
        """
        # Load the VTU file using PyVista
        vtu_mesh = pv.read(self.output_filename)
        # Plot the mesh
        vtu_mesh.plot(show_edges=True, color=True)

