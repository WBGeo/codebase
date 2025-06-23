import meshio
import pyvista as pv
from typing import Union, List, Optional
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes


class VTUInputs:
    def __init__(
    self,
    nodes_array: Union[np.ndarray, List[List[float]]],
    elements_array: Union[np.ndarray, List[meshio.CellBlock]],
    ):
        """
            Initializes the VTUInputs class.

            Args:
                nodes_array (np.ndarray): Array of nodes with columns [node_id, x, y, z, surface_id].
                elements_array (np.ndarray): Array of elements with columns [element_id, node_id_1, ..., node_id_n, surface_id].
        """
        self.nodes_array = np.array(nodes_array)
        self.elements_block = elements_array
        if self.nodes_array.shape[1]  != 3:
            self.nodes = Nodes(node_array=nodes_array)
            self.elements = Elements(element_array=elements_array, node_array=nodes_array)


    def create_mesh(self):
        """
        Creates a VTU mesh using the meshio library and saves it to the specified output file.

        Returns:
            meshio.Mesh: The VTU mesh object.
        """
        if self.nodes_array.shape[1]  == 3:
            tags_per_block = []
            for cell_block in self.elements_block:
                num_cells = len(cell_block.data)
                tag = len(tags_per_block) + 1
                tags_per_block.append([tag] * num_cells)
            mesh = meshio.Mesh(
                points=self.nodes_array,
                cells=[(cb.type, cb.data) for cb in self.elements_block],
                cell_data={"gmsh:physical": tags_per_block}
            )
        else:
            # Get the formatted nodes (excluding the first and last columns)
            formatted_nodes = self.nodes.get_coordinates().astype(float)

            # Get elements grouped by surface ID
            elements_by_surface_id = self.elements.element_by_surface_id()

            # Create cells list and associate surface_id to each element
            cells = []
            cell_data = {"surface_id": []}  # Dictionary to store surface ID for each cell block

            # Iterate over the elements grouped by surface_id
            if self.elements.element_array.shape[1]== 10:
                for surface_id, elements in elements_by_surface_id.items():
                    cells.append(("hexahedron", elements.tolist()))
                    cell_data["surface_id"].append([surface_id] * len(elements))  # Surface ID for this block
            else:
                for surface_id, elements in elements_by_surface_id.items():
                    cells.append(("tetra", elements.tolist()))
                    cell_data["surface_id"].append([surface_id] * len(elements))  # Surface ID for this block

            # Create the meshio.Mesh object
            mesh = meshio.Mesh(
                points=formatted_nodes,
                cells=cells,
                cell_data=cell_data,
            )

        # Write the mesh to a VTU file
        # mesh.write(self.output_filename, file_format="vtU")
        # print(f"VTU file '{self.output_filename}' created successfully!")

        return mesh


    def plot_mesh(self):
        """
        Visualizes the VTU mesh using PyVista.

        This method reads the VTU file and visualizes the nodes and elements of the mesh.
        """
        # Load the VTU file using PyVista
        vtu_mesh = pv.read(self.output_filename)
        # Plot the mesh
        vtu_mesh.plot(show_edges=True, color=True)

