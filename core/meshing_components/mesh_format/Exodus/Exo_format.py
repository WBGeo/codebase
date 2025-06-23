import meshio
from typing import Union, List, Optional
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import pyvista as pv


class ExosInputs:
    def __init__(
    self,
    nodes_array: Union[np.ndarray, List[List[float]]],
    elements_array: Union[np.ndarray, List[meshio.CellBlock]],
    ):

        """
        Initializes the VTMInputs class.

        Args:
            nodes_array: Array of node coordinates.
            elements_block: Either an array of elements with columns
                            [element_id, node_id_1, ..., node_id_n, surface_id]
                            or a list of meshio.CellBlock.
            output_filename: Optional filename for saving/plotting.
        """
        self.nodes_array = np.array(nodes_array, dtype=float)
        self.elements_block = elements_array
        if self.nodes_array.shape[1]  != 3:
            self.nodes = Nodes(node_array=nodes_array)
            self.elements = Elements(element_array=elements_array, node_array=nodes_array)

    def create_mesh(self):
        """
        Creates a mesh using the meshio library and saves it to the specified output file.

        Returns:
            meshio.Mesh: The mesh object.
        """
        if self.nodes_array.shape[1]  == 3:
            # Create the meshio.Mesh object
            mesh = meshio.Mesh(
                points=self.nodes_array,
                cells=self.elements_block
            )


        else:
            # Get the formatted nodes (excluding the first and last columns)
            formatted_nodes = self.nodes.get_coordinates().astype(float)

            # Get elements by surface ID
            elements_by_surface_id = self.elements.element_by_surface_id()

            # Create cells list
            if self.elements.element_array.shape[1] == 10:
                cells = [("hexahedron", elements.tolist()) for elements in elements_by_surface_id.values()]
            else:
                cells = [("tetra", elements.tolist()) for elements in elements_by_surface_id.values()]

            # Get boundary nodes
            nodes_on_boundaries = self.nodes.nodes_on_boundaries()

            # Create the meshio.Mesh object
            mesh = meshio.Mesh(
                points=formatted_nodes,
                cells=cells,
                point_sets=nodes_on_boundaries
            )

        # Write the mesh to an Exodus file
        #mesh.write(self.output_filename, file_format="exodus")
        #print(f"Exodus file '{self.output_filename}' created successfully!")

        return mesh


    def plot_mesh(self):
        """
        Plots the 3D mesh using PyVista.

        This method reads the Exodus file and visualizes the nodes and elements of the mesh.

        """
        # Get node coordinates and elements from the mesh
        mesh = pv.read(self.output_filename)
        # Plot the mesh
        mesh.plot(show_edges=True, color=True)
