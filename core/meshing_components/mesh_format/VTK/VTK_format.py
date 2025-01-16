import meshio
import numpy as np
import pyvista as pv
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes


class VTKInputs:
    def __init__(self, nodes_array, elements_array, output_filename):
        """
        Initializes the VTKInputs class.

        Args:
            nodes_array (np.ndarray): Array of nodes with columns [node_id, x, y, z, surface_id].
            elements_array (np.ndarray): Array of elements with columns [element_id, node_id_1, ..., node_id_n, surface_id].
            output_filename (str): The output filename where the VTK mesh will be saved.
        """
        # Use composition: VTKInputs contains instances of Nodes and Elements
        self.nodes = Nodes(node_array=nodes_array)
        self.elements = Elements(element_array=elements_array, node_array=nodes_array)
        self.output_filename = output_filename


    def create_mesh(self):
        """
        Creates a VTK mesh using the meshio library and saves it to the specified output file.

        Returns:
            meshio.Mesh: The VTK mesh object.
        """
        # Get the formatted nodes (excluding the first and last columns)
        formatted_nodes = self.nodes.get_coordinates().astype(float)

        # Get elements grouped by surface ID
        elements_by_surface_id = self.elements.element_by_surface_id()

        # Create cells list and associate surface_id to each element
        cells = []
        cell_data = {"surface_id": []}  # Dictionary to store surface ID for each cell block

        # Iterate over the elements grouped by surface_id
        for surface_id, elements in elements_by_surface_id.items():
            cells.append(("hexahedron", elements.tolist()))
            cell_data["surface_id"].append([surface_id] * len(elements))  # Surface ID for this block

        # Create the meshio.Mesh object
        mesh = meshio.Mesh(
            points=formatted_nodes,
            cells=cells,
            cell_data=cell_data,
        )

        # Write the mesh to a VTK file
        mesh.write(self.output_filename, file_format="vtk")
        print(f"VTK file '{self.output_filename}' created successfully!")

        return mesh


    def plot_mesh(self):
        """
        Visualizes the VTK mesh using PyVista.

        This method reads the VTK file and visualizes the nodes and elements of the mesh.
        """
        # Load the VTK file using PyVista
        vtk_mesh = pv.read(self.output_filename)
        # Plot the mesh
        vtk_mesh.plot(show_edges=True, color=True)

