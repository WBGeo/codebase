import numpy as np
import pyvista as pv
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes



class VTMInputs:

    def __init__(self, nodes_array, elements_array):
        """
        Initializes the VTMInputs class.

        Args:
            nodes_array (np.ndarray): Array of nodes with columns [node_id, x, y, z, surface_id].
            elements_array (np.ndarray): Array of elements with columns [element_id, node_id_1, ..., node_id_n, surface_id].
            output_filename (str): The output filename where the VTM mesh will be saved.
        """
        # Use composition: VTMInputs contains instances of Nodes and Elements
        self.nodes = Nodes(node_array=nodes_array)
        self.elements = Elements(element_array=elements_array, node_array=nodes_array)

    def create_mesh(self):
        """
        Creates a VTM file with multiple blocks for a structured grid,
        where each block corresponds to a unique surface ID.

        Returns:
            pv.MultiBlock: The PyVista MultiBlock object.
        """
        # Get the formatted nodes
        formatted_nodes = self.nodes.get_coordinates().astype(float)

        # Get elements grouped by surface ID
        elements_by_surface_id = self.elements.element_by_surface_id()

        # Create a PyVista MultiBlock object to hold all blocks
        multi_block = pv.MultiBlock()

        for surface_id, elements in elements_by_surface_id.items():
            # Create cell connectivity and types
            cells = []
            cell_types = []

            for surface_id, elements in elements_by_surface_id.items():

                cells = []
                cell_types = []

                for element in elements:
                    cell = [8] + (element ).tolist()
                    cells.extend(cell)
                    cell_types.append(12)  # VTK_HEXAHEDRON is type 12

                # Convert to numpy arrays
                cells_flat = np.array(cells, dtype=np.int32)
                cell_types = np.array(cell_types, dtype=np.uint8)

                # Validation: Check expected size
                expected_size = len(elements) * 9  # 8 points + 1 count per cell
                if len(cells_flat) != expected_size:
                    raise ValueError(
                        f"Mismatch in cell array size for surface {surface_id}. "
                        f"Expected {expected_size}, got {len(cells_flat)}."
                    )

                # Create the PyVista UnstructuredGrid
                grid = pv.UnstructuredGrid(
                    cells_flat,
                    cell_types,
                    formatted_nodes
                )

                # Add the grid to the MultiBlock dataset
                multi_block[f"Surface_ID_{surface_id}"] = grid


        # Save the MultiBlock as a VTM file
        # multi_block.save(self.output_filename)
        # print(f"VTM file '{self.output_filename}' with multiple blocks created successfully!")

        return multi_block


    def plot_mesh(self):
        """
        Visualizes the VTM mesh using PyVista.

        This method reads the VTM file and visualizes the nodes and elements of the mesh.
        """
        # Load the VTM file using PyVista
        vtm_mesh = pv.read(self.output_filename)
        # Plot the mesh
        vtm_mesh.plot(show_edges=True, color=True)


