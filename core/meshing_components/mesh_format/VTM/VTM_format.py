import numpy as np
import pyvista as pv
import meshio
from typing import Union, List, Optional

from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes


class VTMInputs:
    def __init__(
    self,
    nodes_array: Union[np.ndarray, List[List[float]]],
    elements_array: Union[np.ndarray, List[meshio.CellBlock]],
    output_filename: Optional[str] = None,
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
        self.output_filename = output_filename
        if self.nodes_array.shape[1]  != 3:
            self.nodes = Nodes(node_array=nodes_array)
            self.elements = Elements(element_array=elements_array, node_array=nodes_array)

    def create_mesh(self) -> pv.MultiBlock:
        """
        Creates a VTM (MultiBlock) mesh using PyVista.

        Returns:
            pv.MultiBlock: Multi-block mesh grouped by surface ID or cell block.
        """
        multi_block = pv.MultiBlock()
        # Case 1: meshio.CellBlock list (VTU-like input)
        if self.nodes_array.shape[1]  == 3:
            for idx, cell_block in enumerate(self.elements_block):
                vtk_cell_type = self._cell_block_type_to_vtk(cell_block.type)
                cells_flat = []
                cell_types = []

                for cell in cell_block.data:
                    cells_flat.append(len(cell))
                    cells_flat.extend(cell)
                    cell_types.append(vtk_cell_type)

                cells_flat = np.array(cells_flat, dtype=np.int32)
                cell_types = np.array(cell_types, dtype=np.uint8)
                grid = pv.UnstructuredGrid(cells_flat, cell_types, self.nodes_array)
                multi_block[f"Block_{idx}_{cell_block.type}"] = grid

        # Case 2: elements array with surface ID in last column
        else:
            formatted_nodes = self.nodes.get_coordinates().astype(float)
            elements_by_surface_id = self.elements.element_by_surface_id()

            multi_block = pv.MultiBlock()
            elements = self.elements_block
            surface_ids = np.unique(elements[:, -1])

            for surface_id, elements in elements_by_surface_id.items():
                cells = []
                cell_types = []

                if self.elements.element_array.shape[1] == 10:
                    for element in elements:
                        cell = [8] + element.tolist()  # 8-node hexahedron
                        cells.extend(cell)
                        cell_types.append(12)  # VTK_HEXAHEDRON is type 12

                    expected_size = len(elements) * 9  # 8 points + 1 count per cell

                else:
                    for element in elements:
                        cell = [4] + element.tolist()  # 4-node tetrahedron
                        cells.extend(cell)
                        cell_types.append(10)  # VTK_TETRA is type 10

                    expected_size = len(elements) * 5  # 4 points + 1 count per cell

                cells_flat = np.array(cells, dtype=np.int32)
                cell_types = np.array(cell_types, dtype=np.uint8)

                if len(cells_flat) != expected_size:
                    raise ValueError(
                        f"Mismatch in cell array size for surface {surface_id}. "
                        f"Expected {expected_size}, got {len(cells_flat)}."
                    )

                grid = pv.UnstructuredGrid(cells_flat, cell_types, formatted_nodes)
                multi_block[f"Surface_ID_{surface_id}"] = grid  # Ensure all blocks are added



        return multi_block

    def _cell_block_type_to_vtk(self, cell_type_str: str) -> int:
        """
        Maps meshio cell type string to VTK cell type code.

        Args:
            cell_type_str (str): meshio cell type, e.g. 'tetra', 'hexahedron'

        Returns:
            int: Corresponding VTK cell type.
        """
        vtk_type_map = {
            'tetra': 10,
            'hexahedron': 12,
            'triangle': 5,
            'quad': 9,
            'line': 3,
        }
        return vtk_type_map.get(cell_type_str, 0)

    def plot_mesh(self):
        """
        Visualizes the VTM mesh using PyVista.
        """
        if not self.output_filename:
            raise ValueError("No output filename specified for plot.")

        mesh = pv.read(self.output_filename)
        mesh.plot(show_edges=True)

