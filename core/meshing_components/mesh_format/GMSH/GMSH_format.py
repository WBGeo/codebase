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
        Assigns physical group tags and returns a dict with 'gmsh:physical' and
        'gmsh:geometrical'.

        For unstructured meshes (elements_block is a list of CellBlock): each block
        gets a sequential tag (1-based).
        For structured meshes (elements_block is an ndarray): elements are grouped by
        surface_id (last column) and each group's tag is surface_id + 1.
        """
        if isinstance(self.elements_block, np.ndarray):
            # Structured mesh: group by surface_id stored in the last column
            surface_ids = self.elements_block[:, -1].astype(int)
            physical_tags = []
            for sid in np.unique(surface_ids):
                n_cells = int((surface_ids == sid).sum())
                physical_tags.append(np.full(n_cells, sid + 1, dtype=int))
        else:
            # Unstructured mesh: one CellBlock per group
            physical_tags = []
            for i, cell_block in enumerate(self.elements_block):
                num_cells = len(cell_block.data)
                physical_tags.append(np.full(num_cells, i + 1, dtype=int))

        return {
            "gmsh:physical": physical_tags,
            "gmsh:geometrical": physical_tags,
        }

    def create_mesh(self):
        """
        Creates a meshio Mesh ready to be written in GMSH format.

        Supports two mesh types determined by nodes_array shape:
          - shape (M, 3): unstructured mesh; elements_block is a list of CellBlock.
          - shape (M, 4): structured hexahedral mesh; elements_block is an ndarray
            with columns [elem_id, n0..n7, surface_id].
        """
        if self.nodes_array.shape[1] == 3:
            # ── Unstructured path ─────────────────────────────────────────────
            n_points = self.nodes_array.shape[0]
            dim_tags = np.zeros((n_points, 2), dtype=int)
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

            return meshio.Mesh(
                points=self.nodes_array,
                cells=[(cb.type, cb.data) for cb in self.elements_block],
                cell_data=self.cell_data,
                point_data={"gmsh:dim_tags": dim_tags},
            )

        else:
            # ── Structured hexahedral path ────────────────────────────────────
            # nodes_array: (M, 4) — [node_id, x, y, z]
            # elements_block: (N, 10) — [elem_id, n0..n7, surface_id]
            xyz = self.nodes_array[:, 1:4]                        # (M, 3)
            conn = self.elements_block[:, 1:-1].astype(int)       # (N, 8)
            surface_ids = self.elements_block[:, -1].astype(int)  # (N,)

            cells = []
            cell_data_physical = []
            for sid in np.unique(surface_ids):
                mask = surface_ids == sid
                block_data = conn[mask]
                cells.append(("hexahedron", block_data))
                cell_data_physical.append(
                    np.full(block_data.shape[0], sid + 1, dtype=int)
                )

            # All nodes belong to 3-D hexahedral entities
            dim_tags = np.full((xyz.shape[0], 2), [3, 1], dtype=int)

            return meshio.Mesh(
                points=xyz,
                cells=cells,
                cell_data={
                    "gmsh:physical": cell_data_physical,
                    "gmsh:geometrical": cell_data_physical,
                },
                point_data={"gmsh:dim_tags": dim_tags},
            )




    def plot_mesh(self):
        """
        Visualizes the VTU mesh using PyVista.

        This method reads the VTU file and visualizes the nodes and elements of the mesh.
        """
        # Load the VTU file using PyVista
        vtu_mesh = pv.read(self.output_filename)
        # Plot the mesh
        vtu_mesh.plot(show_edges=True, color=True)

