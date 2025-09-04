import meshio
from typing import Union, List, Optional
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import pyvista as pv
from dataclasses import dataclass, field
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class C_FeFlowTri:
    nodes: List[int]
    index: int = 0

    def __init__(self, n1=0, n2=0, n3=0, index=0):
        self.nodes = sorted([int(n1), int(n2), int(n3)])
        self.index = index

    def __repr__(self):
        return f"Tri(nodes={self.nodes}, index={self.index})"


@dataclass
class C_FeFlowEdg:
    nodes: List[int]
    index: int = 0

    def __init__(self, n1=0, n2=0, index=0):
        self.nodes = sorted([int(n1), int(n2)])
        self.index = index

    def __repr__(self):
        return f"Edge(nodes={self.nodes}, index={self.index})"


class C_FeFlow:
    def __init__(self):
        self.allTriangles: List[C_FeFlowTri] = []
        self.allTrianglesWithoutDuplicates: List[C_FeFlowTri] = []
        self.undefinedTriangles: List[C_FeFlowTri] = []
        self.definedTriangles: List[C_FeFlowTri] = []

    def generateAllTriangles(self, tetrahedronlist: np.ndarray):
        """
        Equivalent to C++ generateAllTriangles
        """
        self.allTriangles.clear()
        n_tet = len(tetrahedronlist)

        # Step 1: Create all tetra faces (match C++ exactly)
        for t in range(n_tet):
            a, b, c, d = map(int, tetrahedronlist[t])
            faces = [
                (a, b, c),
                (a, b, d),
                (b, c, d),
                (c, a, d),  # match C++ ordering
            ]
            for face in faces:
                self.allTriangles.append(C_FeFlowTri(*face, len(self.allTriangles)))

        # Step 2: Sort by nodes then index
        self.allTriangles.sort(key=lambda tri: (tri.nodes[0], tri.nodes[1], tri.nodes[2], tri.index))

        # Step 3: Deduplicate
        self.allTrianglesWithoutDuplicates.clear()
        for tri in self.allTriangles:
            if (not self.allTrianglesWithoutDuplicates or
                tri.nodes != self.allTrianglesWithoutDuplicates[-1].nodes):
                self.allTrianglesWithoutDuplicates.append(tri)

        # Step 4: Resort by index
        self.allTrianglesWithoutDuplicates.sort(key=lambda tri: tri.index)

        # Step 5: Re-index sequentially
        for new_idx, tri in enumerate(self.allTrianglesWithoutDuplicates):
            tri.index = new_idx

        # Step 6: Final sort by nodes (for fast lookup)
        self.allTrianglesWithoutDuplicates.sort(
            key=lambda tri: (tri.nodes[0], tri.nodes[1], tri.nodes[2], tri.index)
        )

    def generateUndefinedTriangles(self, marker: int, triangle_markers: np.ndarray, triangle_list: np.ndarray):
        """
        Equivalent to C++ generateUndefinedTriangles
        """
        self.undefinedTriangles.clear()
        for t, mark in enumerate(triangle_markers):
            if mark == marker:
                n1, n2, n3 = triangle_list[t]
                self.undefinedTriangles.append(C_FeFlowTri(n1, n2, n3, 0))

        # Sort by nodes + index (explicit, like C++)
        self.undefinedTriangles.sort(
            key=lambda tri: (tri.nodes[0], tri.nodes[1], tri.nodes[2], tri.index)
        )

    def generateDefinedTriangles(self):
        """
        Equivalent to C++ generateDefinedTriangles
        """
        self.definedTriangles.clear()
        all_tri_nodes = [tri.nodes for tri in self.allTrianglesWithoutDuplicates]

        last_pos = 0
        for undef in self.undefinedTriangles:
            undef_nodes_sorted = sorted(undef.nodes)
            for a in range(last_pos, len(all_tri_nodes)):
                if undef_nodes_sorted == all_tri_nodes[a]:
                    self.definedTriangles.append(self.allTrianglesWithoutDuplicates[a])
                    last_pos = a
                    break

        # Final sort by index (like C++)
        self.definedTriangles.sort(key=lambda tri: tri.index)

    def generateMarkerTriangles(self, marker_triangles: np.ndarray):
        if len(marker_triangles) == 0:
            return []

        marker_triangles = np.array(marker_triangles, dtype=int).reshape(-1, 3)

        self.generateUndefinedTriangles(
            marker=0,
            triangle_markers=np.ones(len(marker_triangles), dtype=int),
            triangle_list=marker_triangles
        )
        self.generateDefinedTriangles()
        return [tri.index for tri in self.definedTriangles]


    def generateAllEdges(self, tetrahedronlist):
        self.allEdges = []
        numberoftetrahedra = len(tetrahedronlist)

        print("Generating edges from tetrahedra...")

        for t in range(numberoftetrahedra):
            tet = tetrahedronlist[t]

            base_index = len(self.allEdges)  # keeps original order as index

            edges = [
                [int(tet[0]), int(tet[1])],
                [int(tet[0]), int(tet[2])],
                [int(tet[0]), int(tet[3])],
                [int(tet[1]), int(tet[2])],
                [int(tet[1]), int(tet[3])],
                [int(tet[2]), int(tet[3])]
            ]

            for edge in edges:
                sorted_edge = sorted(edge)
                self.allEdges.append(C_FeFlowEdg(sorted_edge[0], sorted_edge[1], len(self.allEdges)))

        print("Total edges generated (with duplicates):", len(self.allEdges))

        # Sort edges by node values and then original index
        self.allEdges.sort(key=lambda edg: (edg.nodes[0], edg.nodes[1], edg.index))

        self.allEdgesWithoutDuplicates = []
        seen = set()

        for edg in self.allEdges:
            node_tuple = tuple(edg.nodes)
            if node_tuple not in seen:
                self.allEdgesWithoutDuplicates.append(edg)
                seen.add(node_tuple)

        print("Total edges after deduplication:", len(self.allEdgesWithoutDuplicates))

        # Re-index
        for i, edg in enumerate(self.allEdgesWithoutDuplicates):
            edg.index = i

        # Final sort for consistent lookup (optional)
        self.allEdgesWithoutDuplicates.sort(key=lambda edg: (edg.nodes[0], edg.nodes[1]))

        print("Final edges ready:", len(self.allEdgesWithoutDuplicates))


    def generateMarkerEdges(self, marker_edges: np.ndarray):
        marker_edge_indices = []
        edge_set = {tuple(sorted(edge.nodes)): edge.index for edge in self.allEdgesWithoutDuplicates}

        for edge in marker_edges:
            sorted_edge = tuple(sorted(map(int, edge)))
            if sorted_edge in edge_set:
                marker_edge_indices.append(edge_set[sorted_edge])
            else:
                print(f"Warning: Edge {sorted_edge} not found in allEdgesWithoutDuplicates")

        return marker_edge_indices


class FeflowInputs:
    def __init__(
    self,
    nodes_array: Union[np.ndarray, List[List[float]]],
    elements_array: Union[np.ndarray, List[meshio.CellBlock]],
    ):

        """
        Initializes the FeflowInputs class.

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

            points = self.nodes_array


            if points.shape[1] < 3:
                raise ValueError("Node coordinates must be 3D for Abaqus")

            # Filter supported Abaqus types
            cells = []
            for block in self.elements_block:
                if block.type in {"line", "triangle", "quad", "tetra", "hexahedron"}:
                    cells.append((block.type, block.data))
                else:
                    print(f"⚠️ Skipping unsupported Abaqus cell type: {block.type}")


        else:
            # Get the formatted nodes (excluding the first and last columns)
            points = self.nodes.get_coordinates().astype(float)

            # Get elements by surface ID
            elements_by_surface_id = self.elements.element_by_surface_id()

            # Create cells list
            if self.elements.element_array.shape[1] == 10:
                cells = [("hexahedron", elements.tolist()) for elements in elements_by_surface_id.values()]
            else:
                cells = [("tetra", elements.tolist()) for elements in elements_by_surface_id.values()]

            # Create mesh and write
        mesh = meshio.Mesh(points=points, cells=cells)

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
