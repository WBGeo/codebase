import meshio
from typing import Union, List, Optional
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import pyvista as pv
from dataclasses import dataclass, field

@dataclass(order=True)
class C_FeFlowEdg:
    index: int = field(compare=False)
    nodes: List[int] = field(default_factory=lambda: [0, 0])

    def __init__(self, n1=0, n2=0, index=0):
        self.nodes = sorted([n1, n2])
        self.index = index

@dataclass(order=True)
class C_FeFlowTri:
    def __init__(self, n1, n2, n3, base_index):
        # Ensure nodes are sorted integers (not NumPy types)
        self.nodes = sorted([int(n1), int(n2), int(n3)])
        self.index = base_index

    def __repr__(self):
        return f"Tri(nodes={self.nodes}, index={self.index})"


class C_FeFlow:
    def __init__(self):
        self.allEdges: List[C_FeFlowEdg] = []
        self.allEdgesWithoutDuplicates: List[C_FeFlowEdg] = []
        self.allTriangles: List[C_FeFlowTri] = []
        self.allTrianglesWithoutDuplicates: List[C_FeFlowTri] = []
        self.definedEdges: List[C_FeFlowEdg] = []
        self.definedTriangles: List[C_FeFlowTri] = []
        self.undefinedEdges: List[C_FeFlowEdg] = []
        self.undefinedTriangles: List[C_FeFlowTri] = []

    def generateAllTriangles(self, tetrahedronlist):
        self.allTriangles = []
        numberoftetrahedra = len(tetrahedronlist)

        print("Generating triangles from tetrahedra...")

        for t in range(numberoftetrahedra):
            tet = tetrahedronlist[t]

            base_index = t

            # Faces of a tetrahedron (each is a triangle)
            faces = [
                [int(tet[0]), int(tet[1]), int(tet[2])],
                [int(tet[0]), int(tet[1]), int(tet[3])],
                [int(tet[0]), int(tet[2]), int(tet[3])],
                [int(tet[1]), int(tet[2]), int(tet[3])]
            ]

            for face in faces:
                self.allTriangles.append(C_FeFlowTri(face[0], face[1], face[2], base_index))

        print("Total triangles generated:", len(self.allTriangles))

        # Step 1: Sort by node values (n1, n2, n3) and then by original index
        self.allTriangles.sort(key=lambda tri: (tri.nodes[0], tri.nodes[1], tri.nodes[2], tri.index))

        # Step 2: Deduplicate by node tuples, keeping first (lowest index)
        self.allTrianglesWithoutDuplicates.clear()
        last_node_tuple = None
        for tri in self.allTriangles:
            node_tuple = tuple(tri.nodes)
            if node_tuple != last_node_tuple:
                self.allTrianglesWithoutDuplicates.append(tri)
                last_node_tuple = node_tuple

        print("Total triangles after deduplication:", len(self.allTrianglesWithoutDuplicates))

        # Step 3: Sort again by original index to restore general order
        self.allTrianglesWithoutDuplicates.sort(key=lambda tri: tri.index)

        # Step 4: Re-index (assign new index)
        for i, tri in enumerate(self.allTrianglesWithoutDuplicates):
            tri.index = i

        # Step 5: Final sort by nodes for fast lookup
        self.allTrianglesWithoutDuplicates.sort(key=lambda tri: (tri.nodes[0], tri.nodes[1], tri.nodes[2]))

        print("Final triangles ready:", len(self.allTrianglesWithoutDuplicates))



    def generateMarkerTriangles(self, marker_triangles: np.ndarray):
        marker_triangle_indices = []
        triangle_set = {tuple(tri.nodes): tri.index for tri in self.allTrianglesWithoutDuplicates}

        for tri in marker_triangles:
            sorted_tri = tuple(sorted(map(int, tri)))
            if sorted_tri in triangle_set:
                marker_triangle_indices.append(triangle_set[sorted_tri])
            else:
                print(f"Warning: Triangle {sorted_tri} not found in allTrianglesWithoutDuplicates")

        return marker_triangle_indices

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
