import meshio
from typing import Union, List
from numpy.typing import NDArray

import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import pyvista as pv
from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass
from typing import List
import importlib
import tempfile
import io
import os
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
@dataclass
class C_FeFlowTri:
    """
    Representation of a triangular face used in FEFLOW export.

    Each triangle is defined by three node indices. The node indices
    are stored in sorted order to allow reliable comparison and
    deduplication. An internal index is used to track the triangle's
    position in generated lists.
    """
    nodes: List[int]
    index: int = 0

    def __init__(self, n1=0, n2=0, n3=0, index=0):
        """
        Create a triangle from three node indices.

        Args:
        n1, n2, n3: Node indices defining the triangle.
        index: Internal triangle index used for ordering and lookup.
        """
        self.nodes = sorted([int(n1), int(n2), int(n3)])
        self.index = index

    def __repr__(self):
        return f"Tri(nodes={self.nodes}, index={self.index})"


@dataclass
class C_FeFlowEdg:
    """
    Representation of an edge used in FEFLOW export.

    Each edge is defined by two node indices, stored in sorted order.
    An internal index is used to track the edge's position after
    deduplication.
    """
    nodes: List[int]
    index: int = 0

    def __init__(self, n1=0, n2=0, index=0):
        """
        Create an edge from two node indices.

        Args:
        n1, n2: Node indices defining the edge.
        index: Internal edge index.
        """
        self.nodes = sorted([int(n1), int(n2)])
        self.index = index

    def __repr__(self):
        return f"Edge(nodes={self.nodes}, index={self.index})"


class C_FeFlow:
    """
    Class for constructing FEFLOW facesets and edgesets.

    This class reproduces the logic of the original C++ FEFLOW
    preprocessing routines. It generates:
    - all unique triangle faces from tetrahedra,
    - all unique edges from tetrahedra,
    - mappings between user-defined surface/edge markers and
      internal triangle/edge indices.
    """
    def __init__(self):

        self.allTriangles: List[C_FeFlowTri] = []
        self.allTrianglesWithoutDuplicates: List[C_FeFlowTri] = []
        self.undefinedTriangles: List[C_FeFlowTri] = []
        self.definedTriangles: List[C_FeFlowTri] = []

    def generateAllTriangles(self, tetrahedronlist: np.ndarray):
        """
        Generate all unique triangular faces from a list of tetrahedra.

        For each tetrahedron, the four faces are generated, sorted,
        deduplicated, and assigned consistent indices. The resulting
        triangle list is optimized for fast lookup.

        Args:
        tetrahedronlist: Array of tetrahedral elements with shape (N, 4).
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
        Collect triangles associated with a specific surface marker.

        Args:
        marker: Marker value identifying the surface.
        triangle_markers: Array of markers corresponding to each triangle.
        triangle_list: Array of triangle connectivities.
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
        Match marked triangles to the internally generated unique triangles.
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
        """
        Generate triangle indices for a given set of marked surface triangles.

        Args:
        marker_triangles: Array of triangle connectivities belonging to a surface.

        Returns:
        list(int): Indices of matching triangles in the global triangle list.
        """
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
        """
        Generate all unique edges from a list of tetrahedra.

        Args:
        tetrahedronlist: Array of tetrahedral elements with shape (N, 4).
        """
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
        """
        Map marked edges to internal edge indices.

        Args:
        marker_edges: Array of edge connectivities with shape (N, 2).

        Returns:
        list(int): Indices of matching edges in the global edge list.
        """
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
    """
    Build and export meshes in FEFLOW (.fem) format.

    This class supports only unstructured meshes
    """
    def __init__(
        self,
        nodes,
        elements: List[meshio.CellBlock],
    ) -> None:

        # Normalize nodes to NumPy
        self.nodes = np.asarray(nodes, dtype=float)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be Nx3 coordinates.")

        if not isinstance(elements, list):
            raise TypeError("elements_array must be List[meshio.CellBlock]")
        self.elements_block: List[meshio.CellBlock] = elements

        self.elements = elements

    def create_mesh(self):
        """
        Creates a mesh using the meshio library and saves it to the specified output file.

        Returns:
            meshio.Mesh: The mesh object.
        """


        # Filter supported feflow types
        cells = []
        for block in self.elements_block:
            if block.type == "hexahedron":
                raise ValueError("Only unstructured meshes with tetrahedral elements are supported for FeFlow export.")
            elif block.type in {"line", "triangle", "quad", "tetra"}:
                cells.append((block.type, block.data))
            else:
                print(f"Skipping unsupported FEFLOW cell type: '{block.type}'")


        mesh = meshio.Mesh(points=self.nodes, cells=cells)

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

    def write(self, filename: str):
        """
        Write the mesh to a FEFLOW `.fem` file.

        The method generates node coordinates,tetrahedral elements, element sets, face sets (surfaces),
         and edge sets (polylines),

        Args:
        filename: Output filename for the `.fem` file.
        """

        mesh = self.create_mesh()
        node_array = mesh.points
        elements = mesh.cells

        # Gather all tetra, triangle, and line elements, and assign group-based markers
        tetra_all = []
        tetra_markers = []
        triangle_all = []
        triangle_markers = []
        edge_all = []
        edge_markers = []

        for idx, block in enumerate(elements):
            if block.type == "tetra":
                tetra_all.append(block.data)
                tetra_markers.append(np.full(len(block.data), idx + 1))  # use idx+1 as region ID
            elif block.type == "triangle":
                triangle_all.append(block.data)
                triangle_markers.append(np.full(len(block.data), idx + 1))  # same logic
            elif block.type == "line":
                edge_all.append(block.data)
                edge_markers.append(np.full(len(block.data), idx + 1))

        # Concatenate all
        tetra = np.vstack(tetra_all) if tetra_all else np.empty((0, 4), dtype=int)
        tetra_markers = np.concatenate(tetra_markers) if tetra_markers else None

        triangles = np.vstack(triangle_all) if triangle_all else np.empty((0, 3), dtype=int)
        triangle_markers = np.concatenate(triangle_markers) if triangle_markers else None

        edges = np.vstack(edge_all) if edge_all else np.empty((0, 2), dtype=int)
        edge_markers = np.concatenate(edge_markers) if edge_markers else None

        # Start writing FeFlow file
        with open(filename, 'w') as f:
            num_points = len(node_array)
            num_tetra = len(tetra)
            f.write("PROBLEM:\n")
            f.write("CLASS (v.7)\n")
            f.write("   2    1    0    3    0    0    8    8    0    0\n")
            f.write("DIMENS\n")
            f.write(f"   {num_points}     {num_tetra}     0      1      0      0      0      0      0      2     0      0      1      0      0      0      0\n")

            f.write("SCALE\n")
            f.write("   1.0, 1.0, 1.0, 1.0, 0.0, 0.0\n")
            f.write("VARNODE\n")
            f.write(f"   {num_tetra}     4     4\n")

            # Write tetrahedra connectivity (1-based)
            for t in range(num_tetra):
                tet_nodes = tetra[t] + 1
                f.write(f"   6     {tet_nodes[0]}     {tet_nodes[1]}     {tet_nodes[2]}     {tet_nodes[3]}\n")

            f.write("XYZCOOR\n")
            for x, y, z in node_array:
                f.write(f"     {x}, {y}, {z}\n")

            # ELEMENTALSETS
            if tetra_markers is not None:
                f.write("ELEMENTALSETS\n")
                for m in sorted(set(tetra_markers)):
                    f.write(f"     \"Region: Name: R{m}\"")
                    havewritten = 0
                    for t, mark in enumerate(tetra_markers):
                        if mark == m:
                            if (havewritten % 10) == 0:
                                f.write("\n\t\t")
                            f.write(f"{t + 1} ")
                            havewritten += 1
                    f.write("\n")

            # Create all unique triangles from tets
            FeFlowObj =C_FeFlow()
            FeFlowObj.generateAllTriangles(tetra)

            if triangle_markers is not None and len(triangle_markers) > 0:
                f.write("FACESETS\n")
                minMat = int(np.min(triangle_markers))
                maxMat = int(np.max(triangle_markers))

                for m in range(minMat, maxMat + 1):
                    mask = triangle_markers == m
                    triangles_with_marker = triangles[mask]

                    # Reuse FeFlowObj (already has all unique triangles from tets)
                    FeFlowObj.generateUndefinedTriangles(
                        marker=m,
                        triangle_markers=triangle_markers,
                        triangle_list=triangles
                    )
                    FeFlowObj.generateDefinedTriangles()

                    marker_triangle_indices = [tri.index for tri in FeFlowObj.definedTriangles]

                    f.write(f'     "Surface: Name: S{m}"')
                    havewritten = 0
                    for tri in marker_triangle_indices:
                        if (havewritten % 10) == 0:
                            f.write("\n\t\t")
                        f.write(f"{tri + 1} ")
                        havewritten += 1
                    f.write("\n")


            # Create all unique edges from lines
            FeFlowObj.generateAllEdges(tetra)
            # DEBUG: check edges and markers
            print("Edges array shape:", edges.shape)
            if edge_markers is not None:
                print("Edge markers unique:", np.unique(edge_markers))
            else:
                print("Edge markers is None")

            if edge_markers is not None and len(edge_markers) > 0:
                f.write("EDGESETS\n")
                minEdgeMat = int(np.min(edge_markers))
                maxEdgeMat = int(np.max(edge_markers))

                for m in range(minEdgeMat, maxEdgeMat + 1):
                    marker_to_extract = m

                    # Boolean mask for edges with the desired marker
                    mask = edge_markers == marker_to_extract

                    # Apply mask to get edges block
                    edges_with_marker = edges[mask]
                    marker_edge_indices = FeFlowObj.generateMarkerEdges(edges_with_marker)

                    f.write(f'     "Polyline: Name: P{m}"')
                    havewritten = 0
                    for edge in marker_edge_indices:
                        if (havewritten % 10) == 0:
                            f.write("\n\t\t")
                        f.write(f"{edge + 1} ")
                        havewritten += 1
                    f.write("\n")


            f.write("END\n")

        print(f" Feflow file '{filename}' written successfully.")




@wbgeo_component(
    title="Download Mesh as Feflow",
    description="Export Mesh to Feflow",
    group="Export",
    identifier="wbgeo::expert_mesh_results_feflow",
)
def export_mesh_results_to_feflow(mesh) -> BasicallyABufferedFile:
    """
    Export a WBGeo MeshResults object to a FEFLOW (.fem) file.
    """
    import importlib
    import tempfile
    import io
    import os

    # Lazy import to avoid circular dependency
    obj_module = importlib.import_module("core.object_components")
    MeshResults = getattr(obj_module, "MeshResults")
    Exporters = getattr(obj_module, "Exporters")

    if not isinstance(mesh, MeshResults):
        raise TypeError(f"Expected a MeshResults instance, got {type(mesh)}")

    exporters = Exporters(**mesh.__dict__)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".fem", delete=False) as tmp:
        tmp_path = tmp.name
        exporters.export_feflow(tmp_path)

    # Read back into memory
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    buf.filename = "mesh_export_feflow.fem"

    # Cleanup
    os.remove(tmp_path)

    return buf
