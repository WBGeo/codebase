import meshio
from typing import Union, List
from numpy.typing import NDArray
import numpy as np
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
import importlib
import tempfile
import io
import os
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile

class AbaqusInputs:
    """
    Class to construct and export Abaqus-compatible meshes.

    It converts node and element data into a meshio.Mesh object and
    provides functionality to write the mesh to an Abaqus `.inp` file.
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


    def create_mesh(self) -> meshio.Mesh:
        """
        Create a meshio Mesh object suitable for Abaqus export.

        Returns:
        meshio.Mesh: Mesh object containing points and cell connectivity.
        """

        points: NDArray[np.float64] = self.nodes


        if points.shape[1] < 3:
            raise ValueError("Node coordinates must be 3D for Abaqus")

        # Filter supported Abaqus types
        cells: List[tuple[str, NDArray[np.int64]]] = []
        for block in self.elements_block:
            if block.type in {"line", "triangle", "quad", "tetra", "hexahedron"}:
                cells.append((block.type, block.data))
            else:
                print(f"Skipping unsupported Abaqus cell type: {block.type}")

            # Create mesh and write
        mesh: meshio.Mesh = meshio.Mesh(points=points, cells=cells)

        return mesh

    def write(self, filename: str):
        """
        Write the mesh to an Abaqus `.inp` file.

        This method:
        - writes nodes and elements in Abaqus format,
        - groups elements into ELSETs by cell block,
        - assigns default materials and section definitions.

        Args:
        filename: Path to the output Abaqus `.inp` file.
        """
        mesh = self.create_mesh()
        node_array = mesh.points
        elements = mesh.cells

        element_type_map = {
            "line": "T3D2",
            "tetra": "C3D4",
            "triangle": "S3R",
            "hexahedron": "C3D8",
            "quad": "S4R",
        }

        with open(filename, "w") as f:
            f.write("*" * 37 + "\n")
            f.write("*HEADING\n")
            f.write("WBGeo Abaqus Export\n")
            f.write("*" * 37 + "\n")

            # Nodes
            f.write("*NODE, NSET=All\n")
            for i, coord in enumerate(node_array, start=1):
                x, y, z = coord
                f.write(f"{i}, {x:.8E}, {y:.8E}, {z:.8E}\n")

            # Track ELSETs
            solid_elsets = []
            tus_elsets = []
            shel_elsets = []

            # Write elements
            element_id = 1
            for i, block in enumerate(elements):
                abaqus_type = element_type_map.get(block.type)
                if abaqus_type is None:
                    print(f"⚠️ Skipping unsupported element type: {block.type}")
                    continue

                elset_name = f"ELSET{i+1}"
                f.write(f"*ELEMENT,TYPE={abaqus_type},ELSET={elset_name}\n")
                for conn in block.data:
                    conn_str = ", ".join(str(int(n) + 1) for n in conn)
                    f.write(f"{element_id}, {conn_str}\n")
                    element_id += 1

                if abaqus_type == "C3D4" or abaqus_type == "C3D8":
                    solid_elsets.append(elset_name)
                elif abaqus_type == "T3D2":
                    tus_elsets.append(elset_name)
                elif abaqus_type == "S3":
                    shel_elsets.append(elset_name)

            # Hardcoded material block (no input)
            # For tetras or hexas
            f.write("*MATERIAL, NAME=DefaultMaterial\n")
            f.write("*ELASTIC\n")
            f.write("2.100000E+05, 0.300000\n")  # Young's modulus, Poisson's ratio

            for elset in solid_elsets:
                f.write(f"*SOLID SECTION, ELSET={elset}, MATERIAL=DefaultMaterial\n")

            # For triangles
            f.write("*MATERIAL, NAME=myrock\n")
            f.write("*ELASTIC\n")
            f.write("2.100000E+05, 0.300000\n")
            for elset in shel_elsets:
                f.write(f"*SHELL SECTION, ELSET={elset}, MATERIAL=myrock\n")
                f.write("0.01\n")

            # For lines
            f.write("*MATERIAL, NAME=STEEL\n")
            f.write("*ELASTIC\n")
            f.write("2.100000E+05, 0.300000\n")
            for elset in tus_elsets:
                f.write(f"*SOLID SECTION, ELSET={elset}, MATERIAL=STEEL\n")
                f.write("0.01\n")

        print(f"[INFO] Abaqus file written: {filename}")



@wbgeo_component(
    title="Download Mesh as Abaqus",
    description="Export Mesh to Abaqus",
    group="Export",
    identifier="wbgeo::expert_mesh_results_abaqus",
)
def export_mesh_results_to_abaqus(mesh) -> BasicallyABufferedFile:
    """
    Export a WBGeo MeshResults object to an Abaqus `.inp` file.
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
    with tempfile.NamedTemporaryFile(suffix=".inp", delete=False) as tmp:
        tmp_path = tmp.name
        exporters.export_abaqus(tmp_path)

    # Read into memory
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    buf.filename = "mesh_export_abaqus.inp"

    # Cleanup
    os.remove(tmp_path)

    return buf
