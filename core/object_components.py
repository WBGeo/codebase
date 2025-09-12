import pyvista
import typing
import numpy as np
#from py_api_wbgeo.nodesapi import wbgeo_type, AnnotatedScriptType
from pydantic.dataclasses import dataclass
from typing import Optional
from pydantic_numpy import NpNDArrayFp64, NpNDArrayInt64
import pandas as pd
from typing import TypeVar, Dict, List
from typing import Dict, List, Any
import pyvista as pv
import meshio
from typing import Optional, Union, List
from pydantic_numpy.typing import NpNDArrayInt64, NpNDArrayFp64
from py_api_wbgeo.nodesapi import wbgeo_type


from core.meshing_components.mesh_format.EXUDOS.Exo_format import ExosInputs
from core.meshing_components.mesh_format.VTU.VTU_format import VTUInputs
from core.meshing_components.mesh_format.VTM.VTM_format import VTMInputs
from core.meshing_components.mesh_format.STL.STL_format import STLInputs
from core.meshing_components.mesh_format.GMSH.GMSH_format import GMSHInputs
from core.meshing_components.mesh_format.ABAQUS.Abaqus_format import AbaqusInputs
from core.meshing_components.mesh_format.FEFLOW.Feflow_format import FeflowInputs, C_FeFlow
from core.meshing_components.mesh_format.ANSYS.Ansys_format import AnsysInputs

from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
from pydantic import BaseModel, field_serializer, field_validator, BeforeValidator, PlainSerializer, PlainValidator, Field



# Pydantic adapter for panda DataFrame
def df_serializer(df: pd.DataFrame) -> list[dict]:
  return df.to_dict(orient="records")


def df_validator(value) -> pd.DataFrame:
  if isinstance(value, pd.DataFrame):
    return value
  elif isinstance(value, list):
    return pd.DataFrame(value)
  raise TypeError("Expected a pandas DataFrame or a list of dictionaries.")


PandasDataFrame = typing.Annotated[
    pd.DataFrame, PlainSerializer(df_serializer), BeforeValidator(df_validator)]


@wbgeo_type(name='Input data for a geological model', color='orange', identifier='InputData')
@dataclass(config={"arbitrary_types_allowed": True})
class InputData:
    """
    A class to represent the input data for a geological model.

        Attributes:
            name (str): The name of the model.
            extent (np.ndarray): The extent of the model.
            resolution (np.ndarray): The resolution of the model.
            mapping_object (dict): Mapping of structural groups to structural elements.
            surface_points (pd.DataFrame): DataFrame containing surface points.
            orientations (Optional[pd.DataFrame]): DataFrame containing orientations.
            mapping_object (dict): Mapping of structural groups to structural elements.
            faults (Optional[List[bool]]): List of groups that are faults.
            fault_relations (Optional[np.ndarray]): Array of fault relations.
    """
    name: str
    extent: NpNDArrayInt64
    resolution: NpNDArrayInt64
    mapping_object: Dict
    surface_points: PandasDataFrame
    orientations: Optional[PandasDataFrame] = None
    faults: Optional[List[bool]] = None
    fault_relations: Optional[NpNDArrayInt64] = None

    def __post_init__(self):
        # reorder surface_points DataFrame by formation column for colormaps
        formation_order = [item for sublist in self.mapping_object.values() for item in
                           (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        formation_cat_type = pd.CategoricalDtype(categories=formation_order, ordered=True)
        self.surface_points['formation'] = self.surface_points['formation'].astype(formation_cat_type)
        self.surface_points = self.surface_points.sort_values(by='formation').reset_index(drop=True)
        self.surface_points['formation'] = self.surface_points['formation'].astype(str)


@wbgeo_type(name='Result os structural geological model', color='blue', identifier='GeomodelResults')
@dataclass(config={"arbitrary_types_allowed": True})
class GeomodelResults:
    """
    A class to represent the results of a geological model.

        Attributes:.
            name (str): The name of the model.
            lith_block (np.ndarray): The lithology block of the model.
            surface_meshes_vertices (list): The vertices of the surface meshes of the model.
            surface_meshes_edges (list): The edges of the surface meshes of the model.
            grid (np.ndarray): The grid of the model.
            extent (np.ndarray): The extent of the model.
            resolution (np.ndarray): The resolution of the model.
            mapping_object (dict): Mapping of structural groups to structural elements.
            scalar_fields (Optional[List[np.ndarray]]): List of scalar fields.
            faults (Optional[List[bool]]): List of groups that are faults.
    """
    name: str
    lith_block: NpNDArrayInt64
    surface_meshes_vertices: List[List[NpNDArrayFp64]]
    surface_meshes_edges: List[List[NpNDArrayInt64]]
    grid: NpNDArrayFp64
    extent: NpNDArrayInt64
    resolution: NpNDArrayInt64
    mapping_object: Dict
    scalar_fields: Optional[List[NpNDArrayFp64]] = None
    faults: Optional[List[bool]] = None


@wbgeo_type(name='Meshing results', color='green', identifier='MeshResults')
@dataclass(config={"arbitrary_types_allowed": True})
class MeshResults:
    # elements: Union[NpNDArrayFp64, List[meshio.CellBlock]]
    elements: List[meshio.CellBlock] # todo: Why union?
    nodes: NpNDArrayFp64 # TODO: int or FP?
    # mesh is a transient/derived field
    mesh : Optional[pyvista.MultiBlock]  = Field(default=None, exclude = True) #  exclude this field from serialization



    def __post_init__(self):

        self.vtm_in = VTMInputs(nodes_array=self.nodes, elements_array=self.elements)
        print('[INFO] VTMInputs initialized successfully.')

        self.mesh = self.vtm_in.create_mesh()

        # initialize node/element objects only for ndarray elements
        if isinstance(self.elements, np.ndarray):
            self.nodes_obj = Nodes(node_array=self.nodes)
            self.elements_obj = Elements(element_array=self.elements, node_array=self.nodes)


    def export_vtu(self, filename: str):
        """
        Export the mesh data to a VTU file.
        Args:
            filename (str): The name of the VTU file to export.
        """
        vtu_in = VTUInputs(nodes_array=self.nodes, elements_array=self.elements)

        # Create the VTU mesh
        mesh = vtu_in.create_mesh()

        # Write the mesh to a VTU file
        mesh.write(filename, file_format="vtu")
        print(f"VTU file '{filename}' created successfully!")

    def export_exodus(self, filename: str):
        """
        Export the mesh data to an Exodus file.
        Args:
            filename (str): The name of the Exodus file to export.
        """
        exo_in = ExosInputs(nodes_array=self.nodes, elements_array=self.elements)
        # Create mesh
        mesh = exo_in.create_mesh()

        # Write the mesh to an Exodus file
        mesh.write(filename, file_format="exodus")
        print(f"Exodus file '{filename}' created successfully!")



    def export_abaqus(self, filename: str):
        """
        Export mesh data to an Abaqus .inp file with nodes, tetrahedral (C3D4),
        and triangular (CP3S) elements, including a valid material and section definition.
        """
        abq_in = AbaqusInputs(nodes_array=self.nodes, elements_array=self.elements)
        mesh = abq_in.create_mesh()
        node_array = mesh.points
        elements = mesh.cells

        element_type_map = {
            "line": "T3D2",
            "tetra": "C3D4",
            "triangle": "S3R",
            "hexahedron": "C3D8"
        }

        with open(filename, 'w') as f:
            # Write header
            f.write("*" * 37 + "\n")
            f.write("*HEADING\n")
            f.write("ICEM - ABAQUS INTERFACES VERSION 4.3.1\n")
            f.write("*" * 37 + "\n")

            # Write nodes
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

        print(f"✅ Abaqus .inp file '{filename}' written successfully.")



    def export_ansys(self, filename: str):
        """
        Export the mesh data to an Ansys file.
        Args:
            filename (str): The name of the Ansys file to export.
        """
        Ansys_in = AnsysInputs(nodes_array=self.nodes, elements_array=self.elements)
        # Create mesh
        mesh = Ansys_in.create_mesh()

        # Write the mesh to an Exodus file
        mesh.write(filename, file_format="ansys")
        print(f"Ansys file '{filename}' created successfully!")

    def export_gmsh(self, filename: str):
        """
        Export the mesh data to an GMSH file.
        Args:
            filename (str): The name of the GMSH file to export.
        """
        gmsh_in = GMSHInputs(nodes_array=self.nodes, elements_array=self.elements)
        # Create mesh
        mesh = gmsh_in.create_mesh()
        if mesh is None:
           print('GMSH cannot be created for structure mesh')
        else:
            # Write the mesh to an Exodus file
            mesh.write(filename, file_format="gmsh22")
            print(f"GMSH file '{filename}' created successfully!")

    def export_stl(self, filename: str):
        """
        Export the mesh data to STL files.
        Args:
            filename (str): The name of the STL files to export.
        """
        stl_in = STLInputs(nodes_array=self.nodes, elements_array=self.elements)
        stl_in.output_filename = filename  # <--- REQUIRED!
        stl_in.create_mesh()


    def export_vtm(self, filename: str):
        """
        Export the mesh data to a VTM file.
        Args:
            filename (str): The name of the VTM file to export.
        """
        self.mesh.save(filename)
        print(f"VTM file '{filename}' with multiple blocks created successfully!")

    def export_feflow(self, filename: str):
        """
        Export mesh data to an Feflow.fem file with nodes, tetrahedral,
        and triangular and line elements.
        """
        feflow_in = FeflowInputs(nodes_array=self.nodes, elements_array=self.elements)
        mesh = feflow_in.create_mesh()
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
            print(tetra)
            FeFlowObj.generateAllTriangles(tetra)
            print('done')

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



            # EDGESETS (Lines)
            # Create all unique edges from lines
            FeFlowObj.generateAllEdges(tetra)

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

        print(f"✅ Feflow file '{filename}' written successfully.")
