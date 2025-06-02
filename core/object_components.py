import pyvista
import typing
from py_api_wbgeo.nodesapi import wbgeo_type, AnnotatedScriptType
from pydantic.dataclasses import dataclass
from typing import Optional
from pydantic_numpy import NpNDArrayFp64, NpNDArrayInt64
import pandas as pd
from typing import TypeVar, Dict, List
from typing import Dict, List, Any
import pyvista as pv
import meshio
from core.meshing_components.mesh_format.Exodus.Exo_format import ExosInputs
from core.meshing_components.mesh_format.VTK.VTK_format import VTKInputs
from core.meshing_components.mesh_format.VTM.VTM_format import VTMInputs
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
from pydantic import BaseModel, field_serializer, field_validator, BeforeValidator, PlainSerializer, PlainValidator



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
    """
    name: str
    lith_block: NpNDArrayInt64
    surface_meshes_vertices: List[List[NpNDArrayFp64]]
    surface_meshes_edges: List[List[NpNDArrayFp64]]
    grid: NpNDArrayFp64
    extent: NpNDArrayInt64
    resolution: NpNDArrayInt64
    mapping_object: Dict
    scalar_fields: Optional[List[NpNDArrayFp64]] = None



@dataclass(config={"arbitrary_types_allowed": True})
class MeshResults:
    """
    Data class to hold 3D mesh data.

    Attributes:
        elements (NpNDArrayInt64): A 2D array representing the hexahedral/tetrahedral elements of the mesh.
        nodes (NpNDArrayFp64): A 2D array representing the information of nodes.
        mesh (Optional[pyvista.MultiBlock]): A pyvista mesh object.
    """
    elements: NpNDArrayInt64
    nodes: NpNDArrayFp64
    mesh: Optional[pyvista.MultiBlock] = None

    def __post_init__(self):
        # Initialize the VTMInputs
        self.vtm_in = VTMInputs(nodes_array=self.nodes, elements_array=self.elements)

        # Create the VTM mesh
        self.mesh = self.vtm_in.create_mesh()

        # Initialize separate attributes for Nodes and Elements
        self.nodes_obj = Nodes(node_array=self.nodes)
        self.elements_obj = Elements(element_array=self.elements, node_array=self.nodes)



    def export_vtk(self, filename: str):
        """
        Export the mesh data to a VTK file.
        Args:
            filename (str): The name of the VTK file to export.
        """
        vtk_in = VTKInputs(nodes_array=self.nodes, elements_array=self.elements)

        # Create the VTK mesh
        mesh = vtk_in.create_mesh()

        # Write the mesh to a VTK file
        mesh.write(filename, file_format="vtk")
        print(f"VTK file '{filename}' created successfully!")

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

    def export_vtm(self, filename: str):
        """
        Export the mesh data to a VTM file.
        Args:
            filename (str): The name of the VTM file to export.
        """
        self.mesh.save(filename)
        print(f"VTM file '{filename}' with multiple blocks created successfully!")
