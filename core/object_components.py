import collections
import os

import pydantic
import pyvista
import typing
import numpy as np
from pydantic.dataclasses import dataclass
from typing import Optional, Tuple
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
from pydantic import BaseModel, field_serializer, field_validator, BeforeValidator, PlainSerializer, PlainValidator, Field, ConfigDict

from core.structural_modeling_components.structural_objects.structural_objects import StructuralFrame, FaultFrame
from pydantic import BaseModel, field_serializer, field_validator, BeforeValidator, PlainSerializer, \
  PlainValidator, Field, ConfigDict
from pydantic.dataclasses import dataclass
from dataclasses import field

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

@wbgeo_type(name='Input input_data for the rock elements of a structural geological model',
            color='orange',
            identifier='InputData_StructuralElements')
@dataclass(config={"arbitrary_types_allowed": True})
class InputData_StructuralElements:
    """
    A class to represent the input input_data for a geological model.

        Attributes:
            name (str): The name of the model.
            mapping_object (dict): Mapping of structural groups to structural elements.
            surface_points (pd.DataFrame): DataFrame containing surface points.
            orientations (Optional[pd.DataFrame]): DataFrame containing orientations.
    """
    name: str
    mapping_object: Dict[str, Tuple[str, ...]]
    surface_points: PandasDataFrame
    orientations: Optional[PandasDataFrame] = None

    @field_validator('mapping_object', mode='before')
    @classmethod
    def coerce_mapping_values_to_tuples(cls, v):
        if isinstance(v, dict):
            return {k: (val,) if isinstance(val, str) else tuple(val) for k, val in v.items()}
        return v

    def __post_init__(self):
        # reorder surface_points DataFrame by formation column for colormaps
        formation_order = [item for sublist in self.mapping_object.values() for item in
                           (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        formation_cat_type = pd.CategoricalDtype(categories=formation_order, ordered=True)
        self.surface_points['formation'] = self.surface_points['formation'].astype(formation_cat_type)
        self.surface_points = self.surface_points.sort_values(by='formation').reset_index(drop=True)
        self.surface_points['formation'] = self.surface_points['formation'].astype(str)

        # Remove duplicate surface points (same X, Y, Z, formation)
        _before = len(self.surface_points)
        self.surface_points = self.surface_points.drop_duplicates(
            subset=['X', 'Y', 'Z', 'formation']).reset_index(drop=True)
        _removed = _before - len(self.surface_points)
        if _removed > 0:
            print(f"[InputData_StructuralElements '{self.name}'] "
                  f"Removed {_removed} duplicate surface point(s) (identical X, Y, Z, formation).")

        # Remove duplicate orientations (same X, Y, Z, formation)
        if self.orientations is not None and not self.orientations.empty:
            _before = len(self.orientations)
            self.orientations = self.orientations.drop_duplicates(
                subset=['X', 'Y', 'Z', 'formation']).reset_index(drop=True)
            _removed = _before - len(self.orientations)
            if _removed > 0:
                print(f"[InputData_StructuralElements '{self.name}'] "
                      f"Removed {_removed} duplicate orientation(s) (identical X, Y, Z, formation).")


@wbgeo_type(name='Input input_data for the fault elements of a structural geological model',
            color='orange',
            identifier='InputData_FaultElements')
@dataclass(config={"arbitrary_types_allowed": True})
class InputData_FaultElements:
    """
    A class to represent the input input_data for a geological model.

        Attributes:
            name (str): The name of the model.
            fault_surface_points (pd.DataFrame): DataFrame containing surface points.
            fault_orientations (pd.DataFrame): DataFrame containing orientations.
    """
    name: str
    fault_surface_points: PandasDataFrame
    fault_orientations: PandasDataFrame  # Might be optional in future when not only UCK is used here
    fault_names: List[str]  # This allows us to use one input data file

    def __post_init__(self):
        # Remove duplicate fault surface points (same X, Y, Z, formation)
        _before = len(self.fault_surface_points)
        self.fault_surface_points = self.fault_surface_points.drop_duplicates(
            subset=['X', 'Y', 'Z', 'formation']).reset_index(drop=True)
        _removed = _before - len(self.fault_surface_points)
        if _removed > 0:
            print(f"[InputData_FaultElements '{self.name}'] "
                  f"Removed {_removed} duplicate fault surface point(s) (identical X, Y, Z, formation).")

        # Remove duplicate fault orientations (same X, Y, Z, formation)
        _before = len(self.fault_orientations)
        self.fault_orientations = self.fault_orientations.drop_duplicates(
            subset=['X', 'Y', 'Z', 'formation']).reset_index(drop=True)
        _removed = _before - len(self.fault_orientations)
        if _removed > 0:
            print(f"[InputData_FaultElements '{self.name}'] "
                  f"Removed {_removed} duplicate fault orientation(s) (identical X, Y, Z, formation).")


@wbgeo_type(name='Result of a structural geological model', color='blue', identifier='StructuralModelResults')
@dataclass(config={"arbitrary_types_allowed": True})
class StructuralModelResults:
    """
    A class to represent the results of a geological model.

        Attributes:.
            structural_frame (StructuralFrame): The structural frame of the model.
    """
    # TODO: ALEX: This is the simplest version I could think of - does this work for you
    structural_frame: StructuralFrame  # this is a deepcopy of the structural frame object


@wbgeo_type(name='Result of a structural fault model', color='blue', identifier='FaultModelResults')
@dataclass(config={"arbitrary_types_allowed": True})
class FaultModelResults:
    """
    A class to represent the results of a fault model.

        Attributes:.
            fault_frame (FaultFrame): The fault frame of the model.
    """
    # TODO: ALEX: This is the simplest version I could think of - does this work for you
    fault_frame: FaultFrame  # this is a deepcopy of the structural frame object

# todo: Move into common class?
def cellblock_encoder(obj: meshio.CellBlock):
  import pickle
  import codecs
  return codecs.encode(pickle.dumps(obj), "base64").decode()


@wbgeo_type(name='Meshing results', color='green', identifier='MeshResults')
@dataclass(config={"arbitrary_types_allowed": True, "json_encoders" : {meshio.CellBlock: cellblock_encoder}})
class MeshResults:
    """
    Container class holding unstructured mesh results.

    Attributes
    ----------
    nodes : np.ndarray
        Array of node coordinates with shape (N, 3)

    elements : list[meshio.CellBlock]
        Mesh elements stored as MeshIO CellBlocks.

    cell_data : dict[str, list[np.ndarray]], optional
        Per-cell data arrays associated with the mesh.
    """

    nodes: NpNDArrayFp64
    elements: List[meshio.CellBlock]
    cell_data: Optional[Dict[str, List[np.ndarray]]] = None

    # transient / derived
    mesh: Optional[pyvista.MultiBlock] = Field(default=None, exclude=True)

    # -----------------------------
    # Decode serialized CellBlocks
    # -----------------------------
    @pydantic.field_validator("elements", mode="before")
    @classmethod
    def decode_cellblock(cls, v):
        if v is None:
            return None

        import pickle
        import codecs

        return [
            e if isinstance(e, meshio.CellBlock) or e is None
            else pickle.loads(codecs.decode(e.encode(), "base64"))
            for e in v
        ]

    # -----------------------------
    # Post init
    # -----------------------------
    def __post_init__(self):
        self.vtm_in = VTMInputs(
            self.nodes,
            self.elements
        )

        self.mesh = self.vtm_in.create_mesh()

@wbgeo_type(name='Exporters', color='grey', identifier='Exporters')
@dataclass(config={"arbitrary_types_allowed": True})
class Exporters(MeshResults):
    """
    Export utility class for MeshResults.

    This class extends `MeshResults` and provides methods to export
    the mesh into various standard geoscientific and engineering formats.

    Supported mesh types depend on the export format:
    - Structured meshes: VTU, VTK, VTM, Exodus, Ansys
    - Unstructured meshes: STL, Abaqus, Gmsh, FeFlow, VTU, VTK, VTM, Exodus, Ansys

    """

    def export_vtu(self, filename: str):
        """
        Export the mesh to a VTU (VTK Unstructured Grid) file.

        Args:
        filename: Output filename ending with `.vtu`.

        Supported Mesh Types:
        - Structured meshes
        - Unstructured meshes
        """
        vtu_in = VTUInputs(self.nodes, self.elements )
        # Create the VTU mesh
        mesh = vtu_in.create_mesh()
        # Write the mesh to a VTU file
        mesh.write(filename, file_format="vtu")
        print(f"VTU file '{filename}' created successfully!")


    def export_stl(self, filename: str):
        """
        Export the mesh surface to STL format.

        Args:
        filename: Output STL filename.

        Supported Mesh Types:
        - Unstructured meshes ONLY
        """
        stl_in = STLInputs(self.nodes, self.elements)
        stl_in.output_filename = filename
        stl_in.create_mesh()
        print(f"Stl files '{filename}' created successfully!")


    def export_exodus(self, filename: str):
        """
        Export the mesh to an Exodus (.exo) file.

        Args:
        filename: Output filename ending with `.exo`.

        Supported Mesh Types:
        - Structured meshes
        - Unstructured meshes
        """
        exo_in = ExosInputs(self.nodes, self.elements)
        # Create mesh
        mesh = exo_in.create_mesh()
        # Write the mesh to an Exodus file
        mesh.write(filename, file_format="exodus")
        print(f"Exodus file '{filename}' created successfully!")


    def export_abaqus(self, filename: str):
        """
        Export the mesh to an Abaqus input (.inp) file.

        Args:
        filename:Output Abaqus input filename.

        Supported Mesh Types:
        --------------------
        - Unstructured meshes ONLY
        """
        abq = AbaqusInputs(self.nodes,self.elements)
        abq.write(filename)
        print(f"Abaqus file '{filename}' created successfully!")



    def export_ansys(self, filename: str):
        """
        Export the mesh to an Ansys-compatible format.

        Args:
        filename:Output filename.

        Supported Mesh Types:
        - Structured meshes
        - Unstructured meshes
        """
        Ansys_in = AnsysInputs(self.nodes, self.elements)
        # Create mesh
        mesh = Ansys_in.create_mesh()
        # Write the mesh to an Exodus file
        mesh.write(filename, file_format="ansys")
        print(f"Ansys file '{filename}' created successfully!")



    def export_gmsh(self, filename: str):
        """
        Export the mesh to Gmsh (.msh) format.

        Args:
        filename: Output filename ending with `.msh`.

        Supported Mesh Types:
        --------------------
        - Unstructured meshes ONLY
        """
        elements =self.elements
        if not isinstance(elements, list):
            raise TypeError("Gmsh export requires unstructured CellBlocks. You can try " \
            "exporting to structured formats like VTU, Exodus, VTM, VTK, ...")
        gmsh_in = GMSHInputs(self.nodes, elements,)
        mesh = gmsh_in.create_mesh()
        meshio.write(filename, mesh, file_format="gmsh")
        print(f"GMSH file '{filename}' created successfully!")



    def export_vtk(self, filename: str):
        """
        Export the mesh to a legacy VTK file.

        Args:
        ----------
        filename: Output filename ending with `.vtk`.

        Supported Mesh Types:
        --------------------
        - Structured meshes
        - Unstructured meshes
        """
        self.mesh.save(filename)
        print(f"VTK file '{filename}' with created successfully!")



    def export_vtm(self, filename: str):
        """
        Export the mesh to a VTM (VTK MultiBlock) file.

        Args:
        filename: Output filename ending with `.vtm`.

        Supported Mesh Types:
        - Structured meshes
        - Unstructured meshes
        """
        self.mesh.save(filename)
        print(f"VTM files '{filename}' created successfully!")



    def export_feflow(self, filename: str):
        """
        Export the mesh to a FeFlow (.fem) file.

        Args:
        filename: Output filename ending with `.fem`.

        Supported Mesh Types:
        - Unstructured meshes ONLY
        """
        elements = self.elements
        # Reject structured meshes
        if not isinstance(elements, list):
            raise TypeError("FeFlow export supports ONLY unstructured meshes.\n"
            "You can export a structured mesh using other formats (e.g., VTU, VTK, VTM, Exodus,..).")

        feflow = FeflowInputs(self.nodes, elements)
        feflow.write(filename)
        print(f"Feflow file '{filename}' created successfully!")


@wbgeo_type(name='SimulationResults', color='pink', identifier='SimulationResults')
@dataclass(config={"arbitrary_types_allowed": True})

class SimulationResults:
    """
    Container class for all simulation results timesteps.

    Attributes
    ----------
    nodes_by_time : Dict[float, np.ndarray]
        Node coordinates for each timestep.

    cells_by_time : Dict[float, np.ndarray]
        Cell connectivity for each timestep.

    celltypes_by_time : Dict[float, np.ndarray]
        Cell types for each timestep.

    node_data_by_time : Dict[float, Dict[str, np.ndarray]]
        Node-based data arrays for each timestep.

    cell_data_by_time : Dict[float, Dict[str, np.ndarray]]
        Cell-based data arrays for each timestep.
    """
    nodes_by_time: Dict[float, np.ndarray] = field(default_factory=dict)
    cells_by_time: Dict[float, np.ndarray] = field(default_factory=dict)
    celltypes_by_time: Dict[float, np.ndarray] = field(default_factory=dict)
    node_data_by_time: Dict[float, Dict[str, np.ndarray]] = field(default_factory=dict)
    cell_data_by_time: Dict[float, Dict[str, np.ndarray]] = field(default_factory=dict)
