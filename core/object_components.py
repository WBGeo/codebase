from __future__ import annotations

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

from core.meshing_components.mesh_format.Exodus.Exo_format import ExosInputs
from core.meshing_components.mesh_format.VTU.VTU_format import VTUInputs
from core.meshing_components.mesh_format.VTM.VTM_format import VTMInputs
from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
from pydantic import BaseModel, field_serializer, field_validator, BeforeValidator, PlainSerializer, PlainValidator

from core.structuralmodeling_components.structural_objects.structural_objects import StructuralFrame


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
    mapping_object: Dict
    surface_points: PandasDataFrame
    orientations: Optional[PandasDataFrame] = None

    def __post_init__(self):
        # reorder surface_points DataFrame by formation column for colormaps
        formation_order = [item for sublist in self.mapping_object.values() for item in
                           (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        formation_cat_type = pd.CategoricalDtype(categories=formation_order, ordered=True)
        self.surface_points['formation'] = self.surface_points['formation'].astype(formation_cat_type)
        self.surface_points = self.surface_points.sort_values(by='formation').reset_index(drop=True)
        self.surface_points['formation'] = self.surface_points['formation'].astype(str)


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

@wbgeo_type(name='Meshing results', color='green', identifier='MeshResults')
@dataclass(config={"arbitrary_types_allowed": True})
class MeshResults:
    elements: Union[np.ndarray, List[meshio.CellBlock]]
    nodes: np.ndarray
    mesh: Optional[pyvista.MultiBlock] = None

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
        Export the mesh input_data to a VTU file.
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
        Export the mesh input_data to an Exodus file.
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
        Export the mesh input_data to a VTM file.
        Args:
            filename (str): The name of the VTM file to export.
        """
        self.mesh.save(filename)
        print(f"VTM file '{filename}' with multiple blocks created successfully!")
