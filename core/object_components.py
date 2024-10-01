from pydantic.dataclasses import dataclass
from typing import Optional
from pydantic_numpy import NpNDArrayFp64, NpNDArrayInt64
import pandas as pd
from typing import TypeVar, Dict, List

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')


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
    """
    name: str
    lith_block: NpNDArrayInt64
    surface_meshes_vertices: List
    surface_meshes_edges: List
    grid: NpNDArrayFp64
    extent: NpNDArrayInt64
    resolution: NpNDArrayInt64
    mapping_object: Dict
