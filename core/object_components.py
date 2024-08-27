import numpy as np
import pandas as pd
from typing import Optional, Sequence


class InputData:
    """
    A class to represent the input data for a geological model.

    Attributes:
        name (str): The name of the model.
        extent (np.ndarray): The extent of the model.
        resolution (np.ndarray): The resolution of the model.
        surface_points (pd.DataFrame): DataFrame containing surface points.
        orientations (pd.DataFrame): DataFrame containing orientations.
        mapping_object (dict[str, tuple]): Mapping of structural groups to structural elements.
        faults (Optional[Sequence[bool]]): List of groups that are faults.
        fault_relations (Optional[np.ndarray]): Array of fault relations.
    """

    def __init__(self,
                 name: str,
                 extent: np.ndarray,
                 resolution: np.ndarray,
                 surface_points: pd.DataFrame,
                 orientations: pd.DataFrame,
                 mapping_object: dict[str, tuple],
                 faults: Optional[Sequence[bool]] = None,
                 fault_relations: Optional[np.ndarray] = None):
        """
        Initialize the InputData class with the given parameters.

        Args:
            name (str): The name of the model.
            extent (np.ndarray): The extent of the model.
            resolution (np.ndarray): The resolution of the model.
            surface_points (pd.DataFrame): DataFrame containing surface points.
            orientations (pd.DataFrame): DataFrame containing orientations.
            mapping_object (dict[str, tuple]): Mapping of structural groups to structural elements.
            faults (Optional[Sequence[bool]]): List of groups that are faults.
            fault_relations (Optional[np.ndarray]): Array of fault relations.
        """
        self.name = name
        self.extent = extent
        self.resolution = resolution
        self.surface_points = surface_points
        self.orientations = orientations
        self.mapping_object = mapping_object
        self.faults = faults
        self.fault_relations = fault_relations

        # reorder surface_points DataFrame by formation column for colormaps
        formation_order = [item for sublist in self.mapping_object.values() for item in
                           (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        # Create a categorical type for the formation column
        formation_cat_type = pd.CategoricalDtype(categories=formation_order, ordered=True)
        # Convert the formation column to the categorical type
        self.surface_points['formation'] = self.surface_points['formation'].astype(formation_cat_type)
        # Sort the surface_points DataFrame by the formation column
        self.surface_points = self.surface_points.sort_values(by='formation').reset_index(drop=True)
        # Convert the formation column back to string dtype
        self.surface_points['formation'] = self.surface_points['formation'].astype(str)


class GeomodelResults:
    """
    A class to represent the results of a geological model.

    Attributes:.
        lith_block (np.ndarray): The lithology block of the model.
        surface_meshes_vertices (list): The vertices of the surface meshes of the model.
        surface_meshes_edges (list): The edges of the surface meshes of the model.
        grid (np.ndarray): The grid of the model.
        extent (np.ndarray): The extent of the model.
        resolution (np.ndarray): The resolution of the model.
    """

    def __init__(self,
                 lith_block: np.ndarray,
                 surface_meshes_vertices: list,
                 surface_meshes_edges: list,
                 grid: np.ndarray,
                 extent: np.ndarray,
                 resolution: np.ndarray):
        """
        Initialize the GeomodelResults class with the given parameters.

        Args:
            lith_block (np.ndarray): The lithology block of the model.
            surface_meshes_vertices (list): The vertices of the surface meshes of the model.
            surface_meshes_edges (list): The edges of the surface meshes of the model.
            grid (np.ndarray): The grid of the model.
            extent (list[int]): The extent of the model.
            resolution (list[int]): The resolution of the model.
        """
        self.lith_block = lith_block
        self.surface_meshes_vertices = surface_meshes_vertices
        self.surface_meshes_edges = surface_meshes_edges
        self.grid = grid
        self.extent = extent
        self.resolution = resolution

        # TODO: Format of dc meshes, maybe I actually want to switch to marching cubes for this outside gempy
