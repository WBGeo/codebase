import numpy as np
import pandas as pd

from typing import Dict, Tuple, Optional, List, Union
from pydantic import BaseModel, Field, PrivateAttr
from enum import Enum



#%%

class InterpolationMethod(str, Enum):
    ORDINARY_KRIGING = "Ordinary Kriging"
    RADIAL_BASIS_FUNCTION = "Radial Basis Function"
    UNIVERSAL_COKRIGING = "Universal Co-Kriging"
    GEOINR = "GeoINR"
    LOOP_STRUCTURAL = "Loop Structural"


class StructuralElement(BaseModel):
    """
    A structural element within a structural group.

    Attributes:
        name: Name of the element.
        scalar_value: Scalar field value (float), set during computation.
        id: Unique identifier for the element (int), set later.
        color: Display color (hex string), set during frame generation.
        vertices: Dictionary of surface mesh vertices arrays keyed by mesh type ('masked', 'unmasked', 'combined').
        edges: Dictionary of surface mesh edges arrays keyed by mesh type ('masked', 'unmasked', 'combined').
    """
    name: str
    _scalar_value: Optional[float] = PrivateAttr(default=None)
    _id: Optional[int] = PrivateAttr(default=None)
    _color: Optional[str] = PrivateAttr(default=None)
    _vertices: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)
    _edges: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    # Read-only properties
    @property
    def scalar_value(self) -> Optional[float]:
        return self._scalar_value

    @property
    def id(self) -> Optional[int]:
        return self._id

    @property
    def color(self) -> Optional[str]:
        return self._color

    @property
    def vertices(self) -> Optional[np.ndarray]:
        return self._vertices

    @property
    def edges(self) -> Optional[np.ndarray]:
        return self._edges

    # Controlled setters
    def set_scalar_value(self, value: float):
        self._scalar_value = value

    def set_id(self, element_id: int):
        self._id = element_id

    def set_color(self, hex_color: str):
        self._color = hex_color

    def set_mesh(self, mesh_type: str, vertices: np.ndarray, edges: np.ndarray):
        """
        Set the vertices and edges for a specific mesh type (e.g., 'masked', 'unmasked', 'combined').

        Raises:
            ValueError if mesh_type is not one of the allowed types or already exists.
        """
        if mesh_type not in {"masked", "unmasked", "combined"}:
            raise ValueError(f"Invalid mesh type '{mesh_type}'. Allowed types are: masked, unmasked, combined.")
        if mesh_type in self._vertices or mesh_type in self._edges:
            raise ValueError(f"Mesh type '{mesh_type}' already set for element '{self.name}'.")

        self._vertices[mesh_type] = vertices
        self._edges[mesh_type] = edges

    def get_mesh(self, mesh_type: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the vertices and edges for the given mesh type.

        Raises:
            KeyError if the mesh type does not exist.
        """
        try:
            return self._vertices[mesh_type], self._edges[mesh_type]
        except KeyError:
            raise KeyError(f"Mesh '{mesh_type}' not found in element '{self.name}'.")


class StructuralGroup(BaseModel):
    """
    A structural group that contains multiple structural elements and associated data.

    Attributes:
        name: Name of the structural group.
        structural_elements: Ordered list of StructuralElement objects.
        interpolation_method: Interpolation method used.
        scalar_field: Computed scalar field (1D or multi-D array), set after interpolation.
        mask: Optional mask for the scalar field (1D or multi-D array), set after interpolation.
    """
    name: str
    structural_elements: List['StructuralElement'] = Field(default_factory=list)
    _interpolation_method: Optional['InterpolationMethod'] = PrivateAttr(default=None)
    _scalar_field: Optional[np.ndarray] = PrivateAttr(default=None)
    _mask: Optional[np.ndarray] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, element_name: str) -> 'StructuralElement':
        for elem in self.structural_elements:
            if elem.name == element_name:
                return elem
        raise KeyError(f"Structural element '{element_name}' not found in group '{self.name}'.")

    @property
    def scalar_field(self) -> Optional[np.ndarray]:
        return self._scalar_field

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def set_scalar_field(self, field: np.ndarray):
        self._scalar_field = field

    def set_mask(self, mask_array: np.ndarray):
        self._mask = mask_array

    @property
    def interpolation_method(self) -> Optional['InterpolationMethod']:
        return self._interpolation_method

    def set_interpolation_method(self, method: Union[str, InterpolationMethod]):
        if isinstance(method, str):
            try:
                method = InterpolationMethod(method)
            except ValueError:
                valid_methods = [m.value for m in InterpolationMethod]
                raise ValueError(
                    f"'{method}' is not a valid interpolation method. "
                    f"Valid options are: {valid_methods}"
                )
        elif not isinstance(method, InterpolationMethod):
            raise TypeError(
                f"Interpolation method must be a string or InterpolationMethod enum, got {type(method)}"
            )

        self._interpolation_method = method


class StructuralFrame(BaseModel):
    structural_groups: List[StructuralGroup] = Field(default_factory=list)
    surface_points: pd.DataFrame
    orientations: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, group_name: str) -> StructuralGroup:
        for group in self.structural_groups:
            if group.name == group_name:
                return group
        raise KeyError(f"Structural group '{group_name}' not found.")

    def get_surface_points_for_element(self, element_name: str) -> pd.DataFrame:
        return self.surface_points[self.surface_points["formation"] == element_name]

    def get_orientations_for_element(self, element_name: str) -> Optional[pd.DataFrame]:
        if self.orientations is None:
            return None
        return self.orientations[self.orientations["formation"] == element_name]

    def get_surface_points_for_group(self, group_name: str) -> pd.DataFrame:
        group = self[group_name]
        element_names = [e.name for e in group.structural_elements]
        return self.surface_points[self.surface_points["formation"].isin(element_names)]

    def get_orientations_for_group(self, group_name: str) -> Optional[pd.DataFrame]:
        if self.orientations is None:
            return None
        group = self[group_name]
        element_names = [e.name for e in group.structural_elements]
        return self.orientations[self.orientations["formation"].isin(element_names)]

    def pretty_print(self):
        print("📦 Structural Frame Overview")
        print("────────────────────────────")
        print(f"• Number of structural groups: {len(self.structural_groups)}")
        print(f"• Total surface points: {len(self.surface_points)} entries")
        if self.orientations is not None:
            print(f"• Total orientations: {len(self.orientations)} entries")
        else:
            print("• Orientation data: None")

        print("\n🧱 Structural Groups:\n")

        for group in self.structural_groups:
            print(f"  ▶ Group: {group.name}")
            print(f"    ├─ Interpolation method: {group.interpolation_method}")

            # Count surface points for group
            group_element_names = [e.name for e in group.structural_elements]
            group_surface_points = self.surface_points[self.surface_points["formation"].isin(group_element_names)]
            print(f"    ├─ Surface points in group: {len(group_surface_points)}")

            # Count orientations for group
            if self.orientations is not None:
                group_orientations = self.orientations[self.orientations["formation"].isin(group_element_names)]
                print(f"    ├─ Orientations in group: {len(group_orientations)}")

            print(f"    └─ Structural Elements:")
            for elem in group.structural_elements:
                print(f"        • {elem.name}")
                if elem.color:
                    print(f"           - Color: {elem.color}")

                elem_surface_points = self.surface_points[self.surface_points["formation"] == elem.name]
                print(f"           - Surface points: {len(elem_surface_points)}")

                if self.orientations is not None:
                    elem_orientations = self.orientations[self.orientations["formation"] == elem.name]
                    print(f"           - Orientations: {len(elem_orientations)}")
            print()  # Blank line between groups



