import numpy as np
import pandas as pd

from typing import Dict, Tuple, Optional, List, Union
from pydantic import BaseModel, Field, PrivateAttr
from enum import Enum
from core.grids.grid_classes import RegularGrid


#%%

class InterpolationMethod(str, Enum):
    ORDINARY_KRIGING = "Ordinary Kriging"
    RADIAL_BASIS_FUNCTION = "Radial Basis Function"
    UNIVERSAL_COKRIGING = "Universal Co-Kriging"
    GEOINR = "GeoINR"
    LOOP_STRUCTURAL = "Loop Structural"
    GEO_ML = "GeoML"
    IDW = "Inverse Distance Weighting"


class OrdinaryKrigingParams(BaseModel):
    """
    Configuration parameters for Ordinary Kriging interpolation.

    Attributes:
        variogram_model: The type of variogram model to use. Common choices are "spherical", "exponential", or "gaussian".
        range: The range of the variogram model, typically the distance at which spatial correlation becomes negligible.
        sill: The sill value (plateau) of the variogram model, representing the maximum semi-variance.
        nugget: The nugget effect, representing microscale variation or measurement error.
        anisotropy_scaling_z: Scaling factor for the z-axis in 3D kriging, used to account for vertical anisotropy.
        neighbors: Optional; the number of nearest neighbors to use in kriging. If None, all points are used.
    """
    variogram_model: str = Field("gaussian",
                                 description="Type of variogram model (e.g., spherical, exponential, gaussian).")
    range: float = Field(500.0, description="Range of the variogram (distance at which correlation tapers off).")
    sill: float = Field(1.0, description="Sill of the variogram (max variance level).")
    nugget: float = Field(0.0, description="Nugget effect (variance at zero distance).")
    anisotropy_scaling_z: float = Field(1.0, description="Scaling factor for the z-axis in 3D kriging.")
    neighbors: Optional[int] = Field(None,
                                     description="Number of nearest neighbors to use in kriging. If None, uses all points.")


class RBFParams(BaseModel):
    """
    Parameters for Radial Basis Function (RBF) interpolation.

    Attributes:
        kernel: The radial basis function kernel to use. Common options include 'linear', 'cubic', 'thin_plate', etc.
        smoothing: Smoothing parameter. Larger values allow more smoothing of the interpolation surface.
        epsilon: Shape parameter for kernels like multiquadric or inverse multiquadric.
        neighbors: Optional number of nearest neighbors to use. If None, all data points are considered.
    """

    kernel: str = Field("linear",
                        description="Radial basis function kernel. Common options: 'linear', 'cubic', 'thin_plate'.")
    smoothing: int = Field(0,
                           description="Smoothing parameter for RBF. Higher values increase smoothing (0 = exact fit).")
    epsilon: int = Field(1,
                         description="Shape parameter for certain kernels like multiquadric or inverse multiquadric.")
    neighbors: Optional[int] = Field(None,
                                     description="Number of nearest neighbors to use. If None, all points are used.")


class GeoINRParams(BaseModel):
    """
    Parameters for GeoINR interpolation.

    Attributes:
        beta: Regularization or weighting parameter controlling the influence of constraints
              in the neural representation.
    """

    beta: int = Field(1,
                      escription="Regularization parameter controlling the influence of geometric constraints in the model.")


class LoopStructuralMethod(str, Enum):
    FDI = "FDI"
    PLI = "PLI"


class LoopStructuralParams(BaseModel):
    """
    Parameters for LoopStructural interpolation.

    Attributes:
        interpolator_type: The type of interpolator to use in LoopStructural.
                           Must be either 'FDI' (Finite Difference Interpolator)
                           or 'PLI' (Piecewise Linear Interpolator).
    """
    interpolator_type: LoopStructuralMethod = Field(
        default=LoopStructuralMethod.FDI,
        description="Type of LoopStructural interpolator. Choose 'FDI' or 'PLI'."
    )


class UniversalCoKrigingParams(BaseModel):
    """
    Placeholder class for Universal Co-Kriging interpolation parameters.

    Currently, Universal Co-Kriging does not require any parameters,
    but this class is in place to support future configuration needs.
    """
    pass


class GeoMLParams(BaseModel):
    """
    Placeholder class for GeoML interpolation parameters.

    Currently, GeoML does not require any parameters,
    but this class is in place to support future configuration needs.
    """
    pass


class IDWParams(BaseModel):
    """
    Parameters for Inverse Distance Weighting (IDW) interpolation.
    Attributes:
        power: The power parameter for IDW, controlling the influence of distance on weights.
        neighbors: Optional; the number of nearest neighbors to consider. If None, all points are used.
    """
    power: float = Field(2.0,
                         description="The power parameter for IDW, controlling the influence of distance on weights")
    neighbors: Optional[int] = Field(None,
                        description="Number of nearest neighbors to use. If None, all points are used.")
    anisotropy_scaling: Optional[tuple[float, float, float]] = Field((1.0, 1.0, 1.0),
                        description="Anisotropy scaling factors for x, y, z axes. Default is (1.0, 1.0, 1.0).")


InterpolationParameterSet = Union[OrdinaryKrigingParams,
RBFParams,
GeoINRParams,
LoopStructuralParams,
UniversalCoKrigingParams,
GeoMLParams,
IDWParams]


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
        # if mesh_type in self._vertices or mesh_type in self._edges:
        #     raise ValueError(f"Mesh type '{mesh_type}' already set for element '{self.name}'.")

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
        interpolation_params: Optional parameters for the interpolation method.
    """
    name: str
    structural_elements: List['StructuralElement'] = Field(default_factory=list)
    _interpolation_method: Optional['InterpolationMethod'] = PrivateAttr(default=None)
    _scalar_field: Optional[np.ndarray] = PrivateAttr(default=None)
    _mask: Optional[np.ndarray] = PrivateAttr(default=None)
    _interpolation_params: Optional[InterpolationParameterSet] = PrivateAttr(default=None)

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

        # Set default parameters when method is set
        if method == InterpolationMethod.ORDINARY_KRIGING:
            self._interpolation_params = OrdinaryKrigingParams()
        elif method == InterpolationMethod.RADIAL_BASIS_FUNCTION:
            self._interpolation_params = RBFParams()
        elif method == InterpolationMethod.UNIVERSAL_COKRIGING:
            self._interpolation_params = UniversalCoKrigingParams()
        elif method == InterpolationMethod.GEOINR:
            self._interpolation_params = GeoINRParams()
        elif method == InterpolationMethod.LOOP_STRUCTURAL:
            self._interpolation_params = LoopStructuralParams()
        elif method == InterpolationMethod.GEO_ML:
            self._interpolation_params = GeoMLParams()
        elif method == InterpolationMethod.IDW:
            self._interpolation_params = IDWParams()
        else:
            self._interpolation_params = None  # fallback

    def set_interpolation_params(self, params: InterpolationParameterSet):
        self._interpolation_params = params

    def get_interpolation_params(self) -> Optional[InterpolationParameterSet]:
        return self._interpolation_params

    def configure_interpolation_params(self, **kwargs):
        if self._interpolation_params is None:
            raise ValueError("Interpolation parameters have not been initialized. "
                             "Make sure to call set_interpolation_method() first.")

        for key, value in kwargs.items():
            if not hasattr(self._interpolation_params, key):
                raise AttributeError(
                    f"'{key}' is not a valid parameter for {type(self._interpolation_params).__name__}."
                )
            setattr(self._interpolation_params, key, value)


class StructuralFrame(BaseModel):
    """
    A structural frame that contains multiple structural groups and associated data.
    """
    structural_groups: List[StructuralGroup] = Field(default_factory=list)

    _grid: Optional[RegularGrid] = PrivateAttr(default=None)
    _surface_points: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _orientations: Optional[pd.DataFrame] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    # Properties to access the private attributes
    @property
    def grid(self) -> RegularGrid:
        return self._grid

    @property
    def surface_points(self) -> pd.DataFrame:
        return self._surface_points

    @property
    def orientations(self) -> Optional[pd.DataFrame]:
        return self._orientations

    # Getters
    def get_surface_points_for_element(self, element_name: str) -> pd.DataFrame:
        return self._surface_points[self._surface_points["formation"] == element_name]

    def get_orientations_for_element(self, element_name: str) -> Optional[pd.DataFrame]:
        if self._orientations is None:
            return None
        return self._orientations[self._orientations["formation"] == element_name]

    def get_surface_points_for_group(self, group_name: str) -> pd.DataFrame:
        group = self[group_name]
        element_names = [e.name for e in group.structural_elements]
        return self._surface_points[self._surface_points["formation"].isin(element_names)]

    def get_orientations_for_group(self, group_name: str) -> Optional[pd.DataFrame]:
        if self._orientations is None:
            return None
        group = self[group_name]
        element_names = [e.name for e in group.structural_elements]
        return self._orientations[self._orientations["formation"].isin(element_names)]

    def __getitem__(self, group_name: str) -> StructuralGroup:
        for group in self.structural_groups:
            if group.name == group_name:
                return group
        raise KeyError(f"Structural group '{group_name}' not found.")

    def summary(self):
        print("📦 Structural Frame Summary")
        print("────────────────────────────")
        print(f"• Groups: {len(self.structural_groups)}\n")

        for group in self.structural_groups:
            print(f"▶ {group.name} — {group.interpolation_method}")
            print("  Elements:")
            for elem in group.structural_elements:
                name = elem.name
                color = elem.color or "#AAAAAA"
                try:
                    # Use ANSI escape for color (truecolor if supported)
                    r, g, b = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
                    print(f"    \033[38;2;{r};{g};{b}m{name}\033[0m")
                except Exception:
                    print(f"    {name} (color: {color})")
            print()  # blank line between groups

    def detailed_report(self):
        print("📦 Structural Frame — Detailed Report")
        print("─────────────────────────────────────")
        print(f"• Number of structural groups: {len(self.structural_groups)}")
        print(f"• Total surface points: {len(self.surface_points)} entries")
        if self.orientations is not None:
            print(f"• Total orientations: {len(self.orientations)} entries\n")
        else:
            print("• Orientation data: None\n")

        for group in self.structural_groups:
            print(f"▶ {group.name}")
            print(f"  ├─ Interpolation method: {group.interpolation_method}")

            # Interpolation parameters
            try:
                params = group.get_interpolation_params()
                param_dict = params.dict()
            except Exception:
                param_dict = {}

            if param_dict:
                param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items())
                print(f"  ├─ Parameters: {param_str}")
            else:
                print(f"  ├─ Parameters: None")

            # Elements with color
            print("  ├─ Elements:")
            element_names = []
            for elem in group.structural_elements:
                name = elem.name
                color = elem.color or "#AAAAAA"
                try:
                    r, g, b = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
                    colored_name = f"\033[38;2;{r};{g};{b}m{name}\033[0m"
                except Exception:
                    colored_name = name
                element_names.append(colored_name)
            print(f"  │   {' | '.join(element_names)}")

            # Surface points in group
            element_names_raw = [e.name for e in group.structural_elements]
            group_surface_points = self.surface_points[self.surface_points["formation"].isin(element_names_raw)]
            print(f"  ├─ Surface points in group: {len(group_surface_points)}")

            # Per-element surface point counts
            sp_counts = [
                f"{name}: {len(self.surface_points[self.surface_points['formation'] == name])}"
                for name in element_names_raw
            ]
            print(f"  ├─ Per-element surface points: {', '.join(sp_counts)}")

            # Per-element orientation counts, if available
            if self.orientations is not None:
                ori_counts = [
                    f"{name}: {len(self.orientations[self.orientations['formation'] == name])}"
                    for name in element_names_raw
                ]
                print(f"  └─ Per-element orientations: {', '.join(ori_counts)}\n")
            else:
                print(f"  └─ Per-element orientations: N/A\n")
