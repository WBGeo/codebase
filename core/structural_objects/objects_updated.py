import numpy as np
import pandas as pd

from typing import Dict, Tuple, Optional, List, Union
from pydantic import BaseModel, Field, PrivateAttr
from enum import Enum
from core.grids.grid_classes import RegularGrid
from core.utility.surface_mesh_extraction import marching_cubes_new
import gempy as gp


#%%

class InterpolationMethod(str, Enum):
    ORDINARY_KRIGING = "Ordinary Kriging"
    RADIAL_BASIS_FUNCTION = "Radial Basis Function"
    UNIVERSAL_COKRIGING = "Universal Co-Kriging"
    GEOINR = "GeoINR"
    LOOP_STRUCTURAL = "Loop Structural"


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


InterpolationParameterSet = Union[OrdinaryKrigingParams,
RBFParams,
GeoINRParams,
LoopStructuralParams,
UniversalCoKrigingParams,
GeoMLParams]


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
    # _scalar_value: Optional[float] = PrivateAttr(default=None)
    _id: Optional[int] = PrivateAttr(default=None)
    _color: Optional[str] = PrivateAttr(default=None)
    # _vertices: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)
    # _edges: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)

    _scalar_values_by_domain: Dict[int, float] = PrivateAttr(default_factory=dict)
    _meshes_by_domain: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    # Read-only properties
    # @property
    # def scalar_value(self) -> Optional[float]:
    #     return self._scalar_value

    @property
    def id(self) -> Optional[int]:
        return self._id

    @property
    def color(self) -> Optional[str]:
        return self._color

    # @property
    # def vertices(self) -> Optional[np.ndarray]:
    #     return self._vertices
    #
    # @property
    # def edges(self) -> Optional[np.ndarray]:
    #     return self._edges

    # Controlled setters
    # def set_scalar_value(self, value: float):
    #     self._scalar_value = value

    def set_id(self, element_id: int):
        self._id = element_id

    def set_color(self, hex_color: str):
        self._color = hex_color

    # def set_mesh(self, mesh_type: str, vertices: np.ndarray, edges: np.ndarray):
    #     """
    #     Set the vertices and edges for a specific mesh type (e.g., 'masked', 'unmasked', 'combined').
    #
    #     Raises:
    #         ValueError if mesh_type is not one of the allowed types or already exists.
    #     """
    #     if mesh_type not in {"masked", "unmasked", "combined"}:
    #         raise ValueError(f"Invalid mesh type '{mesh_type}'. Allowed types are: masked, unmasked, combined.")
    #     # if mesh_type in self._vertices or mesh_type in self._edges:
    #     #     raise ValueError(f"Mesh type '{mesh_type}' already set for element '{self.name}'.")
    #
    #     self._vertices[mesh_type] = vertices
    #     self._edges[mesh_type] = edges
    #
    # def get_mesh(self, mesh_type: str) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Retrieve the vertices and edges for the given mesh type.
    #
    #     Raises:
    #         KeyError if the mesh type does not exist.
    #     """
    #     try:
    #         return self._vertices[mesh_type], self._edges[mesh_type]
    #     except KeyError:
    #         raise KeyError(f"Mesh '{mesh_type}' not found in element '{self.name}'.")

    # -------- Domain-aware scalar values --------
    def set_scalar_value_for_domain(self, domain_id: int, value: float) -> None:
        self._scalar_values_by_domain[domain_id] = float(value)

    def get_scalar_value_for_domain(self, domain_id: int) -> float:
        try:
            return self._scalar_values_by_domain[domain_id]
        except KeyError:
            raise KeyError(f"Element '{self.name}': no scalar value stored for domain {domain_id}.")

    def scalar_values_by_domain(self) -> Dict[int, float]:
        return dict(self._scalar_values_by_domain)

    # -------- Domain-aware meshes --------
    def set_mesh_for_domain(self, domain_id: int, mesh_type: str, vertices: np.ndarray, faces: np.ndarray) -> None:
        if mesh_type not in {"masked", "unmasked", "combined"}:
            raise ValueError("mesh_type must be one of {'masked','unmasked','combined'}.")
        store = self._meshes_by_domain.setdefault(domain_id, {})
        if mesh_type in store:
            raise ValueError(f"Mesh '{mesh_type}' for domain {domain_id} already set on '{self.name}'.")
        store[mesh_type] = (vertices, faces)

    def get_mesh_for_domain(self, domain_id: int, mesh_type: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return self._meshes_by_domain[domain_id][mesh_type]
        except KeyError:
            raise KeyError(
                f"Element '{self.name}': mesh '{mesh_type}' not found for domain {domain_id}."
            )

    def meshes_for_domain(self, domain_id: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        try:
            return dict(self._meshes_by_domain[domain_id])
        except KeyError:
            raise KeyError(f"Element '{self.name}': no meshes stored for domain {domain_id}.")

    def domains_with_meshes(self) -> Tuple[int, ...]:
        return tuple(sorted(self._meshes_by_domain.keys()))


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
    # _mask: Optional[np.ndarray] = PrivateAttr(default=None)
    _interpolation_params: Optional[InterpolationParameterSet] = PrivateAttr(default=None)

    # NEW: domain-aware stores
    _scalar_fields_by_domain: Dict[int, np.ndarray] = PrivateAttr(default_factory=dict)
    _masks_by_domain: Dict[int, np.ndarray] = PrivateAttr(default_factory=dict)

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

    # @property
    # def mask(self) -> Optional[np.ndarray]:
    #     return self._mask

    def set_scalar_field(self, field: np.ndarray):
        self._scalar_field = field

    # def set_mask(self, mask_array: np.ndarray):
    #     self._mask = mask_array

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

    # -------- Domain-aware API --------
    def set_scalar_field_for_domain(self, domain_id: int, field: np.ndarray) -> None:
        self._scalar_fields_by_domain[domain_id] = field

    def get_scalar_field_for_domain(self, domain_id: int) -> np.ndarray:
        try:
            return self._scalar_fields_by_domain[domain_id]
        except KeyError:
            raise KeyError(f"Group '{self.name}': no scalar field stored for domain {domain_id}.")

    def scalar_fields_by_domain(self) -> Dict[int, np.ndarray]:
        """Read-only view (shallow copy) if you want all at once."""
        return dict(self._scalar_fields_by_domain)

    def set_mask_for_domain(self, domain_id: int, mask: np.ndarray) -> None:
        self._masks_by_domain[domain_id] = mask

    def get_mask_for_domain(self, domain_id: int) -> np.ndarray:
        try:
            return self._masks_by_domain[domain_id]
        except KeyError:
            raise KeyError(f"Group '{self.name}': no mask stored for domain {domain_id}.")

    def masks_by_domain(self) -> Dict[int, np.ndarray]:
        return dict(self._masks_by_domain)

    def domains_with_results(self) -> Tuple[int, ...]:
        """Domains where both scalar field and mask exist."""
        return tuple(sorted(set(self._scalar_fields_by_domain) & set(self._masks_by_domain)))


class StructuralFrame(BaseModel):
    """
    A structural frame that contains multiple structural groups and associated data.
    Attributes:
        structural_groups: Ordered list of StructuralGroup objects.
        _grid: RegularGrid for spatial context.
        _surface_points: DataFrame with surface points for all elements.
        _orientations: Optional DataFrame with orientation data for all elements.
        _lith_block: Optional 3D NumPy array representing resulting lithology block.
    """
    structural_groups: List[StructuralGroup] = Field(default_factory=list)

    _grid: Optional[RegularGrid] = PrivateAttr(default=None)
    _surface_points: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _orientations: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _lith_block: Optional[np.ndarray] = PrivateAttr(default=None)

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

    @property
    def lith_block(self) -> Optional[np.ndarray]:
        return self._lith_block

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

    def get_LithBlock(self) -> Optional[np.ndarray]:
        return self._lith_block

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


class FaultElement(BaseModel):
    """
    Represents a geological fault surface to be interpolated.

    Attributes:
        name (str): Unique identifier for the fault.
        scalar_value (Optional[float]): Value used in scalar field interpolation.
        scalar_field (Optional[np.ndarray]): Interpolated scalar field values on the fault surface.
        affects_groups (Optional[List[str]]): Structural groups offset by this fault.
        color (str): Display color for the fault in hex format (default: "#AAAAAA").
        vertices (np.ndarray): Coordinates of the fault surface vertices.
        edges (np.ndarray): Connectivity of the fault surface edges.
        mask (Optional[np.ndarray]): Boolean mask separating two fault blocks.
    """
    _name: str = PrivateAttr()
    _scalar_value: Optional[float] = PrivateAttr(default=None)
    _scalar_field: Optional[np.ndarray] = PrivateAttr(default=None)
    _affects_groups: Optional[List[str]] = PrivateAttr(default=None)
    _color: str = PrivateAttr(default="#AAAAAA")  # Default color in hex format
    _vertices: np.ndarray = PrivateAttr(default_factory=None)
    _edges: np.ndarray = PrivateAttr(default_factory=None)
    _mask: Optional[np.ndarray] = PrivateAttr(default=None)

    def __init__(self, name: str, scalar_value: Optional[float] = None,
                 affects_groups: Optional[List[str]] = None):
        super().__init__()
        self._name = name
        self._scalar_value = scalar_value
        self._affects_groups = affects_groups

    def __repr__(self):
        return f"FaultElement(name='{self.name}')"

    @property
    def name(self) -> str:
        return self._name

    @property
    def scalar_value(self) -> Optional[float]:
        return self._scalar_value

    @property
    def color(self) -> str:
        return self._color

    @property
    def vertices(self) -> Optional[np.ndarray]:
        return self._vertices

    @property
    def edges(self) -> Optional[np.ndarray]:
        return self._edges

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    @property
    def scalar_field(self) -> Optional[np.ndarray]:
        """Get the interpolated scalar field values on the fault surface."""
        return self._scalar_field

    def set_scalar_field(self, scalar_field: np.ndarray):
        """Assign the interpolated scalar field values on the fault surface."""
        if not isinstance(scalar_field, np.ndarray):
            raise ValueError("Scalar field must be a numpy array.")
        self._scalar_field = scalar_field

    def set_scalar_value(self, value: float):
        """Assign scalar value used for interpolation."""
        self._scalar_value = value

    def set_color(self, color: str):
        """Set the display color for this fault in hex format."""
        if not isinstance(color, str) or not color.startswith("#") or len(color) != 7:
            raise ValueError("Color must be a valid hex string (e.g., '#RRGGBB').")
        self._color = color

    def set_vertices(self, vertices: np.ndarray):
        """Assign the coordinates of the fault surface vertices."""
        if not isinstance(vertices, np.ndarray):
            raise ValueError("Vertices must be a numpy array.")
        self._vertices = vertices

    def set_edges(self, edges: np.ndarray):
        """Assign the connectivity of the fault surface edges."""
        if not isinstance(edges, np.ndarray):
            raise ValueError("Edges must be a numpy array.")
        self._edges = edges

    def set_domain_mask(self, mask: np.ndarray):
        """
            Store the boolean mask (True/False) separating two fault blocks.
            """
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise ValueError("Mask must be a boolean NumPy array.")
        self._mask = mask

    def get_domain_mask(self) -> np.ndarray:
        """
            Retrieve the fault mask (True = one block, False = other).
            """
        if self._mask is None:
            raise ValueError(f"No mask set for fault '{self.name}'.")
        return self._mask

    def get_inverse_domain_mask(self) -> np.ndarray:
        """
            Get the inverse of the fault mask (opposite block).
            """
        return ~self.get_domain_mask()

    @property
    def affects_groups(self) -> Optional[List[str]]:
        return self._affects_groups

    def set_affects_groups(self, groups: List[str]):
        """Specify which structural groups are offset by this fault."""
        self._affects_groups = groups


class FaultFrame(BaseModel):
    """
    Container for managing fault elements and their relationships.

    Attributes:
        fault_elements (List[FaultElement]): Ordered list of faults (oldest to youngest).
        fault_relations (np.ndarray): Boolean matrix [younger_idx, older_idx] = True if younger offsets older.
        grid (Optional[RegularGrid]): Regular grid for spatial context.
        fault_surface_points_df (Optional[pd.DataFrame]): DataFrame with fault surface points.
        fault_orientations_df (Optional[pd.DataFrame]): DataFrame with fault surface orientations.
        domain_map (Optional[np.ndarray]): Map of fault domains for scalar field interpolation.
    """
    _fault_elements: List[FaultElement] = PrivateAttr()
    _fault_relations: np.ndarray = PrivateAttr()
    _grid: Optional[RegularGrid] = PrivateAttr(default=None)
    _fault_surface_points_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _fault_orientations_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _domain_map: Optional[np.ndarray] = PrivateAttr(default=None)
    _domain_masks: dict[int, np.ndarray] = PrivateAttr(default_factory=dict)

    def __init__(self, fault_elements: List[FaultElement], fault_relations: Optional[np.ndarray] = None):
        super().__init__()
        self._fault_elements = fault_elements
        self._fault_relations = (
            fault_relations if fault_relations is not None else self._generate_default_relations()
        )

    @property
    def fault_elements(self) -> List[FaultElement]:
        return self._fault_elements

    @property
    def fault_relations(self) -> np.ndarray:
        return self._fault_relations

    @property
    def grid(self) -> RegularGrid:
        return self._grid

    @property
    def fault_surface_points_df(self) -> Optional[pd.DataFrame]:
        return self._fault_surface_points_df

    @property
    def fault_orientations_df(self) -> Optional[pd.DataFrame]:
        return self._fault_orientations_df

    @property
    def domain_map(self) -> Optional[np.ndarray]:
        return self._domain_map

    @property
    def domain_masks(self) -> dict[int, np.ndarray]:
        """Boolean masks for each final domain ID."""
        return self._domain_masks

    def _generate_default_relations(self) -> np.ndarray:
        """By default, younger faults affect all older ones."""
        n = len(self._fault_elements)
        relations = np.zeros((n, n), dtype=bool)
        for younger in range(n):
            for older in range(younger):
                relations[younger, older] = True
        return relations

    def get_element_by_name(self, name: str) -> Optional[FaultElement]:
        """Retrieve a fault element by its name."""
        return next((f for f in self._fault_elements if f.name == name), None)

    def add_fault_element(self, fault: FaultElement):
        """Append a fault and update the relations matrix accordingly."""
        self._fault_elements.append(fault)
        self._fault_relations = self._generate_default_relations()

    def set_fault_relation(self, younger_idx: int, older_idx: int, value: bool):
        """Manually modify a fault-fault relation."""
        self._fault_relations[younger_idx, older_idx] = value

    def set_surface_points_df(self, df: pd.DataFrame):
        self._fault_surface_points_df = df

    def set_orientations_df(self, df: pd.DataFrame):
        self._fault_orientations_df = df

    def get_surface_points_df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame containing all fault surface points."""
        return self._fault_surface_points_df

    def get_orientations_df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame containing all fault surface orientations."""
        return self._fault_orientations_df

    def get_surface_points_for_element(self, name: str) -> pd.DataFrame:
        if self._fault_surface_points_df is not None:
            return self._fault_surface_points_df[self._fault_surface_points_df["formation"] == name]
        return pd.DataFrame()

    def get_orientations_for_element(self, name: str) -> pd.DataFrame:
        if self._fault_orientations_df is not None:
            return self._fault_orientations_df[self._fault_orientations_df["formation"] == name]
        return pd.DataFrame()

    def set_domain_map(self, domain_map: np.ndarray):
        self._domain_map = domain_map

    def set_grid(self, grid: RegularGrid):
        """Set the grid for spatial context."""
        if not isinstance(grid, RegularGrid):
            raise ValueError("Grid must be an instance of RegularGrid.")
        self._grid = grid

    def describe_relations(self) -> List[str]:
        """Return a readable list of which faults offset which others."""
        descriptions = []
        names = [f.name for f in self._fault_elements]
        for y in range(len(names)):
            for o in range(len(names)):
                if self._fault_relations[y, o]:
                    descriptions.append(f"{names[y]} offsets {names[o]}")
        return descriptions

    def detailed_report(self):
        print("🧱 Fault Frame — Detailed Report")
        print("────────────────────────────────")
        print(f"• Number of faults: {len(self.fault_elements)}")

        if self._grid:
            print(f"• Grid extent: {self._grid.extent}")
            print(f"• Grid resolution: {self._grid.resolution}")
        else:
            print("• Grid: Not set")

        if self._fault_surface_points_df is not None:
            print(f"• Surface points: {len(self._fault_surface_points_df)} entries")
        else:
            print("• Surface points: None")

        if self._fault_orientations_df is not None:
            print(f"• Orientations: {len(self._fault_orientations_df)} entries\n")
        else:
            print("• Orientations: None\n")

        print("▶ Faults (youngest → oldest):")
        for fault in reversed(self.fault_elements):
            name = fault.name
            color = fault.color or "#888888"
            try:
                r, g, b = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
                colored_name = f"\033[38;2;{r};{g};{b}m{name}\033[0m"
            except Exception:
                colored_name = name

            sp_count = len(self.get_surface_points_for_element(name))
            ori_count = len(self.get_orientations_for_element(name))

            print(f"  ├─ {colored_name}")
            print(f"  │   ├─ Surface points: {sp_count}")
            print(f"  │   └─ Orientations: {ori_count}")
        print("")

    def check_fault_crosscuts_via_isovalue_bands(
            self,
            thickness_world: float | None = None,
            voxels: float = 1.0,
            use_gradient: bool = True,
    ) -> None:
        """
        Detect cross-cutting faults by overlapping 'isovalue bands' around each fault's own scalar isovalue.

        For each fault i with scalar field φ_i and isovalue L_i (fault.scalar_value):
            band_i = |φ_i - L_i| <= tol_scalar_i

        If any voxel satisfies band_i & band_j for i!=j, the faults crosscut.

        Parameters
        ----------
        thickness_world : float | None
            Desired half-thickness (in world units, e.g. meters) of each isovalue band.
            If None, it is computed as `voxels * min(grid.spacing)`.
        voxels : float
            If `thickness_world` is None, use this many voxels (based on min spacing) as the half-thickness.
        use_gradient : bool
            If True (recommended), convert the world thickness to scalar tolerance per-fault using
            that fault's median gradient magnitude: tol_scalar_i = thickness_world * median(|∇φ_i|).
            If False, assumes φ is approximately a signed distance function and uses tol_scalar_i = thickness_world.

        Raises
        ------
        ValueError
            If any pair of faults' bands overlap (i.e., cross-cut is detected).
        """

        if self._grid is None:
            raise ValueError("FaultFrame grid must be set.")
        spacing = getattr(self._grid, "spacing", None)
        if spacing is None:
            raise ValueError("Grid.spacing must be defined to compute band thickness.")

        # Determine band thickness in world units (meters)
        if thickness_world is None:
            thickness_world = float(voxels) * float(np.min(spacing))

        # Collect faults that have scalar fields and an isovalue
        faults = []
        for f in self._fault_elements:
            field = getattr(f, "scalar_field", None)
            level = getattr(f, "scalar_value", None)
            if field is None or level is None:
                continue
            if not isinstance(field, np.ndarray) or field.size == 0:
                continue
            faults.append((f.name, field, float(level)))

        if len(faults) < 2:
            return  # nothing to compare

        # Compute per-fault scalar tolerances from world thickness
        tol_scalar = []
        for nm, fld, _ in faults:
            if use_gradient:
                # gradient in scalar units per meter along each axis
                gx, gy, gz = np.gradient(fld, *spacing, edge_order=1)
                grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
                med = float(np.nanmedian(grad_mag)) if np.isfinite(grad_mag).any() else 0.0
                # guard against tiny gradients
                if med <= 1e-12:
                    med = 1e-12
                tol_scalar.append(thickness_world * med)
            else:
                # assume φ is approx. signed distance
                tol_scalar.append(thickness_world)

        # Build boolean bands once
        bands = []
        for (nm, fld, level), ts in zip(faults, tol_scalar):
            band = np.abs(fld - level) <= ts
            bands.append((nm, band))

        # Pairwise overlap test
        for i in range(len(bands)):
            name_i, band_i = bands[i]
            if not band_i.any():
                continue
            for j in range(i + 1, len(bands)):
                name_j, band_j = bands[j]
                if not band_j.any():
                    continue
                if np.any(band_i & band_j):
                    raise ValueError(f"❌ Fault '{name_i}' crosscuts fault '{name_j}' (isovalue-band overlap).")

    def interpolate_group_universal_cokriging_for_faults(
            self,
            element: FaultElement,
            grid,
            fault_surface_points_df: pd.DataFrame,
            fault_orientations_points_df: Optional[pd.DataFrame] = None,
    ) -> None:
        if fault_surface_points_df.empty:
            raise ValueError(f"No surface points provided for {element.name}")

        if fault_orientations_points_df is None or fault_orientations_points_df.empty:
            raise ValueError(f"No orientations provided for {element.name}")

        # --- GemPy conversion ---
        surface_data = gp.data.surface_points.SurfacePointsTable.from_arrays(
            x=fault_surface_points_df.X.to_numpy(),
            y=fault_surface_points_df.Y.to_numpy(),
            z=fault_surface_points_df.Z.to_numpy(),
            names=fault_surface_points_df.formation.to_numpy(),
            nugget=np.zeros(len(fault_surface_points_df)),
            name_id_map=None,
        )

        orientation_data = gp.data.orientations.OrientationsTable.from_arrays(
            x=fault_orientations_points_df.X.to_numpy(),
            y=fault_orientations_points_df.Y.to_numpy(),
            z=fault_orientations_points_df.Z.to_numpy(),
            G_x=fault_orientations_points_df.G_x.to_numpy(),
            G_y=fault_orientations_points_df.G_y.to_numpy(),
            G_z=fault_orientations_points_df.G_z.to_numpy(),
            names=fault_orientations_points_df.formation.to_numpy(),
            nugget=np.zeros(len(fault_orientations_points_df)),
            name_id_map=surface_data.name_id_map,
        )

        gempy_structural_frame = gp.data.structural_frame.StructuralFrame.from_data_tables(
            surface_data, orientation_data
        )

        geo_model = gp.create_geomodel(
            project_name="random",
            extent=grid.extent,
            resolution=grid.resolution,
            structural_frame=gempy_structural_frame,
        )

        mapping = {"fault_group": [element.name]}
        gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=mapping)

        gp.compute_model(geo_model)

        # Set scalar value at surface and field
        element.set_scalar_value(
            float(geo_model.solutions.raw_arrays.scalar_field_at_surface_points[0][0])
        )

        # NOTE: transpose if that’s how your marching/plotting expects it
        sf = geo_model.solutions.raw_arrays.scalar_field_matrix[0].reshape(tuple(grid.resolution)).T
        element.set_scalar_field(sf)

        # Domain mask convention (positive side = True)
        element.set_domain_mask(element.scalar_field > element.scalar_value)

    def generate_fault_domains(self) -> None:
        """
        Interpolates all faults and generates a domain map across the model grid.
        Relies on fault.domain_mask being set by the interpolator_func.
        """
        if not self._grid:
            raise ValueError("Grid must be set before domain generation.")
        if self._fault_surface_points_df is None:
            raise ValueError("Fault surface points must be set.")

        # Initialize single-domain model
        domain_map = np.zeros(self._grid.resolution, dtype=int)
        domain_id_counter = 1

        temp_ids = []  # Track temporary domain IDs before remapping

        # Interpolate faults from youngest to oldest
        for i, fault in enumerate(reversed(self._fault_elements)):  # Youngest first
            name = fault.name

            # Extract surface point/orientation data for this fault
            points = self.get_surface_points_for_element(name)
            orientations = self.get_orientations_for_element(name)

            if points.empty:
                raise ValueError(f"❌ No surface points found for fault '{name}'.")

            # Run interpolation (sets scalar field, scalar value, mask internally)
            self.interpolate_group_universal_cokriging_for_faults(self.get_element_by_name(name),
                                                                  self._grid,
                                                                  fault_surface_points_df=points,
                                                                  fault_orientations_points_df=orientations)

            if fault.get_domain_mask() is None:
                raise ValueError(f"❌ Interpolator did not set domain_mask for fault '{name}'.")

            fault_mask = fault.get_domain_mask()
            new_domain_map = domain_map.copy()

            # For each existing domain, split it if affected by this fault
            for existing_id in np.unique(domain_map):
                current_mask = domain_map == existing_id
                overlap = current_mask & fault_mask

                if np.any(overlap):
                    # Assign a temporary large ID
                    new_domain_map[overlap] = 9999 + domain_id_counter
                    temp_ids.append(9999 + domain_id_counter)
                    domain_id_counter += 1

            domain_map = new_domain_map

        # Remap domain IDs to consecutive values starting from 0
        unique_ids = np.unique(domain_map)
        remap = {old: new for new, old in enumerate(unique_ids)}
        remapped_map = np.vectorize(remap.get)(domain_map)
        self._domain_map = remapped_map

        # Remap domain IDs to consecutive values starting from 0
        unique_ids = np.unique(domain_map)
        remap = {old: new for new, old in enumerate(unique_ids)}
        remapped_map = np.vectorize(remap.get)(domain_map)
        self._domain_map = remapped_map

        #  Store per-domain masks
        self._domain_masks = {}
        for uid in np.unique(remapped_map):
            self._domain_masks[uid] = remapped_map == uid

        # Extrac surfaces meshes for faults
        for i, fault in enumerate(reversed(self._fault_elements)):
            vertices, edges = marching_cubes_new(fault.scalar_field.T,
                                                 [fault.scalar_value],
                                                 self._grid.spacing,
                                                 self._grid.extent)

            fault.set_vertices(vertices[0])
            fault.set_edges(edges[0])

        # 🔎 After all faults are processed
        self.check_fault_crosscuts_via_isovalue_bands()
