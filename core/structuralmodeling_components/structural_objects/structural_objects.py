from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, Optional, List, Union, FrozenSet, Tuple
from pydantic import BaseModel, Field, PrivateAttr
from enum import Enum
from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


#%%

class InterpolationMethod(str, Enum):
    ORDINARY_KRIGING = "Ordinary Kriging"
    RADIAL_BASIS_FUNCTION = "Radial Basis Function"
    UNIVERSAL_COKRIGING = "Universal Co-Kriging"
    GEOINR = "GeoINR"
    LOOP_STRUCTURAL = "Loop Structural"


class InterpolationContext(BaseModel):
    data_scale: tuple[float, float, float]
    n_points: int
    mean_nn_distance: float


class OrdinaryKrigingParams(BaseModel):
    """
    Configuration parameters for Ordinary Kriging interpolation.

    Attributes:
        variogram_model: The type of variogram model to use. Common choices are "spherical", "exponential", or "gaussian".
        range: The range of the variogram model, typically the distance at which spatial correlation becomes negligible.
        sill: The sill value (plateau) of the variogram model, representing the maximum semi-variance.
        nugget: The nugget effect, representing microscale variation or measurement error.
        anisotropy_scaling_x: Scaling factor for the x-axis in 3D kriging, used to account for horizontal anisotropy.
        anisotropy_scaling_y: Scaling factor for the y-axis in 3D kriging, used to account for horizontal anisotropy.
        anisotropy_scaling_z: Scaling factor for the z-axis in 3D kriging, used to account for vertical anisotropy.
        neighbors: Optional; the number of nearest neighbors to use in kriging. If None, all points are used.
    """
    variogram_model: str = Field("gaussian",
                                 description="Type of variogram model (e.g., spherical, exponential, gaussian).")
    range: float = Field(500.0, description="Range of the variogram (distance at which correlation tapers off).")
    sill: float = Field(1.0, description="Sill of the variogram (max variance level).")
    nugget: float = Field(0.0, description="Nugget effect (variance at zero distance).")
    anisotropy_scaling_x: float = Field(1.0, description="Scaling factor for the y-axis in 3D kriging.")
    anisotropy_scaling_y: float = Field(1.0, description="Scaling factor for the y-axis in 3D kriging.")
    anisotropy_scaling_z: float = Field(1.0, description="Scaling factor for the z-axis in 3D kriging.")
    neighbors: Optional[int] = Field(None,
                                     description="Number of nearest neighbors to use in kriging. If None, uses all points.")


def default_ok_params(ctx: InterpolationContext) -> OrdinaryKrigingParams:

    sx, sy, sz = ctx.data_scale
    npts = ctx.n_points
    nn_dist = ctx.mean_nn_distance

    # Variogram
    variogram_model = "exponential"

    # Range
    range_ = np.clip(
        20 * nn_dist,
        0.1 * max(sx, sy, sz),
        0.8 * max(sx, sy, sz),
    )

    sill = 1.0
    nugget = 0.0

    # Anisotropy scaling per axis (scale relative to largest dimension)
    max_scale = max(sx, sy, sz)
    anisotropy_scaling_x = np.clip(sx / max_scale, 0.05, 1.0)
    anisotropy_scaling_y = np.clip(sy / max_scale, 0.05, 1.0)
    anisotropy_scaling_z = np.clip(sz / max_scale, 0.05, 1.0)

    # Rotation angles (degrees)
    # Default 0 → no rotation, but could be adapted if you detect tilted layers
    anisotropy_angle_x = 0.0
    anisotropy_angle_y = 0.0
    anisotropy_angle_z = 0.0

    # Neighbors (moving window)
    if npts < 20:
        neighbors = None  # global kriging
    else:
        neighbors = min(200, max(30, npts // 10))

    return OrdinaryKrigingParams(
        variogram_model=variogram_model,
        range=range_,
        sill=sill,
        nugget=nugget,
        anisotropy_scaling_x=anisotropy_scaling_x,
        anisotropy_scaling_y=anisotropy_scaling_y,
        anisotropy_scaling_z=anisotropy_scaling_z,
        neighbors=neighbors,
    )

class RBFParams(BaseModel):
    """
    Parameters for Radial Basis Function (RBF) interpolation.

    Attributes:
        kernel: The radial basis function kernel to use. Common options include 'linear', 'cubic', 'thin_plate', etc.
        smoothing: Smoothing parameter. Larger values allow more smoothing of the interpolation surface.
        epsilon: Shape parameter for kernels like multiquadric or inverse multiquadric.
        neighbors: Optional number of nearest neighbors to use. If None, all input_data points are considered.
    """

    kernel: str = Field("thin_plate_spline",
                        description="Radial basis function kernel. Common options: 'linear', 'cubic', 'thin_plate'.")
    smoothing: float = Field(0,
                             description="Smoothing parameter for RBF. Higher values increase smoothing (0 = exact fit).")
    epsilon: Optional[float] = Field(None,
                                     description="Shape parameter for certain kernels like multiquadric or inverse multiquadric.")
    neighbors: Optional[int] = Field(None,
                                     description="Number of nearest neighbors to use. If None, all points are used.")


def default_rbf_params(ctx: InterpolationContext) -> RBFParams:
    # Characteristic length scale
    L = max(ctx.data_scale)

    kernel = "thin_plate_spline"

    epsilon = None
    if kernel in {"multiquadric", "inverse_multiquadric", "gaussian"}:
        epsilon = ctx.mean_nn_distance

    # Neighbors scale with problem size
    neighbors = None
    if ctx.n_points > 5000:
        neighbors = min(500, int(5 * ctx.n_points ** (2 / 3)))

    return RBFParams(
        kernel="thin_plate_spline",
        smoothing=0.05,  # categorical scalar field → smooth by default
        epsilon=epsilon,
        neighbors=neighbors
    )


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


InterpolationParameterSet = Union[OrdinaryKrigingParams,
RBFParams,
GeoINRParams,
LoopStructuralParams,
UniversalCoKrigingParams]


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
    def vertices(self) -> Optional[dict]:
        return self._vertices

    @property
    def edges(self) -> Optional[dict]:
        return self._edges

    # Controlled setters
    def set_scalar_value(self, value: float):
        self._scalar_value = value

    def get_scalar_value(self) -> Optional[float]:
        return self._scalar_value

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
    A structural group that contains multiple structural elements and associated input_data.

    Attributes:
        name: Name of the structural group.
        structural_elements: Ordered list of StructuralElement objects.
        interpolation_method: Interpolation method used.
        scalar_field: Computed scalar field (1D or multi-D array), set after interpolation.
        mask: Optional mask for the scalar field (1D or multi-D array), set after interpolation.
        interpolation_params: Optional parameters for the interpolation method.
        context: Optional context for interpolation default parameters (e.g., grid extent, number of points).
    """
    name: str
    structural_elements: List['StructuralElement'] = Field(default_factory=list)
    _interpolation_method: Optional['InterpolationMethod'] = PrivateAttr(default=None)
    _scalar_field: Optional[np.ndarray] = PrivateAttr(default=None)
    _interpolation_params: Optional[InterpolationParameterSet] = PrivateAttr(default=None)
    _mask: Optional[np.ndarray] = PrivateAttr(default=None)
    _context: Optional[InterpolationContext] = PrivateAttr(default=None)

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

    def set_scalar_field(self, field: np.ndarray):
        self._scalar_field = field

    def get_scalar_field(self) -> Optional[np.ndarray]:
        return self._scalar_field

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def set_mask(self, mask: np.ndarray):
        self._mask = mask

    def get_mask(self) -> Optional[np.ndarray]:
        return self._mask

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

        if self._context is None:
            raise RuntimeError(
                "Interpolation context not initialized. "
                "Call update_interpolation_context() first."
            )

        self._interpolation_method = method
        self._interpolation_params = self._default_params_for_method(
            method,
            self._context
        )

        # # Set default parameters when method is set
        # if method == InterpolationMethod.ORDINARY_KRIGING:
        #     self._interpolation_params = OrdinaryKrigingParams()
        # elif method == InterpolationMethod.RADIAL_BASIS_FUNCTION:
        #     self._interpolation_params = RBFParams()
        # elif method == InterpolationMethod.UNIVERSAL_COKRIGING:
        #     self._interpolation_params = UniversalCoKrigingParams()
        # elif method == InterpolationMethod.GEOINR:
        #     self._interpolation_params = GeoINRParams()
        # elif method == InterpolationMethod.LOOP_STRUCTURAL:
        #     self._interpolation_params = LoopStructuralParams()
        # else:
        #     self._interpolation_params = None  # fallback

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

    def _default_params_for_method(
            self,
            method: InterpolationMethod,
            ctx: InterpolationContext
    ):
        if method == InterpolationMethod.RADIAL_BASIS_FUNCTION:
            return default_rbf_params(ctx)

        if method == InterpolationMethod.ORDINARY_KRIGING:
            return default_ok_params(ctx)

        if method == InterpolationMethod.UNIVERSAL_COKRIGING:
            return UniversalCoKrigingParams()

        if method == InterpolationMethod.GEOINR:
            return GeoINRParams()

        if method == InterpolationMethod.LOOP_STRUCTURAL:
            return LoopStructuralParams()

        return None

    def set_interpolation_context(self, ctx: InterpolationContext):
        self._context = ctx

    def update_interpolation_context(self, points: np.ndarray):
        """
        Build or update the interpolation context from explicit group point data.
        """
        if points.shape[0] < 2:
            raise ValueError("Not enough points to build interpolation context.")

        p10 = np.percentile(points, 10, axis=0)
        p90 = np.percentile(points, 90, axis=0)

        data_scale = (
            float(p90[0] - p10[0]),
            float(p90[1] - p10[1]),
            float(p90[2] - p10[2]),
        )

        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        mean_nn_distance = float(np.median(dists[:, 1]))

        self._context = InterpolationContext(
            data_scale=data_scale,
            n_points=points.shape[0],
            mean_nn_distance=mean_nn_distance,
        )


class StructuralFrame(BaseModel):
    """
    A structural frame that contains multiple structural groups and associated input_data.
    Attributes:
        structural_groups: Ordered list of StructuralGroup objects.
        _grid: RegularGrid for spatial context.
        _surface_points: DataFrame with surface points for all elements.
        _orientations: Optional DataFrame with orientation input_data for all elements.
        _lith_block: Optional 3D NumPy array representing resulting lithology block.
    """
    structural_groups: List[StructuralGroup] = Field(default_factory=list)

    _grid: Optional[RegularGrid] = PrivateAttr(default=None)
    _surface_points: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _orientations: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _lith_block: Optional[np.ndarray] = PrivateAttr(default=None)
    _fault_frame: Optional[FaultFrame] = PrivateAttr(default=None)
    _fault_activity: Optional[dict[str, int]] = PrivateAttr(default=None)


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

    @property
    def fault_frame(self) -> Optional[FaultFrame]:
        return self._fault_frame

    def set_fault_frame(self, fault_frame: Optional["FaultFrame"]) -> None:
        if fault_frame is None:
            self._fault_frame = None
            self._fault_activity = None
            return

        if self._grid is None:
            raise ValueError(
                "StructuralFrame grid must be set before assigning a FaultFrame."
            )

        if fault_frame.grid is None:
            raise ValueError(
                "FaultFrame grid must be set before being assigned to a StructuralFrame."
            )

        if fault_frame.grid is not self._grid:
            raise ValueError(
                "FaultFrame and StructuralFrame must share the same grid instance."
            )

        # Check that the fault frame has a computed solution
        if fault_frame._domain_map is None:
            raise ValueError(
                "Cannot assign a FaultFrame without a computed solution (domain map missing). "
                "Please run `compute_fault_domains(fault_frame)` first."
            )

        self._fault_frame = fault_frame

        # Initialize fault activity to default (all groups affected)
        self._fault_activity = {fault.name: 0 for fault in fault_frame.fault_elements}

    @property
    def fault_activity(self) -> Optional[dict[str, int]]:
        if self._fault_activity is None:
            return None
        return dict(self._fault_activity)  # defensive copy

    @property
    def fault_activity_verbose(self) -> Optional[dict[str, dict[str, int | str]]]:
        if self._fault_activity is None:
            return None

        return {
            fault: {
                "youngest_group_index": idx,
                "youngest_group_name": self.structural_groups[idx].name,
            }
            for fault, idx in self._fault_activity.items()
        }

    def set_fault_activity_by_index(self, fault_name: str, youngest_group_idx: int) -> None:
        if self._fault_frame is None or self._fault_activity is None:
            raise RuntimeError(
                "Cannot set fault activity without a FaultFrame attached."
            )

        if fault_name not in self._fault_activity:
            raise KeyError(f"Fault '{fault_name}' not found in fault frame.")

        if youngest_group_idx < 0 or youngest_group_idx >= len(self.structural_groups):
            raise ValueError(
                f"max_group_idx must be in range [0, {len(self.structural_groups) - 1}]."
            )

        self._fault_activity[fault_name] = youngest_group_idx

    def set_fault_activity_by_group(self, fault_name: str, group_name: str) -> None:
        for idx, group in enumerate(self.structural_groups):
            if group.name == group_name:
                self.set_fault_activity_by_index(fault_name, idx)
                return

        raise KeyError(f"Structural group '{group_name}' not found.")

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

    def detailed_report(self):
        print("📦 Structural Frame — Detailed Report")
        print("─────────────────────────────────────")
        print(f"• Number of structural groups: {len(self.structural_groups)}")
        print(f"• Total surface points: {len(self.surface_points)} entries")
        if self.orientations is not None:
            print(f"• Total orientations: {len(self.orientations)} entries\n")
        else:
            print("• Orientation input_data: None\n")

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

        # --- Fault frame report ---
        if self._fault_frame is not None:
            print("⚡ Fault Frame Present")
            num_faults = len(self._fault_frame.fault_elements)
            print(f"• Number of faults: {num_faults}")

            if self.fault_activity_verbose is not None:
                print("• Fault activity (max group affected):")
                for fault_name, group_name in self.fault_activity_verbose.items():
                    idx = self._fault_activity[fault_name]
                    print(f"  ├─ {fault_name}: group index {idx} → {group_name}")
            else:
                print("• Fault activity: default (all groups affected)")
        else:
            print("⚡ No fault frame assigned")

    def plot_scalar_field_section(
            self,
            group_nr=0,
            axis='y',
            index=0,
            plot_elements=True
    ):
        """
        Plot a section of a scalar field along a specified axis at a given index.
        Optionally overlay contour lines for structural elements.
        """

        # --- Pre checks ---
        group = self.structural_groups[group_nr]

        if group.scalar_field is None:
            raise ValueError(
                f"Structural group at index {group_nr} has no scalar field computed."
            )

        # --- Extract slice and extent ---
        if axis == 'y':
            data_slice = group.scalar_field[:, index, :].T
            extent = self._grid.extent[:4]
            xlabel, ylabel = 'X', 'Z'
        elif axis == 'x':
            data_slice = group.scalar_field[index, :, :].T
            extent = self._grid.extent[[0, 2, 4, 1]]
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 'z':
            data_slice = group.scalar_field[:, :, index].T
            extent = self._grid.extent[[0, 2, 1, 3]]
            xlabel, ylabel = 'X', 'Y'
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        fig, ax = plt.subplots()

        # --- Scalar field ---
        im = ax.imshow(
            data_slice,
            extent=extent,
            origin='lower',
            cmap='viridis'
        )
        plt.colorbar(im, ax=ax, label='Scalar Value')

        # --- Element isolines ---
        if plot_elements:
            handles = []

            for elem in group.structural_elements:
                cs = ax.contour(
                    data_slice,
                    levels=[elem.scalar_value],
                    colors=[elem.color],
                    linewidths=1.5,
                    extent=extent,
                    origin='lower'
                )

                # Create legend handle only once per element
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        color=elem.color,
                        lw=1.5,
                        label=elem.name
                    )
                )

            if handles:
                ax.legend(handles=handles, title="Structural Elements", loc="lower left")

        # --- Labels and title ---
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Scalar Field Section along {axis.upper()} at Index {index}\n"
            f"Group '{group.name}'"
        )

        plt.show()

    def plot_age_mask_section(
            self,
            group_nr=0,
            axis='y',
            index=0
    ):
        """
        Plot a section of an age mask (boolean field) along a specified axis
        at a given index.
        """

        # --- Pre checks ---
        group = self.structural_groups[group_nr]

        if getattr(group, "_mask", None) is None:
            raise ValueError(
                f"Structural group at index {group_nr} has no age mask computed."
            )

        # --- Extract slice and extent ---
        if axis == 'y':
            mask_slice = group.get_mask()[:, index, :].T
            extent = self._grid.extent[:4]
            xlabel, ylabel = 'X', 'Z'
        elif axis == 'x':
            mask_slice = group.get_mask()[index, :, :].T
            extent = self._grid.extent[[0, 2, 4, 1]]
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 'z':
            mask_slice = group.get_mask()[:, :, index].T
            extent = self._grid.extent[[0, 2, 1, 3]]
            xlabel, ylabel = 'X', 'Y'
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        fig, ax = plt.subplots()

        # --- Plot boolean mask ---
        im = ax.imshow(
            mask_slice.astype(float),
            extent=extent,
            origin='lower',
            cmap='gray',
            vmin=0,
            vmax=1
        )

        # Optional colorbar for clarity
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['False', 'True'])
        cbar.set_label('Age Mask')

        # --- Labels and title ---
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Age Mask Section along {axis.upper()} at Index {index}\n"
            f"Group '{group.name}'"
        )

        plt.show()


class FaultElement(BaseModel):
    """
    Represents a geological fault surface to be interpolated.

    Attributes:
        name (str): Unique identifier for the fault.
        scalar_value (Optional[float]): Value used in scalar field interpolation.
        scalar_field (Optional[np.ndarray]): Interpolated scalar field values on the fault surface.
        affects_groups (Optional[List[str]]): Structural groups offset by this fault.
        color (str): Display color for the fault in hex format (default: "#AAAAAA").
        separated_domains (Optional[tuple[int, int]]): Tuple of domain IDs separated by the fault.
        vertices (np.ndarray): Coordinates of the fault surface vertices.
        edges (np.ndarray): Connectivity of the fault surface edges.
        mask (Optional[np.ndarray]): Boolean mask separating two fault blocks.
    """
    _name: str = PrivateAttr()
    _scalar_value: Optional[float] = PrivateAttr(default=None)
    _scalar_field: Optional[np.ndarray] = PrivateAttr(default=None)
    _color: str = PrivateAttr(default="#AAAAAA")  # Default color in hex format
    _separated_domains: Optional[Tuple[FrozenSet[int], FrozenSet[int]]] = PrivateAttr(default=None)
    _vertices: Optional[np.ndarray] = PrivateAttr(default_factory=None)
    _edges: Optional[np.ndarray] = PrivateAttr(default_factory=None)
    _mask: Optional[np.ndarray] = PrivateAttr(default=None)

    def __init__(self, name: str, scalar_value: Optional[float] = None):
        super().__init__()
        self._name = name
        self._scalar_value = scalar_value

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

    def get_separated_domains(self) -> Optional[Tuple[FrozenSet[int], FrozenSet[int]]]:
        """Get the tuple of separated domain IDs, if any."""
        return self._separated_domains

    def set_separated_domains(self, domain_ids: Tuple[FrozenSet[int], FrozenSet[int]]):
        """Set the tuple of separated domain IDs."""
        self._separated_domains = domain_ids

    def separated_domains_flat(self) -> set[int]:
        """Return all domain IDs this fault splits."""
        return set().union(*self._separated_domains)


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
    _grid: Optional[RegularGrid] = PrivateAttr(default=None)
    _fault_surface_points_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _fault_orientations_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _domain_map: Optional[np.ndarray] = PrivateAttr(default=None)
    _domain_masks: dict[int, np.ndarray] = PrivateAttr(default_factory=dict)

    def __init__(self, fault_elements: List[FaultElement], fault_relations: Optional[np.ndarray] = None):
        super().__init__()
        self._fault_elements = fault_elements

    @property
    def fault_elements(self) -> List[FaultElement]:
        return self._fault_elements

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

    def get_element_by_name(self, name: str) -> Optional[FaultElement]:
        """Retrieve a fault element by its name."""
        return next((f for f in self._fault_elements if f.name == name), None)

    def add_fault_element(self, fault: FaultElement):
        """Append a fault and update the relations matrix accordingly."""
        self._fault_elements.append(fault)
        self._fault_relations = self._generate_default_relations()

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

    def plot_fault_domain_section(
            self,
            axis='y',
            index=0,
            plot_faults=True
    ):
        """
        Plot a section of the fault domain map along a specified axis at a given index.
        Optionally overlay isolines for fault elements.
        """

        # --- Pre checks ---
        if self._domain_map is None:
            raise ValueError("FaultFrame has no domain_map computed.")

        if self._grid is None:
            raise ValueError("FaultFrame has no grid associated.")

        if plot_faults:
            for fault in self._fault_elements:
                if fault._scalar_field is None or fault._scalar_value is None:
                    raise ValueError(
                        f"Fault '{fault.name}' has no scalar field or scalar value computed."
                    )

        # --- Extract slice and extent ---
        if axis == 'y':
            domain_slice = self._domain_map[:, index, :].T
            extent = self._grid.extent[:4]
            xlabel, ylabel = 'X', 'Z'
        elif axis == 'x':
            domain_slice = self._domain_map[index, :, :].T
            extent = self._grid.extent[[0, 2, 4, 1]]
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 'z':
            domain_slice = self._domain_map[:, :, index].T
            extent = self._grid.extent[[0, 2, 1, 3]]
            xlabel, ylabel = 'X', 'Y'
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        fig, ax = plt.subplots()

        # --- Plot domain map ---
        # --- Discrete colormap for fault domains ---
        domain_values = np.unique(domain_slice)
        domain_values = domain_values[~np.isnan(domain_values)]  # safety

        n_domains = len(domain_values)

        base_cmap = plt.get_cmap("Set3")
        colors = base_cmap(np.linspace(0, 1, n_domains))

        cmap = ListedColormap(colors)
        bounds = np.append(domain_values, domain_values[-1] + 1)
        norm = BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(
            domain_slice,
            extent=extent,
            origin='lower',
            cmap=cmap,
            norm=norm
        )

        cbar = plt.colorbar(im, ax=ax, ticks=domain_values)
        cbar.set_label("Fault Domain")

        # --- Fault isolines ---
        if plot_faults:
            handles = []

            for fault in self._fault_elements:
                ax.contour(
                    fault._scalar_field[:, index, :].T if axis == 'y' else
                    fault._scalar_field[index, :, :].T if axis == 'x' else
                    fault._scalar_field[:, :, index].T,
                    levels=[fault._scalar_value],
                    colors=[fault.color],
                    linewidths=1.5,
                    extent=extent,
                    origin='lower'
                )

                handles.append(
                    plt.Line2D(
                        [0], [0],
                        color=fault.color,
                        lw=1.5,
                        label=fault.name
                    )
                )

            if handles:
                ax.legend(handles=handles, title="Faults", loc="lower left")

        # --- Labels and title ---
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Fault Domain Section along {axis.upper()} at Index {index}"
        )

        plt.show()

