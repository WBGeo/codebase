from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from pydantic import BaseModel, Field, PrivateAttr

from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    RegularGrid,
)

from core.structural_modeling_components.interpolator_functions.interpolator_parameters import (
    InterpolationMethod,
    InterpolationContext,
    OrdinaryKrigingParams,
    default_ok_params,
    RBFParams,
    default_rbf_params,
    GeoINRParams,
    LoopStructuralMethod,
    LoopStructuralParams,
    UniversalCoKrigingParams,
    UniversalKrigingParams,
    default_uk_params,
    InterpolationParameterSet, default_geo_inr_params,
)

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
BoolArray = npt.NDArray[np.bool_]

MeshType3 = str  # expected: "masked" | "unmasked" | "combined"
MeshType2 = str  # expected: "masked" | "unmasked"
MeshDict = Dict[str, npt.NDArray[np.generic]]


class StructuralElement(BaseModel):
    """
    A structural element within a structural group.

    Notes:
        This model stores most computed state in PrivateAttr to avoid Pydantic validation
        overhead for large numpy arrays. Properties provide read-only access.

    Attributes:
        name: Name of the element.
    """

    name: str

    # Computed / assigned later (kept private; accessed via properties)
    _scalar_value: Optional[float] = PrivateAttr(default=None)
    _id: Optional[int] = PrivateAttr(default=None)
    _color: Optional[str] = PrivateAttr(default=None)
    _vertices: Dict[str, npt.NDArray[np.generic]] = PrivateAttr(default_factory=dict)
    _edges: Dict[str, npt.NDArray[np.generic]] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

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
    def vertices(self) -> Dict[str, npt.NDArray[np.generic]]:
        return self._vertices

    @property
    def edges(self) -> Dict[str, npt.NDArray[np.generic]]:
        return self._edges

    def set_scalar_value(self, value: float) -> None:
        self._scalar_value = value

    def get_scalar_value(self) -> Optional[float]:
        return self._scalar_value

    def set_id(self, element_id: int) -> None:
        self._id = element_id

    def set_color(self, hex_color: str) -> None:
        self._color = hex_color

    def set_mesh(
            self,
            mesh_type: MeshType3,
            vertices: npt.NDArray[np.generic],
            edges: npt.NDArray[np.generic],
    ) -> None:
        """
        Set the vertices and edges for a specific mesh type.

        Args:
            mesh_type: One of {"masked", "unmasked", "combined"}.
            vertices: Vertex array for the mesh.
            edges: Edge/connectivity array for the mesh.

        Raises:
            ValueError: If mesh_type is not an allowed mesh type.
        """
        if mesh_type not in {"masked", "unmasked", "combined"}:
            raise ValueError(
                f"Invalid mesh type '{mesh_type}'. Allowed types are: masked, unmasked, combined."
            )

        self._vertices[mesh_type] = vertices
        self._edges[mesh_type] = edges

    def get_mesh(
            self, mesh_type: MeshType3
    ) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
        """
        Retrieve the vertices and edges for the given mesh type.

        Raises:
            KeyError: If the mesh type does not exist.
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
        structural_elements: Ordered list of :class:`StructuralElement` objects.
    """

    name: str
    structural_elements: List["StructuralElement"] = Field(default_factory=list)

    _interpolation_method: Optional[InterpolationMethod] = PrivateAttr(default=None)
    _scalar_field: Optional[npt.NDArray[np.floating]] = PrivateAttr(default=None)
    _interpolation_params: Optional[InterpolationParameterSet] = PrivateAttr(default=None)
    _mask: Optional[npt.NDArray[np.bool_]] = PrivateAttr(default=None)
    _context: Optional[InterpolationContext] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, element_name: str) -> "StructuralElement":
        for elem in self.structural_elements:
            if elem.name == element_name:
                return elem
        raise KeyError(
            f"Structural element '{element_name}' not found in group '{self.name}'."
        )

    @property
    def scalar_field(self) -> Optional[npt.NDArray[np.floating]]:
        return self._scalar_field

    def set_scalar_field(self, field: npt.NDArray[np.floating]) -> None:
        self._scalar_field = field

    def get_scalar_field(self) -> Optional[npt.NDArray[np.floating]]:
        return self._scalar_field

    @property
    def mask(self) -> Optional[npt.NDArray[np.bool_]]:
        return self._mask

    def set_mask(self, mask: npt.NDArray[np.bool_]) -> None:
        self._mask = mask

    def get_mask(self) -> Optional[npt.NDArray[np.bool_]]:
        return self._mask

    @property
    def interpolation_method(self) -> Optional[InterpolationMethod]:
        return self._interpolation_method

    def set_interpolation_method(self, method: Union[str, InterpolationMethod]) -> None:
        """
        Set interpolation method and initialize default parameter set based on context.

        Raises:
            ValueError: If a provided string does not map to a valid enum member.
            TypeError: If method is neither str nor InterpolationMethod.
            RuntimeError: If interpolation context has not been set yet.
        """
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
        self._interpolation_params = self._default_params_for_method(method, self._context)

    def set_interpolation_params(self, params: InterpolationParameterSet) -> None:
        self._interpolation_params = params

    def get_interpolation_params(self) -> Optional[InterpolationParameterSet]:
        return self._interpolation_params

    def configure_interpolation_params(self, **kwargs) -> None:
        """
        Update fields on the active interpolation parameter set.

        Raises:
            ValueError: If parameters have not been initialized.
            AttributeError: If any provided key is not a parameter field.
        """
        if self._interpolation_params is None:
            raise ValueError(
                "Interpolation parameters have not been initialized. "
                "Make sure to call set_interpolation_method() first."
            )

        for key, value in kwargs.items():
            if not hasattr(self._interpolation_params, key):
                raise AttributeError(
                    f"'{key}' is not a valid parameter for {type(self._interpolation_params).__name__}."
                )
            setattr(self._interpolation_params, key, value)

    def _default_params_for_method(
            self, method: InterpolationMethod, ctx: InterpolationContext
    ) -> Optional[InterpolationParameterSet]:
        """Internal helper to map a method to its default parameter set."""
        if method == InterpolationMethod.RADIAL_BASIS_FUNCTION:
            return default_rbf_params(ctx)

        if method == InterpolationMethod.ORDINARY_KRIGING:
            return default_ok_params(ctx)

        if method == InterpolationMethod.UNIVERSAL_COKRIGING:
            return UniversalCoKrigingParams()

        if method == InterpolationMethod.GEOINR:
            return default_geo_inr_params(ctx)

        if method == InterpolationMethod.LOOP_STRUCTURAL:
            return LoopStructuralParams()

        if method == InterpolationMethod.UNIVERSAL_KRIGING:
            return default_uk_params(ctx)

        return None

    def set_interpolation_context(self, ctx: InterpolationContext) -> None:
        self._context = ctx

    def update_interpolation_context(self, points: npt.NDArray[np.floating]) -> None:
        """
        Build/update the interpolation context from explicit group point data.

        Args:
            points: Array of shape (N, 3) representing xyz point constraints.

        Raises:
            ValueError: If fewer than 2 points are provided.
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
            n_points=int(points.shape[0]),
            mean_nn_distance=mean_nn_distance,
        )


class StructuralFrame(BaseModel):
    """
    A structural frame containing structural groups and optional fault information.

    Attributes:
        structural_groups: Ordered list of :class:`StructuralGroup` objects.

    Private state (set/derived elsewhere):
        _grid: Grid defining evaluation coordinates.
        _surface_points: Surface points for all elements.
        _orientations: Optional orientation data for all elements.
        _lith_block: Optional 3D lithology block.
        _fault_frame: Optional fault frame sharing the same grid.
        _fault_activity: Optional mapping of fault name -> the youngest affected group index.
    """

    structural_groups: List[StructuralGroup] = Field(default_factory=list)

    _grid: Optional[RegularGrid] = PrivateAttr(default=None)
    _surface_points: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _orientations: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _lith_block: Optional[npt.NDArray[np.generic]] = PrivateAttr(default=None)
    _fault_frame: Optional["FaultFrame"] = PrivateAttr(default=None)
    _fault_activity: Optional[dict[str, int]] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

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
    def lith_block(self) -> Optional[npt.NDArray[np.generic]]:
        return self._lith_block

    @property
    def fault_frame(self) -> Optional["FaultFrame"]:
        return self._fault_frame

    def set_fault_frame(self, fault_frame: Optional["FaultFrame"]) -> None:
        """
        Attach/detach a FaultFrame with consistency checks.

        Notes:
            The checks in this method enforce that both frames share the *same* grid
            instance and that the fault frame has a computed domain map.
        """
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

        if fault_frame.grid != self._grid:
            raise ValueError("FaultFrame and StructuralFrame must share the same grid.")

        # Check that the fault frame has a computed solution
        if fault_frame.domain_map is None:
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
            raise RuntimeError("Cannot set fault activity without a FaultFrame attached.")

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

    def get_LithBlock(self) -> Optional[npt.NDArray[np.generic]]:
        return self._lith_block

    def __getitem__(self, group_name: str) -> StructuralGroup:
        for group in self.structural_groups:
            if group.name == group_name:
                return group
        raise KeyError(f"Structural group '{group_name}' not found.")

    def get_element_by_name(self, element_name: str) -> StructuralElement:
        for group in self.structural_groups:
            for elem in group.structural_elements:
                if elem.name == element_name:
                    return elem
        raise KeyError(f"Structural element '{element_name}' not found.")

    def get_group_by_name(self, group_name: str) -> StructuralGroup:
        for group in self.structural_groups:
            if group.name == group_name:
                return group
        raise KeyError(f"Structural group '{group_name}' not found.")

    def detailed_report(self) -> None:
        """Print a human-readable report of the structural frame state."""
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
                param_dict = params.dict()  # pydantic model -> dict
            except Exception:
                param_dict = {}

            if param_dict:
                param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items())
                print(f"  ├─ Parameters: {param_str}")
            else:
                print(f"  ├─ Parameters: None")

            # Elements with color
            print("  ├─ Elements:")
            element_names: list[str] = []
            for elem in group.structural_elements:
                name = elem.name
                color = elem.color or "#AAAAAA"
                try:
                    r, g, b = tuple(int(color[i: i + 2], 16) for i in (1, 3, 5))
                    colored_name = f"\033[38;2;{r};{g};{b}m{name}\033[0m"
                except Exception:
                    colored_name = name
                element_names.append(colored_name)
            print(f"  │   {' | '.join(element_names)}")

            # Surface points in group
            element_names_raw = [e.name for e in group.structural_elements]
            group_surface_points = self.surface_points[
                self.surface_points["formation"].isin(element_names_raw)
            ]
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
            group_nr: int = 0,
            axis: str = "y",
            index: int = 0,
            plot_elements: bool = True,
    ) -> None:
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
        if axis == "y":
            data_slice = group.scalar_field[:, index, :].T
            extent = self._grid.extent[:4]
            xlabel, ylabel = "X", "Z"
        elif axis == "x":
            data_slice = group.scalar_field[index, :, :].T
            extent = (
                self._grid.extent[0],
                self._grid.extent[2],
                self._grid.extent[4],
                self._grid.extent[1],
            )
            xlabel, ylabel = "Y", "Z"
        elif axis == "z":
            data_slice = group.scalar_field[:, :, index].T
            extent = (
                self._grid.extent[0],
                self._grid.extent[2],
                self._grid.extent[1],
                self._grid.extent[3],
            )
            xlabel, ylabel = "X", "Y"
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        fig, ax = plt.subplots()

        # --- Scalar field ---
        im = ax.imshow(data_slice, extent=extent, origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, label="Scalar Value")

        # --- Element isolines ---
        if plot_elements:
            handles: list[plt.Line2D] = []

            for elem in group.structural_elements:
                cs = ax.contour(
                    data_slice,
                    levels=[elem.scalar_value],
                    colors=[elem.color],
                    linewidths=1.5,
                    extent=extent,
                    origin="lower",
                )

                # Create legend handle only once per element
                handles.append(
                    plt.Line2D([0], [0], color=elem.color, lw=1.5, label=elem.name)
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

    def plot_age_mask_section(self, group_nr: int = 0, axis: str = "y", index: int = 0) -> None:
        """
        Plot a section of an age mask (boolean field) along a specified axis at a given index.
        Assumes masks are stored in XYZ order matching grid.resolution.
        """
        group = self.structural_groups[group_nr]
        mask = group.get_mask()

        if mask is None:
            raise ValueError(f"Structural group at index {group_nr} has no age mask computed.")

        xmin, xmax, ymin, ymax, zmin, zmax = self._grid.extent

        if axis == "y":
            # X-Z at fixed Y=index
            mask_slice = mask[:, index, :].T  # (Z, X)
            extent = (xmin, xmax, zmin, zmax)
            xlabel, ylabel = "X", "Z"
        elif axis == "x":
            # Y-Z at fixed X=index
            mask_slice = mask[index, :, :].T  # (Z, Y)
            extent = (ymin, ymax, zmin, zmax)
            xlabel, ylabel = "Y", "Z"
        elif axis == "z":
            # X-Y at fixed Z=index
            mask_slice = mask[:, :, index].T  # (Y, X)
            extent = (xmin, xmax, ymin, ymax)
            xlabel, ylabel = "X", "Y"
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        fig, ax = plt.subplots()
        im = ax.imshow(
            mask_slice.astype(float),
            extent=extent,
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(["False", "True"])
        cbar.set_label("Age Mask")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Age Mask Section along {axis.upper()} at Index {index}\nGroup '{group.name}'")
        plt.show()


class FaultElement(BaseModel):
    """
    Represents a geological fault surface to be interpolated.

    Notes:
        This model intentionally stores many fields in PrivateAttr (numpy arrays, sets)
        to avoid Pydantic overhead.

    Publicly exposed properties provide read-only access.
    """

    _name: str = PrivateAttr()
    _scalar_value: Optional[float] = PrivateAttr(default=None)
    _scalar_field: Optional[npt.NDArray[np.floating]] = PrivateAttr(default=None)
    _color: str = PrivateAttr(default="#AAAAAA")  # Default color in hex format
    _separated_domains: Optional[Tuple[FrozenSet[int], FrozenSet[int]]] = PrivateAttr(
        default=None
    )
    _domain_pairs: Optional[FrozenSet[Tuple[int, int]]] = PrivateAttr(default=None)
    _vertices: Dict[str, npt.NDArray[np.generic]] = PrivateAttr(default_factory=dict)
    _edges: Dict[str, npt.NDArray[np.generic]] = PrivateAttr(default_factory=dict)
    _mask: Optional[npt.NDArray[np.bool_]] = PrivateAttr(default=None)

    def __init__(self, name: str, scalar_value: Optional[float] = None):
        super().__init__()
        self._name = name
        self._scalar_value = scalar_value

    def __repr__(self) -> str:
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
    def vertices(self) -> Dict[str, npt.NDArray[np.generic]]:
        return self._vertices

    @property
    def edges(self) -> Dict[str, npt.NDArray[np.generic]]:
        return self._edges

    @property
    def mask(self) -> Optional[npt.NDArray[np.bool_]]:
        return self._mask

    @property
    def scalar_field(self) -> Optional[npt.NDArray[np.floating]]:
        """Get the interpolated scalar field values on the fault surface."""
        return self._scalar_field

    def set_scalar_field(self, scalar_field: npt.NDArray[np.floating]) -> None:
        """Assign the interpolated scalar field values on the fault surface."""
        if not isinstance(scalar_field, np.ndarray):
            raise ValueError("Scalar field must be a numpy array.")
        self._scalar_field = scalar_field

    def set_scalar_value(self, value: float) -> None:
        """Assign scalar value used for interpolation."""
        self._scalar_value = value

    def set_color(self, color: str) -> None:
        """Set the display color for this fault in hex format."""
        if not isinstance(color, str) or not color.startswith("#") or len(color) != 7:
            raise ValueError("Color must be a valid hex string (e.g., '#RRGGBB').")
        self._color = color

    def set_mesh(
            self,
            mesh_type: MeshType2,
            vertices: npt.NDArray[np.generic],
            edges: npt.NDArray[np.generic],
    ) -> None:
        """
        Set the vertices and edges for a specific mesh type.

        Args:
            mesh_type: One of {"masked", "unmasked"}.
            vertices: Vertex array.
            edges: Edge/connectivity array.

        Raises:
            ValueError: If mesh_type is not an allowed mesh type.
        """
        if mesh_type not in {"masked", "unmasked"}:
            raise ValueError(
                f"Invalid mesh type '{mesh_type}'. Allowed types are: masked, unmasked, combined."
            )

        self._vertices[mesh_type] = vertices
        self._edges[mesh_type] = edges

    def get_mesh(
            self, mesh_type: MeshType2
    ) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
        """
        Retrieve the vertices and edges for the given mesh type.

        Raises:
            KeyError: If the mesh type does not exist.
        """
        try:
            return self._vertices[mesh_type], self._edges[mesh_type]
        except KeyError:
            raise KeyError(f"Mesh '{mesh_type}' not found in element '{self.name}'.")

    def set_domain_mask(self, mask: npt.NDArray[np.bool_]) -> None:
        """
        Store the boolean mask (True/False) separating two fault blocks.
        """
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise ValueError("Mask must be a boolean NumPy array.")
        self._mask = mask

    def get_domain_mask(self) -> npt.NDArray[np.bool_]:
        """
        Retrieve the fault mask (True = one block, False = other).

        Raises:
            ValueError: If no mask has been set.
        """
        if self._mask is None:
            raise ValueError(f"No mask set for fault '{self.name}'.")
        return self._mask

    def get_inverse_domain_mask(self) -> npt.NDArray[np.bool_]:
        """
        Get the inverse of the fault mask (opposite block).
        """
        return ~self.get_domain_mask()

    def get_separated_domains(self) -> Optional[Tuple[FrozenSet[int], FrozenSet[int]]]:
        """Get the tuple of separated domain IDs, if any."""
        return self._separated_domains

    def set_separated_domains(
            self, domain_ids: Tuple[FrozenSet[int], FrozenSet[int]]
    ) -> None:
        """Set the tuple of separated domain IDs."""
        self._separated_domains = domain_ids

    def get_domain_pairs(self) -> Optional[FrozenSet[Tuple[int, int]]]:
        """Return adjacent domain-id pairs across this fault surface."""
        return self._domain_pairs

    def set_domain_pairs(self, pairs: FrozenSet[Tuple[int, int]]) -> None:
        """
        Set adjacent domain-id pairs across this fault surface.

        Each pair must be (a, b) with a != b. Order will be normalized to (min, max).
        """
        if pairs is None:
            self._domain_pairs = None
            return

        if not isinstance(pairs, frozenset):
            pairs = frozenset(pairs)  # allow set/list input

        norm: set[Tuple[int, int]] = set()
        for p in pairs:
            if not (isinstance(p, tuple) and len(p) == 2):
                raise ValueError("Each domain pair must be a tuple (a, b).")
            a, b = int(p[0]), int(p[1])
            if a == b:
                continue
            norm.add((a, b) if a < b else (b, a))

        if not norm:
            raise ValueError(
                f"Fault '{self.name}' domain_pairs is empty after normalization."
            )

        self._domain_pairs = frozenset(norm)

    def domain_pairs_flat(self) -> set[int]:
        """Return all domain IDs touched by this fault's adjacency pairs."""
        if self._domain_pairs is None:
            return set()
        out: set[int] = set()
        for a, b in self._domain_pairs:
            out.add(a)
            out.add(b)
        return out

    def separated_domains_flat(self) -> set[int]:
        """Return all domain IDs this fault splits (from separated_domains)."""
        if self._separated_domains is None:
            return set()
        return set().union(*self._separated_domains)


class FaultFrame(BaseModel):
    """
    Container for managing fault elements and their relationships.

    Private attributes are used for numpy arrays / large data objects.
    """

    _fault_elements: List[FaultElement] = PrivateAttr()
    _fault_relations: Optional[npt.NDArray[np.bool_]] = PrivateAttr(default=None)
    _grid: Optional[RegularGrid] = PrivateAttr(default=None)
    _fault_surface_points_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _fault_orientations_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _domain_map: Optional[npt.NDArray[np.floating]] = PrivateAttr(default=None)
    _domain_masks: dict[int, npt.NDArray[np.bool_]] = PrivateAttr(default_factory=dict)

    def __init__(
            self,
            fault_elements: List[FaultElement],
            fault_relations: Optional[npt.NDArray[np.bool_]] = None,
    ):
        super().__init__()
        self._fault_elements = fault_elements
        self._fault_relations = fault_relations

    @property
    def fault_elements(self) -> List[FaultElement]:
        return self._fault_elements

    @property
    def grid(self) -> Optional[RegularGrid]:
        return self._grid

    @property
    def fault_surface_points_df(self) -> Optional[pd.DataFrame]:
        return self._fault_surface_points_df

    @property
    def fault_orientations_df(self) -> Optional[pd.DataFrame]:
        return self._fault_orientations_df

    @property
    def domain_map(self) -> Optional[npt.NDArray[np.floating]]:
        return self._domain_map

    @property
    def domain_masks(self) -> dict[int, npt.NDArray[np.bool_]]:
        """Boolean masks for each final domain ID."""
        return self._domain_masks

    def get_element_by_name(self, name: str) -> Optional[FaultElement]:
        """Retrieve a fault element by its name."""
        return next((f for f in self._fault_elements if f.name == name), None)

    def set_surface_points_df(self, df: pd.DataFrame) -> None:
        self._fault_surface_points_df = df

    def set_orientations_df(self, df: pd.DataFrame) -> None:
        self._fault_orientations_df = df

    def get_surface_points_df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame containing all fault surface points."""
        return self._fault_surface_points_df

    def get_orientations_df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame containing all fault surface orientations."""
        return self._fault_orientations_df

    def get_surface_points_for_element(self, name: str) -> pd.DataFrame:
        if self._fault_surface_points_df is not None:
            return self._fault_surface_points_df[
                self._fault_surface_points_df["formation"] == name
                ]
        return pd.DataFrame()

    def get_orientations_for_element(self, name: str) -> pd.DataFrame:
        if self._fault_orientations_df is not None:
            return self._fault_orientations_df[
                self._fault_orientations_df["formation"] == name
                ]
        return pd.DataFrame()

    def set_domain_map(self, domain_map: npt.NDArray[np.floating]) -> None:
        self._domain_map = domain_map

    def set_grid(self, grid: RegularGrid) -> None:
        """Set the grid for spatial context."""
        if not isinstance(grid, RegularGrid):
            raise ValueError("Grid must be an instance of RegularGrid.")
        self._grid = grid

    def detailed_report(self) -> None:
        """Print a human-readable report of the fault frame state."""
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
                r, g, b = tuple(int(color[i: i + 2], 16) for i in (1, 3, 5))
                colored_name = f"\033[38;2;{r};{g};{b}m{name}\033[0m"
            except Exception:
                colored_name = name

            sp_count = len(self.get_surface_points_for_element(name))
            ori_count = len(self.get_orientations_for_element(name))

            print(f"  ├─ {colored_name}")
            print(f"  │   ├─ Surface points: {sp_count}")
            print(f"  │   └─ Orientations: {ori_count}")
        print("")

