from __future__ import annotations

from enum import Enum, StrEnum
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from pydantic import BaseModel, Field, model_validator
from pydantic_numpy import NpNDArrayFp64, NpNDArrayBool, NpNDArrayInt32

from core.utility.pydantic_bridge import PandasDataFrame

from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    ModelGrid,
    OctreeCells,
    OctreeGrid,
    Resolution3,
    cell_array_shape,
    cell_centers,
    to_regular,
)

from core.structural_modeling_components.interpolator_functions.interpolator_parameters import (
    InterpolationMethod,
    InterpolationContext,
    default_ok_params,
    default_rbf_params,
    default_uk_params,
    default_pli_params,
    default_fdi_params,
    default_geo_inr_params,
    UCKParams,
    InterpolationParameterSet, default_uck_params,
    FDIParams,
    GeoINRParams,
    OKParams,
    PLIParams,
    RBFParams,
    UKParams,
)

#: Parameter class of each interpolation method. Several classes share the same fields, so a parameter
#: set read back from JSON is only unambiguous together with its group's method.
PARAMS_CLASS_BY_METHOD = {
    InterpolationMethod.RADIAL_BASIS_FUNCTION: RBFParams,
    InterpolationMethod.ORDINARY_KRIGING: OKParams,
    InterpolationMethod.UNIVERSAL_KRIGING: UKParams,
    InterpolationMethod.UNIVERSAL_COKRIGING: UCKParams,
    InterpolationMethod.FINITE_DIFFERENCES: FDIParams,
    InterpolationMethod.PIECEWISE_LINEAR: PLIParams,
    InterpolationMethod.GEOINR: GeoINRParams,
}

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray = NpNDArrayFp64
IntArray = npt.NDArray[np.integer]
BoolArray = NpNDArrayBool

MeshDict = Dict[str, NpNDArrayFp64]


def grid_report_lines(grid: Optional[ModelGrid], cells: Optional[OctreeCells]) -> List[str]:
    """
    The grid part of a frame's ``detailed_report``, as printable lines.

    Regular grid: extent and resolution. Octree: extent, base resolution, max
    level and the equivalent regular grid, the gap-mitigation and cell-balance
    settings, and, once computed, the stored cells per level and their share of
    the finest cells.
    """
    if grid is None:
        return ["• Grid: Not set"]
    if not isinstance(grid, OctreeGrid):
        return [f"• Grid extent: {grid.extent}", f"• Grid resolution: {grid.resolution}"]
    finest = tuple(int(n) for n in grid.finest_resolution)
    n_finest = int(np.prod(finest))
    lines = [
        f"• Grid extent: {grid.extent}",
        f"• Octree: base resolution {tuple(grid.base_resolution)}, max level {grid.max_level} "
        f"(equivalent regular grid {'×'.join(map(str, finest))} = {n_finest:,} cells)",
        f"  ├─ Gap mitigation: {grid.gap_mitigation.value} · cell balance: {grid.cell_balance.value}",
    ]
    if cells is None:
        lines.append("  └─ Cells: not computed")
        return lines
    levels = np.bincount(np.asarray(cells.level, dtype=np.int64), minlength=grid.max_level + 1)
    lines.append(f"  ├─ Cells: {len(cells):,} ({100 * len(cells) / n_finest:.1f} % of the finest cells)")
    lines.append("  └─ Cells per level: " + " · ".join(f"L{level} {count:,}" for level, count in enumerate(levels)))
    return lines


class _PerCellResults:
    """
    Grid-independent access to per-cell result arrays, shared by both frames.

    Per-cell arrays (scalar fields, masks, ``lith_block``, ``domain_map``, ...)
    have shape ``grid.resolution`` on a :class:`RegularGrid` and ``(len(cells),)``
    on an :class:`OctreeGrid`, where ``cells`` holds the realized octree cells.
    Expects ``grid`` and ``cells`` attributes on the class it is mixed into.
    """

    def cell_array_shape(self) -> Tuple[int, ...]:
        """Shape every per-cell array of this frame must have."""
        return cell_array_shape(self.grid, self.cells)

    def cell_centers(self) -> npt.NDArray[np.float64]:
        """Cell centers, shape ``(n_cells, 3)``, in the order of ``array.ravel()``."""
        return cell_centers(self.grid, self.cells)

    def to_regular(
            self, values: npt.NDArray[np.generic], resolution: Optional[Resolution3] = None
    ) -> npt.NDArray[np.generic]:
        """
        A per-cell array of this frame resampled onto a regular grid.

        See :func:`grid_classes.to_regular`. The default resolution is the grid's
        own for a regular grid and the finest octree level for an octree.
        """
        return to_regular(self.grid, self.cells, values, resolution)

    def section(self, values: npt.NDArray[np.generic], axis: int, index: int) -> npt.NDArray[np.generic]:
        """
        One slice of a per-cell array, as ``np.take(values, index, axis)`` of a regular grid.

        Regular grid: exactly that. Octree: the slice through the centers of the
        finest cells with that index along ``axis`` (the equivalent regular
        grid's slice), each value taken from the cell containing the center.

        Args:
            values: A per-cell array of this frame.
            axis: The axis the slice is normal to: 0 (x), 1 (y) or 2 (z).
            index: Index of the slice along ``axis``, at the finest resolution.

        Returns:
            Array of shape ``(n_a, n_b)`` over the two other axes, in order.
        """
        if not isinstance(self.grid, OctreeGrid):
            return np.take(np.asarray(values), index, axis=axis)
        grid = self.grid
        n = np.array(grid.finest_resolution, dtype=np.int64)
        if not 0 <= index < n[axis]:
            raise IndexError(f"Slice index {index} is outside 0..{n[axis] - 1} along axis {axis}.")
        a, b = [k for k in range(3) if k != axis]
        ia, ib = np.meshgrid(np.arange(n[a]), np.arange(n[b]), indexing="ij")
        ijk = np.empty((ia.size, 3), dtype=np.int64)
        ijk[:, axis] = index
        ijk[:, a] = ia.ravel()
        ijk[:, b] = ib.ravel()
        cells = self.cells.locate(grid, grid.cell_centers(grid.max_level, ijk))
        return np.asarray(values)[cells].reshape(n[a], n[b])

    def _check_arrays(self, arrays: Dict[str, Optional[npt.NDArray[np.generic]]]) -> None:
        """Raise if any of the named arrays doesn't have this frame's per-cell shape."""
        expected = self.cell_array_shape()
        for name, values in arrays.items():
            if values is not None and np.shape(values) != expected:
                raise ValueError(f"{name} has shape {np.shape(values)}, expected {expected}.")


class GeoMeshType(StrEnum):
    MASKED = "masked"
    UNMASKED = "unmasked"
    COMBINED = "combined"
    EXTENDED = "extended"
    EXTENDED_MASKED = "extended_masked"


class FaultMeshType(StrEnum):
    MASKED = "masked"
    UNMASKED = "unmasked"
    EXTENDED = "extended"
    EXTENDED_MASKED = "extended_masked"


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
    scalar_value: Optional[float] = Field(default=None)
    id: Optional[int] = Field(default=None)
    color: Optional[str] = Field(default=None)
    vertices: Dict[str, NpNDArrayFp64] = Field(default_factory=dict)
    edges: Dict[str, NpNDArrayInt32] = Field(default_factory=dict)

    def set_scalar_value(self, value: float) -> None:
        self.scalar_value = value

    def get_scalar_value(self) -> Optional[float]:
        return self.scalar_value

    def set_id(self, element_id: int) -> None:
        self.id = element_id

    def set_color(self, hex_color: str) -> None:
        self.color = hex_color

    def set_mesh(
            self,
            mesh_type: GeoMeshType,
            vertices: npt.NDArray[np.generic],
            edges: npt.NDArray[np.generic],
    ) -> None:
        """
        Set the vertices and edges for a specific mesh type.

        Args:
            mesh_type: A ``GeoMeshType`` member.
            vertices: Vertex array for the mesh.
            edges: Edge/connectivity array for the mesh.

        Raises:
            ValueError: If mesh_type is not a valid ``GeoMeshType``.
        """
        try:
            mesh_type = GeoMeshType(mesh_type)
        except ValueError:
            raise ValueError(
                f"Invalid mesh type '{mesh_type}'. Allowed types are: {', '.join(e.value for e in GeoMeshType)}."
            )
        self.vertices[mesh_type] = vertices
        self.edges[mesh_type] = edges

    def get_mesh(
            self, mesh_type: GeoMeshType
    ) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
        """
        Retrieve the vertices and edges for the given mesh type.

        Raises:
            KeyError: If the mesh type does not exist.
        """
        try:
            return self.vertices[mesh_type], self.edges[mesh_type]
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

    interpolation_method: Optional[InterpolationMethod] = Field(default=None)
    scalar_field: Optional[NpNDArrayFp64] = Field(default=None)
    extended_scalar_field: Optional[NpNDArrayFp64] = Field(default=None)
    interpolation_params: Optional[InterpolationParameterSet] = Field(default=None)
    mask: Optional[NpNDArrayBool] = Field(default=None)
    context: Optional[InterpolationContext] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _params_class_from_method(cls, data):
        """Build serialized interpolation parameters with the class of the group's method.

        ``interpolation_params`` is a plain Union, and e.g. PLIParams/FDIParams or UCKParams/OKParams
        have the same fields, so validation alone would pick the first matching class.
        """
        if not isinstance(data, dict):
            return data
        method, params = data.get("interpolation_method"), data.get("interpolation_params")
        if method is None or not isinstance(params, dict):
            return data
        try:
            params_class = PARAMS_CLASS_BY_METHOD.get(InterpolationMethod(method))
        except ValueError:
            return data
        if params_class is None:
            return data
        return {**data, "interpolation_params": params_class.model_validate(params)}

    def __getitem__(self, element_name: str) -> "StructuralElement":
        for elem in self.structural_elements:
            if elem.name == element_name:
                return elem
        raise KeyError(
            f"Structural element '{element_name}' not found in group '{self.name}'."
        )


    def set_scalar_field(self, field: NpNDArrayFp64) -> None:
        self.scalar_field = field

    def get_scalar_field(self) -> Optional[NpNDArrayFp64]:
        return self.scalar_field

    def set_mask(self, mask: NpNDArrayBool) -> None:
        self.mask = mask

    def get_mask(self) -> Optional[NpNDArrayBool]:
        return self.mask

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

        if self.context is None:
            raise RuntimeError(
                "Interpolation context not initialized. "
                "Call update_interpolation_context() first."
            )

        self.interpolation_method = method
        self.interpolation_params = self._default_params_for_method(method, self.context)

    def set_interpolation_params(self, params: InterpolationParameterSet) -> None:
        self.interpolation_params = params

    def get_interpolation_params(self) -> Optional[InterpolationParameterSet]:
        return self.interpolation_params

    def configure_interpolation_params(self, **kwargs) -> None:
        """
        Update fields on the active interpolation parameter set.

        Raises:
            ValueError: If parameters have not been initialized.
            AttributeError: If any provided key is not a parameter field.
        """
        if self.interpolation_params is None:
            raise ValueError(
                "Interpolation parameters have not been initialized. "
                "Make sure to call set_interpolation_method() first."
            )

        for key, value in kwargs.items():
            if not hasattr(self.interpolation_params, key):
                raise AttributeError(
                    f"'{key}' is not a valid parameter for {type(self.interpolation_params).__name__}."
                )
            setattr(self.interpolation_params, key, value)

    def _default_params_for_method(
            self, method: InterpolationMethod, ctx: InterpolationContext
    ) -> Optional[InterpolationParameterSet]:
        """Internal helper to map a method to its default parameter set."""
        if method == InterpolationMethod.RADIAL_BASIS_FUNCTION:
            return default_rbf_params(ctx)

        if method == InterpolationMethod.ORDINARY_KRIGING:
            return default_ok_params(ctx)

        if method == InterpolationMethod.UNIVERSAL_COKRIGING:
            return default_uck_params(ctx)

        if method == InterpolationMethod.GEOINR:
            return default_geo_inr_params(ctx)

        if method == InterpolationMethod.FINITE_DIFFERENCES:
            return default_fdi_params(ctx)

        if method == InterpolationMethod.PIECEWISE_LINEAR:
            return default_pli_params(ctx)

        if method == InterpolationMethod.UNIVERSAL_KRIGING:
            return default_uk_params(ctx)

        return None

    def set_interpolation_context(self, ctx: InterpolationContext) -> None:
        self.context = ctx

    def update_interpolation_context(self, points: NpNDArrayFp64) -> None:
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

        self.context = InterpolationContext(
            data_scale=data_scale,
            n_points=int(points.shape[0]),
            mean_nn_distance=mean_nn_distance,
        )


class StructuralFrame(_PerCellResults, BaseModel):
    """
    A structural frame containing structural groups and optional fault information.

    Attributes:
        structural_groups: Ordered list of :class:`StructuralGroup` objects.

    Private state (set/derived elsewhere):
        _grid: Grid defining evaluation coordinates.
        _cells: Realized octree cells when ``grid`` is an OctreeGrid, else None.
            Per-cell arrays (``lith_block``, group scalar fields and masks) are
            aligned with them; see :class:`_PerCellResults`.
        _surface_points: Surface points for all elements.
        _orientations: Optional orientation data for all elements.
        _lith_block: Optional lithology per cell.
        _fault_frame: Optional fault frame sharing the same grid.
        _fault_activity: Optional mapping of fault name -> the youngest affected group index.
    """

    structural_groups: List[StructuralGroup] = Field(default_factory=list)

    grid: Optional[ModelGrid] = Field(default=None)
    cells: Optional[OctreeCells] = Field(default=None)
    surface_points: Optional[PandasDataFrame] = Field(default=None)
    orientations: Optional[PandasDataFrame] = Field(default=None)
    lith_block: Optional[NpNDArrayInt32] = Field(default=None)
    fault_frame: Optional["FaultFrame"] = Field(default=None)
    fault_activity: Optional[dict[str, int]] = Field(default=None)

    def set_fault_frame(self, fault_frame: Optional["FaultFrame"]) -> None:
        """
        Attach/detach a FaultFrame with consistency checks.

        Notes:
            The checks in this method enforce that both frames share the *same* grid
            instance and that the fault frame has a computed domain map.
        """
        if fault_frame is None:
            self.fault_frame = None
            self.fault_activity = None
            return

        if self.grid is None:
            raise ValueError(
                "StructuralFrame grid must be set before assigning a FaultFrame."
            )

        if fault_frame.grid is None:
            raise ValueError(
                "FaultFrame grid must be set before being assigned to a StructuralFrame."
            )

        if fault_frame.grid != self.grid:
            raise ValueError("FaultFrame and StructuralFrame must share the same grid.",
                             fault_frame.grid,
                             self.grid,
                             self.grid != fault_frame.grid)

        # Check that the fault frame has a computed solution
        if fault_frame.domain_map is None:
            raise ValueError(
                "Cannot assign a FaultFrame without a computed solution (domain map missing). "
                "Please run `compute_fault_domains(fault_frame)` first."
            )

        self.fault_frame = fault_frame

        # Initialize fault activity to default (all groups affected)
        self.fault_activity = {fault.name: 0 for fault in fault_frame.fault_elements}

    # @property
    # def fault_activity(self) -> Optional[dict[str, int]]:
    #     if self._fault_activity is None:
    #         return None
    #     return dict(self._fault_activity)  # defensive copy # todo

    @property
    def fault_activity_verbose(self) -> Optional[dict[str, dict[str, int | str]]]:
        if self.fault_activity is None:
            return None

        return {
            fault: {
                "youngest_group_index": idx,
                "youngest_group_name": self.structural_groups[idx].name,
            }
            for fault, idx in self.fault_activity.items()
        }

    def set_fault_activity_by_index(self, fault_name: str, youngest_group_idx: int) -> None:
        if self.fault_frame is None or self.fault_activity is None:
            raise RuntimeError("Cannot set fault activity without a FaultFrame attached.")

        if fault_name not in self.fault_activity:
            raise KeyError(f"Fault '{fault_name}' not found in fault frame.")

        if youngest_group_idx < 0 or youngest_group_idx >= len(self.structural_groups):
            raise ValueError(
                f"max_group_idx must be in range [0, {len(self.structural_groups) - 1}]."
            )

        self.fault_activity[fault_name] = youngest_group_idx

    def set_fault_activity_by_group(self, fault_name: str, group_name: str) -> None:
        for idx, group in enumerate(self.structural_groups):
            if group.name == group_name:
                self.set_fault_activity_by_index(fault_name, idx)
                return

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

    def get_LithBlock(self) -> Optional[npt.NDArray[np.generic]]:
        return self.lith_block

    def cell_domain_ids(self) -> npt.NDArray[np.int_]:
        """
        Fault domain id of every cell of this frame, shape ``cell_array_shape()``.

        Regular grid: the fault frame's ``domain_map``. Octree: each cell takes
        the domain of the fault cell that contains its center. A structural cell
        can span two domains only where the fault between them doesn't affect
        the group present there; the exact domains are the fault frame's
        ``domain_map`` on its own cells. Without a fault frame
        every cell is in domain 0.
        """
        if self.fault_frame is None:
            return np.zeros(self.cell_array_shape(), dtype=np.int32)
        ff = self.fault_frame
        if ff.domain_map is None:
            raise ValueError("The attached fault frame has no domain_map.")
        if not isinstance(self.grid, OctreeGrid):
            return np.asarray(ff.domain_map)
        index = ff.cells.locate(ff.grid, self.cell_centers())
        if np.any(index < 0):
            raise ValueError("The fault frame's cells do not cover this frame's cells.")
        return np.asarray(ff.domain_map)[index]

    def check_cell_arrays(self) -> None:
        """
        Raise if a per-cell array of this frame or its fault frame doesn't match the grid.

        Checks ``lith_block`` and every group's ``scalar_field`` and ``mask``, then
        the attached fault frame. Extended (padded) fields are not per-cell arrays
        of this grid and are not checked.
        """
        arrays: Dict[str, Optional[npt.NDArray[np.generic]]] = {"lith_block": self.lith_block}
        for group in self.structural_groups:
            arrays[f"{group.name}.scalar_field"] = group.scalar_field
            arrays[f"{group.name}.mask"] = group.mask
        self._check_arrays(arrays)
        if self.fault_frame is not None:
            self.fault_frame.check_cell_arrays()

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
        for line in grid_report_lines(self.grid, self.cells):
            print(line)
        print(f"• Total surface points: {len(self.surface_points)} entries")
        if self.orientations is not None:
            print(f"• Total orientations: {len(self.orientations)} entries\n")
        else:
            print("• Orientations: None\n")

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
        if self.fault_frame is not None:
            print("⚡ Fault Frame Present")
            num_faults = len(self.fault_frame.fault_elements)
            print(f"• Number of faults: {num_faults}")

            if self.fault_activity_verbose is not None:
                print("• Fault activity (max group affected):")
                for fault_name, group_name in self.fault_activity_verbose.items():
                    idx = self.fault_activity[fault_name]
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
        xmin, xmax, ymin, ymax, zmin, zmax = self.grid.extent
        if axis == "y":
            data_slice = self.section(group.scalar_field, 1, index).T
            extent = (xmin, xmax, zmin, zmax)
            xlabel, ylabel = "X", "Z"
        elif axis == "x":
            data_slice = self.section(group.scalar_field, 0, index).T
            extent = (ymin, ymax, zmin, zmax)
            xlabel, ylabel = "Y", "Z"
        elif axis == "z":
            data_slice = self.section(group.scalar_field, 2, index).T
            extent = (xmin, xmax, ymin, ymax)
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
        Index along the finest resolution on an octree (see ``section``).
        """
        group = self.structural_groups[group_nr]
        mask = group.get_mask()

        if mask is None:
            raise ValueError(f"Structural group at index {group_nr} has no age mask computed.")

        xmin, xmax, ymin, ymax, zmin, zmax = self.grid.extent

        if axis == "y":
            # X-Z at fixed Y=index
            mask_slice = self.section(mask, 1, index).T  # (Z, X)
            extent = (xmin, xmax, zmin, zmax)
            xlabel, ylabel = "X", "Z"
        elif axis == "x":
            # Y-Z at fixed X=index
            mask_slice = self.section(mask, 0, index).T  # (Z, Y)
            extent = (ymin, ymax, zmin, zmax)
            xlabel, ylabel = "Y", "Z"
        elif axis == "z":
            # X-Y at fixed Z=index
            mask_slice = self.section(mask, 2, index).T  # (Y, X)
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
            aspect="equal",
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

    name: str = Field()
    scalar_value: Optional[float] = Field(default=None)
    scalar_field: Optional[NpNDArrayFp64] = Field(default=None)
    extended_scalar_field: Optional[NpNDArrayFp64] = Field(default=None)
    color: str = Field(default="#AAAAAA")  # Default color in hex format
    separated_domains: Optional[Tuple[FrozenSet[int], FrozenSet[int]]] = Field(
        default=None
    )
    domain_pairs: Optional[FrozenSet[Tuple[int, int]]] = Field(default=None)
    vertices: Dict[str, NpNDArrayFp64] = Field(default_factory=dict)
    edges: Dict[str, NpNDArrayInt32] = Field(default_factory=dict)
    mask: Optional[NpNDArrayBool] = Field(default=None)

    def __init__(self, *args, **kwargs):
        # shortcut definition - not sure if I like this
        if args:
          if len(args) == 1 or len(args) == 2:
            kwargs["name"] = args[0]
          else:
            raise TypeError("Unsupported shortcut, name:str, scalar_value: Optional[float] = None ")
          if len(args) == 2:
            kwargs["scalar_value"] = args[1]
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"FaultElement(name='{self.name}')"


    def set_scalar_field(self, scalar_field: NpNDArrayFp64) -> None:
        """Assign the interpolated scalar field values on the fault surface."""
        if not isinstance(scalar_field, np.ndarray):
            raise ValueError("Scalar field must be a numpy array.")
        self.scalar_field = scalar_field

    def set_scalar_value(self, value: float) -> None:
        """Assign scalar value used for interpolation."""
        self.scalar_value = value

    def set_color(self, color: str) -> None:
        """Set the display color for this fault in hex format."""
        if not isinstance(color, str) or not color.startswith("#") or len(color) != 7:
            raise ValueError("Color must be a valid hex string (e.g., '#RRGGBB').")
        self.color = color

    def set_mesh(
            self,
            mesh_type: FaultMeshType,
            vertices: npt.NDArray[np.generic],
            edges: npt.NDArray[np.generic],
    ) -> None:
        """
        Set the vertices and edges for a specific mesh type.

        Args:
            mesh_type: A ``FaultMeshType`` member.
            vertices: Vertex array.
            edges: Edge/connectivity array.

        Raises:
            ValueError: If mesh_type is not a valid ``FaultMeshType``.
        """
        try:
            mesh_type = FaultMeshType(mesh_type)
        except ValueError:
            raise ValueError(
                f"Invalid mesh type '{mesh_type}'. Allowed types are: {', '.join(e.value for e in FaultMeshType)}."
            )
        self.vertices[mesh_type] = vertices
        self.edges[mesh_type] = edges

    def get_mesh(
            self, mesh_type: FaultMeshType
    ) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
        """
        Retrieve the vertices and edges for the given mesh type.

        Raises:
            KeyError: If the mesh type does not exist.
        """
        try:
            return self.vertices[mesh_type], self.edges[mesh_type]
        except KeyError:
            raise KeyError(f"Mesh '{mesh_type}' not found in element '{self.name}'.")

    def set_domain_mask(self, mask: NpNDArrayBool) -> None:
        """
        Store the boolean mask (True/False) separating two fault blocks.
        """
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise ValueError("Mask must be a boolean NumPy array.")
        self.mask = mask

    def get_domain_mask(self) -> NpNDArrayBool:
        """
        Retrieve the fault mask (True = one block, False = other).

        Raises:
            ValueError: If no mask has been set.
        """
        if self.mask is None:
            raise ValueError(f"No mask set for fault '{self.name}'.")
        return self.mask

    def get_inverse_domain_mask(self) -> NpNDArrayBool:
        """
        Get the inverse of the fault mask (opposite block).
        """
        return ~self.get_domain_mask()

    def get_separated_domains(self) -> Optional[Tuple[FrozenSet[int], FrozenSet[int]]]:
        """Get the tuple of separated domain IDs, if any."""
        return self.separated_domains

    def set_separated_domains(
            self, domain_ids: Tuple[FrozenSet[int], FrozenSet[int]]
    ) -> None:
        """Set the tuple of separated domain IDs."""
        self.separated_domains = domain_ids

    def get_domain_pairs(self) -> Optional[FrozenSet[Tuple[int, int]]]:
        """Return adjacent domain-id pairs across this fault surface."""
        return self.domain_pairs

    def set_domain_pairs(self, pairs: FrozenSet[Tuple[int, int]]) -> None:
        """
        Set adjacent domain-id pairs across this fault surface.

        Each pair must be (a, b) with a != b. Order will be normalized to (min, max).
        """
        if pairs is None:
            self.domain_pairs = None
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

        self.domain_pairs = frozenset(norm)

    def domain_pairs_flat(self) -> set[int]:
        """Return all domain IDs touched by this fault's adjacency pairs."""
        if self.domain_pairs is None:
            return set()
        out: set[int] = set()
        for a, b in self.domain_pairs:
            out.add(a)
            out.add(b)
        return out

    def separated_domains_flat(self) -> set[int]:
        """Return all domain IDs this fault splits (from separated_domains)."""
        if self.separated_domains is None:
            return set()
        return set().union(*self.separated_domains)


class FaultFrame(_PerCellResults, BaseModel):
    """
    Container for fault elements and their computed relationships.

    Stores fault surface data, the domain map produced by interpolation, and per-fault
    adjacency information (separated domains, domain pairs). Large numpy arrays (scalar
    fields, masks, domain map) are stored as Pydantic fields with pydantic-numpy types.

    When ``grid`` is an OctreeGrid, ``cells`` holds the realized octree cells and the
    per-cell arrays (``domain_map``, ``domain_masks``, each fault's ``scalar_field``
    and ``mask``) are aligned with them; see :class:`_PerCellResults`.
    """

    fault_elements: List[FaultElement]
    fault_relations: Optional[NpNDArrayBool] = Field(default=None)
    grid: Optional[ModelGrid] = Field(default=None)
    cells: Optional[OctreeCells] = Field(default=None)
    fault_surface_points_df: Optional[PandasDataFrame] = Field(default=None)
    fault_orientations_df: Optional[PandasDataFrame] = Field(default=None)
    domain_map: Optional[NpNDArrayInt32] = Field(default=None)
    domain_masks: dict[int, NpNDArrayBool] = Field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        # Positional shortcut: FaultFrame([elements]) or FaultFrame([elements], relations)
        if args:
            if len(args) == 1 or len(args) == 2:
                kwargs["fault_elements"] = args[0]
            else:
                raise TypeError("FaultFrame positional args: fault_elements, fault_relations (optional)")
            if len(args) == 2:
                kwargs["fault_relations"] = args[1]
        super().__init__(**kwargs)


    def get_element_by_name(self, name: str) -> Optional[FaultElement]:
        """Retrieve a fault element by its name."""
        return next((f for f in self.fault_elements if f.name == name), None)

    def set_surface_points_df(self, df: pd.DataFrame) -> None:
        self.fault_surface_points_df = df

    def set_orientations_df(self, df: pd.DataFrame) -> None:
        self.fault_orientations_df = df

    def get_surface_points_df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame containing all fault surface points."""
        return self.fault_surface_points_df

    def get_orientations_df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame containing all fault surface orientations."""
        return self.fault_orientations_df

    def get_surface_points_for_element(self, name: str) -> pd.DataFrame:
        if self.fault_surface_points_df is not None:
            return self.fault_surface_points_df[
              self.fault_surface_points_df["formation"] == name
              ]
        return pd.DataFrame()

    def get_orientations_for_element(self, name: str) -> pd.DataFrame:
        if self.fault_orientations_df is not None:
            return self.fault_orientations_df[
              self.fault_orientations_df["formation"] == name
              ]
        return pd.DataFrame()

    def set_domain_map(self, domain_map: NpNDArrayInt32) -> None:
        self.domain_map = domain_map

    def set_grid(self, grid: ModelGrid) -> None:
        """Set the grid for spatial context."""
        self.grid = grid

    def check_cell_arrays(self) -> None:
        """
        Raise if a per-cell array of this frame doesn't match the grid.

        Checks ``domain_map``, every entry of ``domain_masks`` and each fault's
        ``scalar_field`` and ``mask``. Extended (padded) fields are not checked.
        """
        arrays: Dict[str, Optional[npt.NDArray[np.generic]]] = {"domain_map": self.domain_map}
        for domain_id, mask in self.domain_masks.items():
            arrays[f"domain_masks[{domain_id}]"] = mask
        for fault in self.fault_elements:
            arrays[f"{fault.name}.scalar_field"] = fault.scalar_field
            arrays[f"{fault.name}.mask"] = fault.mask
        self._check_arrays(arrays)

    def detailed_report(self) -> None:
        """Print a human-readable report of the fault frame state."""
        print("🧱 Fault Frame — Detailed Report")
        print("────────────────────────────────")
        print(f"• Number of faults: {len(self.fault_elements)}")

        for line in grid_report_lines(self.grid, self.cells):
            print(line)

        if self.fault_surface_points_df is not None:
            print(f"• Surface points: {len(self.fault_surface_points_df)} entries")
        else:
            print("• Surface points: None")

        if self.fault_orientations_df is not None:
            print(f"• Orientations: {len(self.fault_orientations_df)} entries\n")
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

