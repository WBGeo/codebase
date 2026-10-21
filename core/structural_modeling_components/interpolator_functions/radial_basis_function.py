"""
Radial Basis Function (RBF) interpolation for a single structural group.

This module provides a pure interpolator function that:
- derives strictly increasing scalar values per element (oldest=1 .. youngest=n),
- fits an RBF interpolator to surface points,
- evaluates the scalar field on a regular grid,
and a fit-only variant (``fit_group_radial_basis_function``) that returns the
fitted field for evaluation at arbitrary points (octree grids).

No mutation of the passed `group` is performed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
from scipy.interpolate import RBFInterpolator

from core.structural_modeling_components.interpolator_functions.fitted_field import Extent6
from core.structural_modeling_components.interpolator_functions.interpolator_parameters import (
    RBF_KERNELS_USING_EPSILON,
)

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]


def interpolate_group_radial_basis_function(
    *,
    group: Any,  # expected: StructuralGroup-like (group.name, group.structural_elements, group.get_interpolation_params())
    grid: Any,  # expected: RegularGrid-like (grid.resolution and either grid.grid_coordinates or grid.gridx/y/z)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,  # unused here
) -> ScalarFieldAndValues:
    """
    Radial Basis Function interpolation for a single structural group (pure).

    Parameters
    ----------
    group
        StructuralGroup-like object. Expected members:
        - `name: str`
        - `structural_elements: Sequence[... with .name]`
        - `get_interpolation_params() -> RBFParams-like` (kernel/smoothing/epsilon/neighbors)
    grid
        RegularGrid-like object providing:
        - `resolution: tuple[int, int, int]`
        - optionally `grid_coordinates: (N, 3) ndarray`
        - otherwise `grid.gridx`, `grid.gridy`, `grid.gridz` 1D coordinate arrays
    group_surface_points_df
        DataFrame of constraint points with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        Optional orientations DataFrame. Accepted for API consistency but not used.

    Returns
    -------
    scalar_field : np.ndarray
        Interpolated scalar field reshaped to `grid.resolution` and then transposed
        exactly as in the original implementation.
    scalar_values_by_element : dict[str, float]
        Mapping element_name -> scalar value. Values are strictly increasing from
        oldest=1 to youngest=n (group.structural_elements is assumed youngest->oldest,
        so reversed order is used here).

    Notes
    -----
    - Assigns strictly increasing scalar values: oldest = 1, youngest = n
    - Uses params from `group.get_interpolation_params()` (expected to be your RBFParams)
    - Does NOT mutate `group`
    """
    rbfi, scalar_values_by_element = _fit_rbf(group, group_surface_points_df, group_orientations_points_df)

    # 6) evaluate on grid
    # Prefer a precomputed (N,3) array if your grid exposes it; otherwise build from axes.
    if hasattr(grid, "grid_coordinates") and grid.grid_coordinates is not None:
        grid_points: FloatArray = grid.grid_coordinates
    else:
        gx, gy, gz = np.meshgrid(grid.gridx, grid.gridy, grid.gridz, indexing="ij")
        grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    scalar_flat: FloatArray = rbfi(grid_points)
    scalar_field = scalar_flat.reshape(tuple(grid.resolution)).T  # (nx, ny, nz)

    return scalar_field, scalar_values_by_element


class RBFFittedField:
    """
    A fitted RBF scalar field of one group, evaluable at any points (see ``FittedField``).

    Attributes:
        scalar_values: Element name -> isovalue, oldest = 1 .. youngest = n.
    """

    def __init__(self, interpolator: RBFInterpolator, scalar_values: Dict[str, float]) -> None:
        self._interpolator = interpolator
        self.scalar_values = scalar_values

    def evaluate(self, xyz: FloatArray) -> FloatArray:
        """Field values at points of shape ``(M, 3)``; returns shape ``(M,)``."""
        return self._interpolator(np.asarray(xyz, dtype=np.float64).reshape(-1, 3))


def fit_group_radial_basis_function(
    *,
    group: Any,  # expected: StructuralGroup-like (group.name, group.structural_elements, group.get_interpolation_params())
    extent: Optional[Extent6] = None,  # unused here: an RBF has no support mesh
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,  # unused here
) -> RBFFittedField:
    """
    Fit the RBF scalar field of a single structural group without evaluating it.

    Same inputs, checks, warnings and scalar values as
    :func:`interpolate_group_radial_basis_function`; evaluating the result at a
    regular grid's ``grid_coordinates`` gives that function's scalar field.

    Parameters
    ----------
    group
        StructuralGroup-like object, as for :func:`interpolate_group_radial_basis_function`.
    extent
        Accepted for the common ``fit_group_*`` interface; the RBF doesn't depend on it.
    group_surface_points_df
        DataFrame of constraint points with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        Optional orientations DataFrame. Accepted for API consistency but not used.

    Returns
    -------
    RBFFittedField
        The fitted field and the element scalar values.
    """
    rbfi, scalar_values_by_element = _fit_rbf(group, group_surface_points_df, group_orientations_points_df)
    return RBFFittedField(rbfi, scalar_values_by_element)


def _fit_rbf(
    group: Any,
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame],
) -> Tuple[RBFInterpolator, Dict[str, float]]:
    """Check the inputs, derive the element scalar values and fit the RBF (shared by both entry points)."""
    # Check if any surface points are provided
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")

    # Check if orientations are provided and warn that they will not be used
    if group_orientations_points_df is not None and not group_orientations_points_df.empty:
        warnings.warn(
            f"Orientations provided for group '{group.name}' will not be used in RBF interpolation",
            UserWarning,
        )

    # Check if there are at least two distinct elements in the group
    _phantom_name: Optional[str] = None
    if len(group.structural_elements) < 2:
        if len(group.structural_elements) == 1:
            _coords_tmp = group_surface_points_df[["X", "Y", "Z"]].to_numpy()
            _bbox_diag = float(np.linalg.norm(_coords_tmp.max(axis=0) - _coords_tmp.min(axis=0)))
            _shift = max(0.1 * _bbox_diag, 1.0)
            _phantom_name = "__phantom__"
            _phantom_df = group_surface_points_df.copy()
            _phantom_df = _phantom_df.assign(Z=_phantom_df["Z"] - _shift, formation=_phantom_name)
            group_surface_points_df = pd.concat(
                [group_surface_points_df, _phantom_df], ignore_index=True
            )
            warnings.warn(
                f"Group '{group.name}' has only one structural element. RBF interpolation "
                f"requires at least two. A phantom 'older' element has been synthesized by "
                f"shifting all points downward by {_shift:.4g} units (10%% of the point "
                f"cloud bounding-box diagonal). This approximation assumes sub-horizontal "
                f"layering and WILL produce incorrect results for steeply dipping or "
                f"overturned structures. Consider using FDI, PLI, UCK, or GeoINR instead.",
                UserWarning,
                stacklevel=3,
            )
        else:
            raise ValueError(
                f"Group '{group.name}' must contain at least two structural elements for RBF interpolation"
            )

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")

    # 1) scalar values oldest->youngest = 1..n (groups list is youngest->oldest, so reverse)
    scalar_values_by_element: Dict[str, float] = {
        elem.name: float(i)
        for i, elem in enumerate(reversed(group.structural_elements), start=1)
    }
    if _phantom_name is not None:
        scalar_values_by_element[_phantom_name] = 0.0

    # 2) map formations -> scalar values
    vals: FloatArray = (
        group_surface_points_df["formation"]
        .map(scalar_values_by_element)
        .astype(float)
        .to_numpy()
    )
    if _phantom_name is not None:
        scalar_values_by_element.pop(_phantom_name)

    # 3) coordinates of control points
    coords: FloatArray = group_surface_points_df[["X", "Y", "Z"]].to_numpy()

    # 4) parameters (expected: your RBFParams-like object)
    params = group.get_interpolation_params()

    # 5) fit RBF
    rbfi = RBFInterpolator(
        coords,
        vals,
        kernel=params.kernel,
        smoothing=params.smoothing,
        # scale-invariant kernels are evaluated without the shape parameter
        epsilon=params.epsilon if params.kernel in RBF_KERNELS_USING_EPSILON else None,
        neighbors=params.neighbors,  # None => all points
    )
    return rbfi, scalar_values_by_element
