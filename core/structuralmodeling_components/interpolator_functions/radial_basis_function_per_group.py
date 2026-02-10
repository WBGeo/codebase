"""
Radial Basis Function (RBF) interpolation for a single structural group.

This module provides a pure interpolator function that:
- derives strictly increasing scalar values per element (oldest=1 .. youngest=n),
- fits an RBF interpolator to surface points,
- evaluates the scalar field on a regular grid.

No mutation of the passed `group` is performed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
from scipy.interpolate import RBFInterpolator

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
    if len(group.structural_elements) < 2:
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

    # 2) map formations -> scalar values
    vals: FloatArray = (
        group_surface_points_df["formation"]
        .map(scalar_values_by_element)
        .astype(float)
        .to_numpy()
    )

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
        epsilon=params.epsilon,
        neighbors=params.neighbors,  # None => all points
    )

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
