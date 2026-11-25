"""
Ordinary Kriging (3D) interpolation for a single structural group.

This module provides a pure interpolator function that:
- assigns strictly increasing scalar values per element (oldest=1 .. youngest=n),
- fits a 3D Ordinary Kriging model to surface points,
- evaluates the scalar field on a regular grid.

The function does NOT mutate the passed `group`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeAlias
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D

from core.structural_modeling_components.interpolator_functions.minimum_data import (
    PHANTOM_ELEMENT_NAME,
    add_phantom_element,
)

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]

OK_MIN_POINTS = 2


def effective_neighbors(neighbors: Optional[int], n_points: int) -> Optional[int]:
    """Moving-window size for PyKrige: None (all points) if `neighbors` would cover all points anyway."""
    if neighbors is None or neighbors >= n_points:
        return None
    return int(neighbors)


def ok_input_problem(points: FloatArray) -> Optional[str]:
    """
    Check whether Ordinary Kriging can be fitted to `points` ((N, 3), including phantom points).

    Returns a description of the problem for the user, or None if the points are sufficient.
    """
    if len(points) < OK_MIN_POINTS:
        return (
            f"needs at least {OK_MIN_POINTS} surface points, but only {len(points)} "
            f"{'is' if len(points) == 1 else 'are'} available. Add surface points."
        )
    return None


def interpolate_group_ordinary_kriging(
    *,
    group: Any,  # expected: StructuralGroup-like (group.name, group.structural_elements, group.get_interpolation_params())
    grid: Any,  # expected: RegularGrid-like with grid.gridx/grid.gridy/grid.gridz 1D arrays
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,  # unused here
) -> ScalarFieldAndValues:
    """
    Ordinary Kriging for a single structural group (pure function).

    Parameters
    ----------
    group
        StructuralGroup-like object. Expected members:
        - `name: str`
        - `structural_elements: Sequence[... with .name]`
        - `get_interpolation_params() -> OrdinaryKrigingParams-like`
          (variogram_model, sill, range, nugget, anisotropy_scaling_y/z, neighbors)
    grid
        RegularGrid-like object providing:
        - `gridx`, `gridy`, `gridz`: 1D coordinate arrays (cell centers)
        - `resolution`: tuple[int, int, int] (not directly used by this interpolator)
    group_surface_points_df
        DataFrame of constraint points with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        Optional orientations DataFrame. Accepted for API consistency but not used.

    Returns
    -------
    scalar_field : np.ndarray
        Interpolated scalar field evaluated on the grid.
        Shape matches pykrige's output for `execute("grid", ...)`.
    scalar_values_by_element : dict[str, float]
        Mapping element_name -> scalar value. Values are strictly increasing from
        oldest=1 to youngest=n (group.structural_elements is assumed youngest->oldest,
        so reversed order is used here).

    Notes
    -----
    - Assigns strictly increasing scalar values: oldest = 1, youngest = n.
    - Uses params from `group.get_interpolation_params()` (expected: OrdinaryKrigingParams).
    - Does NOT mutate `group`.
    """
    # Check if any surface points are provided
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")

    # Check if orientations are provided and warn that they will not be used
    if (
        group_orientations_points_df is not None
        and not group_orientations_points_df.empty
    ):
        warnings.warn(
            f"Orientations provided for group '{group.name}' will not be used in Ordinary Kriging",
            UserWarning,
        )

    # Check if there are at least two distinct elements in the group
    _phantom_name: Optional[str] = None
    if len(group.structural_elements) < 2:
        if len(group.structural_elements) == 1:
            _phantom_name = PHANTOM_ELEMENT_NAME
            group_surface_points_df, _shift = add_phantom_element(group_surface_points_df)
            warnings.warn(
                f"Group '{group.name}' has only one structural element. Ordinary Kriging "
                f"requires at least two. A phantom 'older' element has been synthesized by "
                f"shifting all points downward by {_shift:.4g} units (10%% of the point "
                f"cloud bounding-box diagonal). This approximation assumes sub-horizontal "
                f"layering and WILL produce incorrect results for steeply dipping or "
                f"overturned structures. Consider using FDI, PLI, UCK, or GeoINR instead.",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"Group '{group.name}' must contain at least two structural elements for Ordinary Kriging"
            )

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")

    # 1) scalar values oldest->youngest = 1..n (your groups are youngest->oldest, so reverse)
    scalar_values_by_element: Dict[str, float] = {
        elem.name: float(i)
        for i, elem in enumerate(reversed(group.structural_elements), start=1)
    }
    if _phantom_name is not None:
        scalar_values_by_element[_phantom_name] = 0.0

    # Guard: all formations in DF must belong to this group
    unknown = set(group_surface_points_df["formation"].unique()) - set(
        scalar_values_by_element.keys()
    )
    if unknown:
        raise ValueError(
            f"Surface points for group '{group.name}' contain formations not in the group: {sorted(unknown)}"
        )

    # 2) map formations -> scalar values
    vals: FloatArray = (
        group_surface_points_df["formation"]
        .map(scalar_values_by_element)
        .astype(float)
        .to_numpy()
    )
    if _phantom_name is not None:
        scalar_values_by_element.pop(_phantom_name)

    # 3) coordinates
    x: FloatArray = group_surface_points_df["X"].to_numpy()
    y: FloatArray = group_surface_points_df["Y"].to_numpy()
    z: FloatArray = group_surface_points_df["Z"].to_numpy()

    # 4) parameters (expected: your OrdinaryKrigingParams-like object)
    params = group.get_interpolation_params()

    # 5) fit & evaluate on grid
    ok3d = OrdinaryKriging3D(
        x,
        y,
        z,
        vals,
        variogram_model=params.variogram_model,
        variogram_parameters=[params.sill, params.range, params.nugget],
        anisotropy_scaling_y=params.anisotropy_scaling_y,
        anisotropy_scaling_z=params.anisotropy_scaling_z,
    )

    # Never ask for more neighbours than there are points in this fit (e.g. a fault
    # domain with few points): PyKrige then fills the neighbourhood with a dummy index,
    # and the kriging matrix becomes singular. Use all points instead.
    n_closest_points = effective_neighbors(params.neighbors, len(x))

    # PyKrige uses a slower looping backend when requesting a moving neighborhood.
    backend = "loop" if n_closest_points is not None else "vectorized"

    scalar_field, _ = ok3d.execute(
        "grid",
        grid.gridx,
        grid.gridy,
        grid.gridz,
        backend=backend,
        n_closest_points=n_closest_points,
    )

    return scalar_field, scalar_values_by_element
