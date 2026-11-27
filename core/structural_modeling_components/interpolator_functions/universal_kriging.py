"""
Universal Kriging (3D) interpolation for a single structural group.

This mirrors the API and behavior of `interpolate_group_ordinary_kriging` but uses
PyKrige's 3D Universal Kriging implementation.

Key behavior (same as the OK version):
- assigns strictly increasing scalar values per element (oldest=1 .. youngest=n),
- fits a 3D kriging model to surface points,
- evaluates the scalar field on a regular grid,
- does NOT mutate the passed `group`.

Universal Kriging additionally fits a linear trend (PyKrige's "regional_linear"
drift, a + b·x + c·y + d·z) and always uses all points.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeAlias
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pykrige.uk3d import UniversalKriging3D

from core.structural_modeling_components.interpolator_functions.interpolator_parameters import (
    kriging_variogram_parameters,
)
from core.structural_modeling_components.interpolator_functions.minimum_data import (
    PHANTOM_ELEMENT_NAME,
    POLYNOMIAL_TERMS_3D,
    add_phantom_element,
    points_support_polynomial,
)

FloatArray: TypeAlias = npt.NDArray[np.floating]
ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]

# The regional linear drift is a plane a + b·x + c·y + d·z fitted through the points.
_UK_DRIFT_DEGREE = 1


def uk_input_problem(points: FloatArray) -> Optional[str]:
    """
    Check whether Universal Kriging (with its linear drift) can be fitted to `points`
    ((N, 3), including phantom points). Otherwise PyKrige fails with "singular matrix".

    Returns a description of the problem for the user, or None if the points are sufficient.
    """
    n_min = POLYNOMIAL_TERMS_3D[_UK_DRIFT_DEGREE]
    if len(points) < n_min:
        return (
            f"fits a linear trend and needs at least {n_min} surface points, but only "
            f"{len(points)} are available. Add surface points, or use Ordinary Kriging, "
            f"which works with fewer points."
        )
    if not points_support_polynomial(points, _UK_DRIFT_DEGREE):
        return (
            f"all {len(points)} surface points lie in one plane, so the linear trend cannot be "
            f"fitted. Add surface points outside this plane, or use Ordinary Kriging."
        )
    return None


def interpolate_group_universal_kriging(
    *,
    group: Any,
    grid: Any,
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,  # unused here
) -> ScalarFieldAndValues:
    """
    Universal Kriging for a single structural group (pure function).

    Parameters
    ----------
    group
        StructuralGroup-like object. Expected members:
        - `name: str`
        - `structural_elements: Sequence[... with .name]`
        - `get_interpolation_params() -> UKParams-like`
          Required attributes: variogram_model, range, nugget,
          anisotropy_scaling_y, anisotropy_scaling_z.
    grid
        RegularGrid-like object providing:
        - `gridx`, `gridy`, `gridz`: 1D coordinate arrays (cell centers)
    group_surface_points_df
        DataFrame of constraint points with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        Optional orientations DataFrame. Accepted for API consistency but not used.

    Returns
    -------
    scalar_field : np.ndarray
        Interpolated scalar field evaluated on the grid.
    scalar_values_by_element : dict[str, float]
        Mapping element_name -> scalar value. Values are strictly increasing from
        oldest=1 to youngest=n (group.structural_elements is assumed youngest->oldest,
        so reversed order is used here).
    """
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")

    if group_orientations_points_df is not None and not group_orientations_points_df.empty:
        warnings.warn(
            f"Orientations provided for group '{group.name}' will not be used in Universal Kriging",
            UserWarning,
        )

    _phantom_name: Optional[str] = None
    if len(group.structural_elements) < 2:
        if len(group.structural_elements) == 1:
            _phantom_name = PHANTOM_ELEMENT_NAME
            group_surface_points_df, _shift = add_phantom_element(group_surface_points_df)
            warnings.warn(
                f"Group '{group.name}' has only one structural element. Universal Kriging "
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
                f"Group '{group.name}' must contain at least two structural elements for Universal Kriging"
            )

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")

    # scalar values oldest->youngest = 1..n (groups are youngest->oldest, so reverse)
    scalar_values_by_element: Dict[str, float] = {
        elem.name: float(i)
        for i, elem in enumerate(reversed(group.structural_elements), start=1)
    }
    if _phantom_name is not None:
        scalar_values_by_element[_phantom_name] = 0.0

    unknown = set(group_surface_points_df["formation"].unique()) - set(scalar_values_by_element.keys())
    if unknown:
        raise ValueError(
            f"Surface points for group '{group.name}' contain formations not in the group: {sorted(unknown)}"
        )

    vals: FloatArray = (
        group_surface_points_df["formation"]
        .map(scalar_values_by_element)
        .astype(float)
        .to_numpy()
    )
    if _phantom_name is not None:
        scalar_values_by_element.pop(_phantom_name)

    x: FloatArray = group_surface_points_df["X"].to_numpy()
    y: FloatArray = group_surface_points_df["Y"].to_numpy()
    z: FloatArray = group_surface_points_df["Z"].to_numpy()

    params = group.get_interpolation_params()

    uk3d = UniversalKriging3D(
        x,
        y,
        z,
        vals,
        variogram_model=params.variogram_model,
        variogram_parameters=kriging_variogram_parameters(params),  # sill 1, nugget + jitter
        anisotropy_scaling_y=params.anisotropy_scaling_y,
        anisotropy_scaling_z=params.anisotropy_scaling_z,
        drift_terms=["regional_linear"],
    )

    # PyKrige's 3D Universal Kriging has no moving window: always all points.
    scalar_field, _ = uk3d.execute(
        "grid",
        grid.gridx,
        grid.gridy,
        grid.gridz,
        backend="vectorized",
    )

    return scalar_field, scalar_values_by_element
