"""
Universal Kriging (3D) interpolation for a single structural group.

This mirrors the API and behavior of `interpolate_group_ordinary_kriging` but uses
PyKrige's 3D Universal Kriging implementation.

Key behavior (same as the OK version):
- assigns strictly increasing scalar values per element (oldest=1 .. youngest=n),
- fits a 3D kriging model to surface points,
- evaluates the scalar field on a regular grid,
- does NOT mutate the passed `group`.

Universal Kriging additionally supports a deterministic "drift" (trend) via
`drift_terms` (e.g., "regional_linear") and related optional parameters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeAlias
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pykrige.uk3d import UniversalKriging3D

FloatArray: TypeAlias = npt.NDArray[np.floating]
ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]


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
          Minimum required attributes: variogram_model, sill, range, nugget,
          anisotropy_scaling_y, anisotropy_scaling_z, neighbors.
          Optional UK-specific: drift_terms (default "regional_linear"),
          specified_drift_arrays, external_drift, external_drift_grid.
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

    drift_terms = getattr(params, "drift_terms", None)
    if drift_terms is None:
        drift_terms = "regional_linear"
    if isinstance(drift_terms, (tuple, set)):
        drift_terms = list(drift_terms)

    specified_drift_arrays = getattr(params, "specified_drift_arrays", None)
    specified_drift = getattr(params, "specified_drift", None)
    if specified_drift_arrays is None and specified_drift is not None:
        if isinstance(specified_drift, dict):
            specified_drift_arrays = list(specified_drift.values())
        else:
            specified_drift_arrays = [specified_drift]

    external_drift = getattr(params, "external_drift", None)
    external_drift_grid = getattr(params, "external_drift_grid", None)

    uk3d_kwargs = dict(
        variogram_model=params.variogram_model,
        variogram_parameters=[params.sill, params.range, params.nugget],
        anisotropy_scaling_y=params.anisotropy_scaling_y,
        anisotropy_scaling_z=params.anisotropy_scaling_z,
        drift_terms=drift_terms,
    )
    if specified_drift_arrays is not None:
        uk3d_kwargs["specified_drift"] = specified_drift_arrays
    if external_drift is not None:
        uk3d_kwargs["external_drift"] = external_drift

    uk3d = UniversalKriging3D(x, y, z, vals, **uk3d_kwargs)

    backend = "loop" if params.neighbors is not None else "vectorized"

    execute_kwargs = dict(
        backend=backend
    )

    if external_drift_grid is not None:
        execute_kwargs["external_drift"] = external_drift_grid

    scalar_field, _ = uk3d.execute(
        "grid",
        grid.gridx,
        grid.gridy,
        grid.gridz,
        **execute_kwargs,
    )

    return scalar_field, scalar_values_by_element
