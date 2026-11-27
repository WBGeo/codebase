"""
LoopStructural FDI interpolation for a single structural group.

This module provides a pure interpolator function that:
- assigns strictly increasing scalar values per element (oldest=1 .. youngest=n),
- builds a LoopStructural GeologicalModel from surface points + orientations,
- creates an FDI foliation (finite-difference interpolator),
- evaluates the feature value on a RegularGrid,
and a fit-only variant (``fit_group_finite_differences``) that returns the fitted
feature for evaluation at arbitrary points (octree grids).

Notes
-----
- The function does NOT mutate the passed `group`.
- `group.structural_elements` are assumed ordered youngest -> oldest (framework convention).
- The scalar field is returned in ZYX shape (reshaped to the grid resolution, then transposed).
- Orientations only constrain the direction of the gradient at their location; they
  carry no scalar value, so they may lie anywhere, not only on an interface. With
  only one formation (all values equal), LoopStructural turns them into normal-vector
  constraints, which also fix the gradient's length and so define which way is up.

FDI-specific parameters
-----------------------
All LoopStructural creation/evaluation parameters are sourced from
`group.get_interpolation_params()`.

Expected attributes on params
-----------------------------
- nelements: int | float
    Discretisation size passed to `create_and_add_foliation`.
- regularisation: float
    Weight of the smoothness term, passed as `regularisation`.
- orientation_weight: float
    Weight of the orientation constraints, passed as `gpw` (surface points: `cpw` = 1).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from LoopStructural import GeologicalModel

from core.structural_modeling_components.interpolator_functions.fitted_field import (
    Extent6,
    LoopStructuralFittedField,
    evaluate_loopstructural_feature,
)

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]


def interpolate_group_finite_differences(
    *,
    group: Any,  # StructuralGroup-like (name, structural_elements with .name, get_interpolation_params())
    grid: Any,  # RegularGrid-like (extent, resolution, grid_coordinates)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: pd.DataFrame,
) -> ScalarFieldAndValues:
    """
    LoopStructural FDI interpolation for a single structural group (pure function).

    Parameters
    ----------
    group
        StructuralGroup-like object. Expected members:
        - `name: str`
        - `structural_elements: Sequence[... with .name]`
        - `get_interpolation_params() -> params-like`
          The params object may provide FDI-specific attributes described in the module docstring.
    grid
        RegularGrid-like object. Expected members:
        - `extent: tuple[float, float, float, float, float, float]`
        - `resolution: tuple[int, int, int]`
        - `grid_coordinates: (N, 3) ndarray` (cell-center coordinates)
    group_surface_points_df
        DataFrame with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        DataFrame with columns ["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"].

    Returns
    -------
    scalar_field : np.ndarray
        Evaluated scalar field on the grid in ZYX shape (shaped to `tuple(grid.resolution)`,
        then transposed).
    scalar_values_by_element : dict[str, float]
        Mapping element_name -> scalar value. Values are strictly increasing from
        oldest=1 to youngest=n (group.structural_elements is assumed youngest->oldest,
        so reversed order is used here).

    Raises
    ------
    ValueError
        If surface/orientation inputs are missing/empty or required columns are absent,
        or if the DataFrames include formations not present in the group.
    """
    model, scalar_values_by_element = _fit_fdi_model(
        group, grid.extent, group_surface_points_df, group_orientations_points_df)

    # Evaluate on grid
    # Same NaN guard as PLI, so both LoopStructural interpolators evaluate alike.
    regular_grid: FloatArray = grid.grid_coordinates  # (N, 3)
    sf_flat = evaluate_loopstructural_feature(model, group.name, regular_grid, scale=True)

    # grid coordinates are ordered with x slowest: reshape to (nx, ny, nz), transpose to ZYX
    scalar_field = np.asarray(sf_flat).reshape(tuple(grid.resolution)).T

    return scalar_field, scalar_values_by_element


def fit_group_finite_differences(
    *,
    group: Any,  # StructuralGroup-like (name, structural_elements with .name, get_interpolation_params())
    extent: Extent6,
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: pd.DataFrame,
) -> LoopStructuralFittedField:
    """
    Fit the FDI scalar field of a single structural group without evaluating it.

    Same inputs, checks and scalar values as :func:`interpolate_group_finite_differences`.
    LoopStructural builds its support mesh over ``extent``, so the field depends on it:
    with the extent of a regular grid, evaluating the result at its ``grid_coordinates``
    gives that function's scalar field.

    Parameters
    ----------
    group
        StructuralGroup-like object, as for :func:`interpolate_group_finite_differences`.
    extent
        Model extent ``(xmin, xmax, ymin, ymax, zmin, zmax)`` the support mesh covers.
    group_surface_points_df
        DataFrame with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        DataFrame with columns ["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"].

    Returns
    -------
    LoopStructuralFittedField
        The fitted feature and the element scalar values.
    """
    model, scalar_values_by_element = _fit_fdi_model(
        group, extent, group_surface_points_df, group_orientations_points_df)
    return LoopStructuralFittedField(model, group.name, True, scalar_values_by_element)


def _fit_fdi_model(
    group: Any,
    extent: Extent6,
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: pd.DataFrame,
) -> Tuple[GeologicalModel, Dict[str, float]]:
    """Check the inputs and fit the FDI foliation of the group over ``extent`` (shared by both entry points)."""
    # --- validation ---
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")
    if group_orientations_points_df is None or group_orientations_points_df.empty:
        raise ValueError(f"No orientations provided for group '{group.name}'")

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")
    for col in ("X", "Y", "Z", "G_x", "G_y", "G_z", "formation"):
        if col not in group_orientations_points_df.columns:
            raise ValueError(f"Orientations for '{group.name}' missing column '{col}'")

    # Ensure all formations in the DFs belong to this group
    group_elem_names = [e.name for e in group.structural_elements]
    unknown_sp = set(group_surface_points_df["formation"].unique()) - set(group_elem_names)
    unknown_ori = set(group_orientations_points_df["formation"].unique()) - set(group_elem_names)
    if unknown_sp:
        raise ValueError(
            f"Surface points for group '{group.name}' contain formations not in the group: {sorted(unknown_sp)}"
        )
    if unknown_ori:
        raise ValueError(
            f"Orientations for group '{group.name}' contain formations not in the group: {sorted(unknown_ori)}"
        )

    # --- per-element scalar values (oldest=1 ... youngest=n) ---
    names_old_to_young = [e.name for e in reversed(group.structural_elements)]
    scalar_values_by_element: Dict[str, float] = {
        name: float(i) for i, name in enumerate(names_old_to_young, start=1)
    }

    # Map formation name -> scalar ('val') for LoopStructural
    formation_to_scalar = scalar_values_by_element

    # --- build LoopStructural input tables ---
    # Surface points table
    surface_points = group_surface_points_df.copy()
    surface_points["feature_name"] = group.name
    surface_points["val"] = surface_points["formation"].map(formation_to_scalar)

    # LoopStructural expects columns: X, Y, Z, val, feature_name, gx, gy, gz.
    surface_points = surface_points[["X", "Y", "Z", "val", "feature_name"]]
    surface_points["gx"] = np.nan
    surface_points["gy"] = np.nan
    surface_points["gz"] = np.nan

    # Orientations table
    orientations = group_orientations_points_df.copy()
    orientations["feature_name"] = group.name
    # No value: orientations may lie anywhere, not only on the interface of their formation
    # (the formation only assigns them to the group). LoopStructural turns every row with a
    # `val` into a value constraint, which would pull the field to that formation's value
    # at the orientation's location.
    orientations["val"] = np.nan
    orientations = orientations.rename(columns={"G_x": "gx", "G_y": "gy", "G_z": "gz"})
    orientations = orientations[["X", "Y", "Z", "val", "feature_name", "gx", "gy", "gz"]]

    # Combined table
    data_combined = pd.concat([surface_points, orientations], ignore_index=True)

    # --- build LoopStructural model ---
    # GeologicalModel(min_bounds, max_bounds) expects 3-vectors
    model = GeologicalModel(np.array(extent[::2]), np.array(extent[1::2]))
    if hasattr(model, "data"):
        model.data = data_combined
    else:
        # backward compatibility
        model.set_model_data(data_combined)

    # No stratigraphic column is set: LoopStructural only reads it in
    # evaluate_model()/fault topology, neither of which is used here -- we only
    # evaluate this one feature's scalar field and assign lithology ourselves.

    # --- parameters (FDI) ---
    params = group.get_interpolation_params() if hasattr(group, "get_interpolation_params") else None
    if params is None:
        raise ValueError(f"Missing interpolation params for group '{group.name}' (required for FDI)")

    # Create the foliation / feature (FDI)
    #
    # buffer=0.2 (LoopStructural's own default, same as piecewise_linear.py):
    # measured against buffer=0.0 under LoopStructural 1.8.2 on a folded
    # surface at a non-zero origin, interface accuracy is unchanged (interior
    # and domain-edge errors within ~0.5 m), but the solve is 3-5x faster.
    create_kwargs: Dict[str, Any] = dict(
        interpolatortype="FDI",
        nelements=params.nelements,
        buffer=0.2,
        # Conjugate gradients with LoopStructural's default tolerance (not exposed).
        solver="cg",
        # Least-squares weights: surface points 1 (LoopStructural default), orientations
        # (direction only) `orientation_weight`, smoothness `regularisation`.
        regularisation=params.regularisation,
        gpw=params.orientation_weight,
    )

    _ = model.create_and_add_foliation(group.name, **create_kwargs)

    return model, scalar_values_by_element
