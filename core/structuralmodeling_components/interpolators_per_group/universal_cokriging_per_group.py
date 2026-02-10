"""
Universal Co-Kriging (via GemPy) interpolation for a single structural group.

This module provides a pure interpolator function that:
- builds GemPy SurfacePoints/Orientations tables from the given DataFrames,
- computes a GemPy model for a single series (the group),
- extracts the scalar field and per-surface scalar values,
- rescales the scalar field so that element isosurfaces map to 1..n.

Notes
-----
- The function does NOT mutate the passed `group`.
- `group.structural_elements` are assumed ordered youngest -> oldest (framework convention).
- Axis order / transposes are preserved exactly as implemented.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import gempy as gp

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]


def interpolate_group_universal_cokriging(
    *,
    group: Any,  # expected: StructuralGroup-like (group.name, group.structural_elements with .name)
    grid: Any,  # expected: RegularGrid-like (grid.extent, grid.resolution)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: pd.DataFrame,
) -> ScalarFieldAndValues:
    """
    Universal Co-Kriging (via GemPy) for a single structural group (pure function).

    Parameters
    ----------
    group
        StructuralGroup-like object. Expected members:
        - `name: str`
        - `structural_elements: Sequence[... with .name]`
    grid
        RegularGrid-like object. Expected members:
        - `extent: tuple[float, float, float, float, float, float]`
        - `resolution: tuple[int, int, int]`
    group_surface_points_df
        DataFrame with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        DataFrame with columns ["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"].

    Returns
    -------
    scalar_field : np.ndarray
        Rescaled scalar field. Reshape/transposes are preserved from the original code.
    scalar_values_by_element : dict[str, float]
        Rescaled scalar values mapping `element_name -> isovalue` (targets 1..n).

    Raises
    ------
    ValueError
        If required input data is missing/empty, or expected columns are absent,
        or the DataFrames include formations not in the group.
    RuntimeError
        If GemPy returns an unexpected number of scalar values at surface points.
    """
    # ---- validation ----
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

    # Ensure the formations in the DFs belong to this group
    group_elem_names = [e.name for e in group.structural_elements]
    unknown_sp = set(group_surface_points_df["formation"].unique()) - set(group_elem_names)
    if unknown_sp:
        raise ValueError(
            f"Surface points for group '{group.name}' contain formations not in the group: {sorted(unknown_sp)}"
        )
    unknown_ori = set(group_orientations_points_df["formation"].unique()) - set(group_elem_names)
    if unknown_ori:
        raise ValueError(
            f"Orientations for group '{group.name}' contain formations not in the group: {sorted(unknown_ori)}"
        )

    # ---- build GemPy data tables ----
    surface_data = gp.data.surface_points.SurfacePointsTable.from_arrays(
        x=group_surface_points_df["X"].to_numpy(),
        y=group_surface_points_df["Y"].to_numpy(),
        z=group_surface_points_df["Z"].to_numpy(),
        names=group_surface_points_df["formation"].to_numpy(),
        nugget=np.zeros(len(group_surface_points_df)),
        name_id_map=None,
    )

    orientation_data = gp.data.orientations.OrientationsTable.from_arrays(
        x=group_orientations_points_df["X"].to_numpy(),
        y=group_orientations_points_df["Y"].to_numpy(),
        z=group_orientations_points_df["Z"].to_numpy(),
        G_x=group_orientations_points_df["G_x"].to_numpy(),
        G_y=group_orientations_points_df["G_y"].to_numpy(),
        G_z=group_orientations_points_df["G_z"].to_numpy(),
        names=group_orientations_points_df["formation"].to_numpy(),
        nugget=np.zeros(len(group_orientations_points_df)),
        name_id_map=surface_data.name_id_map,
    )

    gp_structural_frame = gp.data.structural_frame.StructuralFrame.from_data_tables(
        surface_data, orientation_data
    )

    # ---- create model ----
    # NOTE: GemPy typing expects list/ndarray; convert tuples at call site only.
    geo_model = gp.create_geomodel(
        project_name="uk_group",
        extent=list(grid.extent),
        resolution=list(grid.resolution),
        structural_frame=gp_structural_frame,
    )

    # Map one series (group) to its surfaces (elements)
    series_to_surfaces: Dict[str, list[str]] = {group.name: group_elem_names}
    gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=series_to_surfaces)

    # ---- compute ----
    gp.compute_model(geo_model)

    # ---- fetch results ----
    raw_scalar_vals = geo_model.solutions.raw_arrays.scalar_field_at_surface_points[0]
    if len(raw_scalar_vals) != len(group_elem_names):
        raise RuntimeError(
            f"GemPy returned {len(raw_scalar_vals)} scalar values but group '{group.name}' has "
            f"{len(group_elem_names)} elements."
        )
    scalar_values_by_element: Dict[str, float] = {
        name: float(val) for name, val in zip(group_elem_names, raw_scalar_vals)
    }

    raw_scalar_field = geo_model.solutions.raw_arrays.scalar_field_matrix[0]
    scalar_field = np.asarray(raw_scalar_field, dtype=float).reshape(tuple(grid.resolution)).T

    # ---- rescale scalar field so that element isosurfaces map exactly to 1..n ----
    # Original code keeps affine_rescale_scalar_field commented out; preserved.
    rescaled_scalar_field, rescaled_scalar_values_by_element = rescale_scalar_field_safe(
        scalar_field=scalar_field,
        scalar_values_by_element=scalar_values_by_element,
    )

    return rescaled_scalar_field, rescaled_scalar_values_by_element


def affine_rescale_scalar_field(
    scalar_field: np.ndarray,
    scalar_values_by_element: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Affine rescaling of a scalar field so that element isosurfaces map exactly to 1..n
    WITHOUT clipping the field.

    Parameters
    ----------
    scalar_field
        Scalar field array (any shape).
    scalar_values_by_element
        Mapping element_name -> scalar value extracted at the corresponding surface points.

    Returns
    -------
    scalar_field_rescaled
        Rescaled scalar field.
    scalar_values_rescaled
        Element scalar values after applying the same affine transform.
    """
    # Sort elements by scalar value
    items_sorted = sorted(
        scalar_values_by_element.items(),
        key=lambda x: x[1],
    )

    old_vals = np.array([v for _, v in items_sorted], dtype=float)
    n = len(old_vals)
    new_vals = np.arange(1, n + 1, dtype=float)

    if n < 2:
        raise ValueError("Need at least two elements for affine rescaling")

    # Use first and last element as anchors
    s1, s2 = old_vals[0], old_vals[-1]
    t1, t2 = new_vals[0], new_vals[-1]

    a = (t2 - t1) / (s2 - s1)
    b = t1 - a * s1

    # Apply globally
    scalar_field_rescaled = a * scalar_field + b

    # New scalar values per element
    scalar_values_rescaled: Dict[str, float] = {
        name: float(a * val + b)
        for name, val in scalar_values_by_element.items()
    }

    return scalar_field_rescaled, scalar_values_rescaled


def rescale_scalar_field_safe(
    scalar_field: np.ndarray,
    scalar_values_by_element: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Rescale a scalar field so that element isosurfaces map to 1..n.

    This uses a piecewise affine transform so that:
      - for n == 1: a pure shift makes the only surface map to 1
      - for n >= 2: each interval between consecutive surfaces is rescaled independently

    Parameters
    ----------
    scalar_field
        Scalar field array (any shape).
    scalar_values_by_element
        Mapping element_name -> scalar value extracted at the corresponding surface points.

    Returns
    -------
    S_new
        Rescaled scalar field.
    scalar_values_rescaled
        Target scalar values by element (exactly 1..n in sorted order).
    """
    items = sorted(
        scalar_values_by_element.items(),
        key=lambda x: x[1],
    )

    old_vals = np.array([v for _, v in items], dtype=float)
    n = len(old_vals)

    if n == 0:
        raise ValueError("No elements to rescale")

    # ------------------
    # n = 1 → pure shift
    # ------------------
    if n == 1:
        s0 = old_vals[0]
        shift = 1.0 - s0
        S_new = scalar_field + shift

        scalar_values_rescaled: Dict[str, float] = {
            items[0][0]: 1.0
        }

        return S_new, scalar_values_rescaled

    # ------------------
    # n >= 2 → piecewise affine
    # ------------------
    new_vals = np.arange(1, n + 1, dtype=float)

    slopes = np.diff(new_vals) / np.diff(old_vals)
    slope_lo = slopes[0]
    slope_hi = slopes[-1]

    S = scalar_field
    S_new = np.empty_like(S, dtype=float)

    # Below first surface
    mask = S <= old_vals[0]
    S_new[mask] = new_vals[0] + slope_lo * (S[mask] - old_vals[0])

    # Between surfaces
    for i in range(n - 1):
        lo, hi = old_vals[i], old_vals[i + 1]
        mask = (S > lo) & (S <= hi)
        S_new[mask] = new_vals[i] + slopes[i] * (S[mask] - lo)

    # Above last surface
    mask = S > old_vals[-1]
    S_new[mask] = new_vals[-1] + slope_hi * (S[mask] - old_vals[-1])

    scalar_values_rescaled: Dict[str, float] = {
        name: float(val)
        for name, val in zip([k for k, _ in items], new_vals)
    }

    return S_new, scalar_values_rescaled
