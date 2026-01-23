import numpy as np
import pandas as pd
import gempy as gp
from typing import Dict, Tuple

def interpolate_group_universal_cokriging(
    *,
    group,                       # StructuralGroup
    grid,                        # RegularGrid (has extent, resolution)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Universal Co-Kriging (via GemPy) for a single structural group (pure function).

    Returns:
        scalar_field : np.ndarray with shape == tuple(grid.resolution)  (nx, ny, nz)
        scalar_values_by_element : Dict[str, float]  (element_name -> scalar value at surface points)

    Notes:
        - Does NOT mutate `group`.
        - Assumes `group.structural_elements` are ordered youngest -> oldest (as in your framework).
        - We build the GemPy StructuralFrame from the provided dataframes, map a single series
          (this group's name) to its surfaces (element names), compute the model, and then
          fetch the scalar field and per-surface scalar values.
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

    # ---- build GemPy input_data tables ----
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
    geo_model = gp.create_geomodel(
        project_name="uk_group",
        extent=np.asarray(grid.extent, dtype=float),
        resolution=np.asarray(grid.resolution, dtype=int),
        structural_frame=gp_structural_frame,
    )

    # Map one series (group) to its surfaces (elements)
    series_to_surfaces = {group.name: group_elem_names}
    gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=series_to_surfaces)

    # ---- compute ----
    gp.compute_model(geo_model)

    # ---- fetch results ----
    # scalar values per surface (GemPy returns one per surface in the order of surfaces in the series)
    # Indexing [0] is the first series; then [i] for each surface in that series
    raw_scalar_vals = geo_model.solutions.raw_arrays.scalar_field_at_surface_points[0]
    if len(raw_scalar_vals) != len(group_elem_names):
        raise RuntimeError(
            f"GemPy returned {len(raw_scalar_vals)} scalar values but group '{group.name}' has "
            f"{len(group_elem_names)} elements."
        )
    scalar_values_by_element = {
        name: float(val) for name, val in zip(group_elem_names, raw_scalar_vals)
    }

    # The scalar field (flattened) for this series is typically in scalar_field_matrix[0]
    raw_scalar_field = geo_model.solutions.raw_arrays.scalar_field_matrix[0]
    scalar_field = np.asarray(raw_scalar_field, dtype=float).reshape(tuple(grid.resolution)).T

    # ---- rescale scalar field so that element isosurfaces map exactly to 1, 2, ..., n ----

    # This affine transform does not really work if n < 3, as it distorts the spacing.
    # rescaled_scalar_field, rescaled_scalar_values_by_element = affine_rescale_scalar_field(
    #     scalar_field=scalar_field,
    #     scalar_values_by_element=scalar_values_by_element,
    # )

    # TODO: Check if this works for n>3 correctly
    rescaled_scalar_field, rescaled_scalar_values_by_element = rescale_scalar_field_safe(
        scalar_field=scalar_field,
        scalar_values_by_element=scalar_values_by_element,
    )

    return rescaled_scalar_field, rescaled_scalar_values_by_element


def affine_rescale_scalar_field(
    scalar_field: np.ndarray,
    scalar_values_by_element: dict,
):
    """
    Affine rescaling of a scalar field so that element isosurfaces
    map exactly to 1, 2, ..., n WITHOUT clipping the field.
    """

    # Sort elements by scalar value
    items_sorted = sorted(
        scalar_values_by_element.items(),
        key=lambda x: x[1]
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
    scalar_values_rescaled = {
        name: float(a * val + b)
        for name, val in scalar_values_by_element.items()
    }

    return scalar_field_rescaled, scalar_values_rescaled

def rescale_scalar_field_safe(
    scalar_field: np.ndarray,
    scalar_values_by_element: dict,
):
    """
    Rescale scalar field so that element isosurfaces map to 1..n.
    Handles n=1, n=2, n>=3 correctly.
    """

    items = sorted(
        scalar_values_by_element.items(),
        key=lambda x: x[1]
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

        scalar_values_rescaled = {
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

    scalar_values_rescaled = {
        name: float(val)
        for name, val in zip(
            [k for k, _ in items],
            new_vals
        )
    }

    return S_new, scalar_values_rescaled